import torch
from comfy_api.latest import io
import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention


def _nag_mix(z_pos, z_neg, nag_scale, nag_alpha, nag_tau):
    guided = z_pos * nag_scale - z_neg * (nag_scale - 1.0)
    eps = 1e-6
    norm_pos = torch.norm(z_pos, p=1, dim=-1, keepdim=True).clamp_min(eps)
    norm_guided = torch.norm(guided, p=1, dim=-1, keepdim=True).clamp_min(eps)
    ratio = norm_guided / norm_pos
    guided = guided * (torch.minimum(ratio, torch.full_like(ratio, nag_tau)) / ratio)
    return guided * nag_alpha + z_pos * (1.0 - nag_alpha)


def _cross_attn(attn, q, context, mask, transformer_options):
    k = attn.k_norm(attn.to_k(context)).to(dtype=q.dtype)
    v = attn.to_v(context).to(dtype=q.dtype)
    return optimized_attention(
        q, k, v, attn.heads,
        mask=mask,
        attn_precision=attn.attn_precision,
        transformer_options=transformer_options,
    )


def _make_nag_forward(attn, nag_context, nag_scale, nag_alpha, nag_tau):
    def forward(x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
        if context is None:
            context = x

        if context.shape[0] == 1:
            x_pos, x_cfg = x, None
            context_pos = context
            context_cfg = None
        else:
            x_pos, x_cfg = torch.chunk(x, 2, dim=0)
            context_pos, context_cfg = torch.chunk(context, 2, dim=0)

        q_pos = attn.q_norm(attn.to_q(x_pos))
        nag_ctx = nag_context.to(device=q_pos.device, dtype=q_pos.dtype)
        if nag_ctx.shape[0] == 1 and q_pos.shape[0] != 1:
            nag_ctx = nag_ctx.expand(q_pos.shape[0], -1, -1)

        z_pos = _cross_attn(attn, q_pos, context_pos, mask, transformer_options)
        z_neg = _cross_attn(attn, q_pos, nag_ctx, None, transformer_options)
        out = _nag_mix(z_pos, z_neg, nag_scale, nag_alpha, nag_tau)

        if x_cfg is not None:
            q_cfg = attn.q_norm(attn.to_q(x_cfg))
            out_cfg = _cross_attn(attn, q_cfg, context_cfg, mask, transformer_options)
            out = torch.cat([out, out_cfg], dim=0)

        if attn.to_gate_logits is not None:
            gate_logits = attn.to_gate_logits(x)
            b, t, _ = out.shape
            out = out.view(b, t, attn.heads, attn.dim_head)
            out = out * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
            out = out.view(b, t, attn.heads * attn.dim_head)

        return attn.to_out(out)

    return forward


def _move_modules(dm, device):
    moved = []
    for name in (
        "caption_projection",
        "audio_caption_projection",
        "video_embeddings_connector",
        "audio_embeddings_connector",
    ):
        mod = getattr(dm, name, None)
        if isinstance(mod, torch.nn.Module):
            mod.to(device)
            moved.append(mod)
    return moved


def _prepare_nag_context(dm, nag_cond, device, dtype, offload_device):
    """Match LTX2 sampling: preprocess_text_embeds then video half of _prepare_context."""
    context = nag_cond[0][0].to(device=device, dtype=dtype)
    extra = nag_cond[0][1] if len(nag_cond[0]) > 1 else {}
    unprocessed = bool(extra.get("unprocessed_ltxav_embeds", False))

    moved = _move_modules(dm, device)
    try:
        processed = dm.preprocess_text_embeds(context, unprocessed=unprocessed)
        vid_dim = dm.caption_channels if not dm.caption_proj_before_connector else dm.inner_dim
        v_context = processed[..., :vid_dim]
        if not dm.caption_proj_before_connector:
            v_context = dm.caption_projection(v_context)
        return v_context.reshape(1, -1, dm.inner_dim).contiguous()
    finally:
        for mod in moved:
            mod.to(offload_device)


class Nag(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DA_Nag",
            display_name="NAG (LTX2)",
            description="LTX2 Normalized Attention Guidance. Encode a negative prompt (e.g. subtitles, text, watermark) and connect it to nag_cond.",
            category="DA_Nodes",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("nag_cond", tooltip="Negative prompt after CLIP Text Encode."),
                io.Float.Input("nag_scale", default=11.0, min=0.0, max=100.0, step=0.001, tooltip="Guidance strength. Higher pushes farther from nag_cond."),
                io.Float.Input("nag_alpha", default=0.25, min=0.0, max=1.0, step=0.001, tooltip="Blend with original positive attention. 1.0 = full NAG."),
                io.Float.Input("nag_tau", default=2.5, min=0.0, max=10.0, step=0.001, tooltip="L1 clamp; limits how far guided attention can deviate."),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, nag_cond, nag_scale, nag_alpha, nag_tau) -> io.NodeOutput:
        if nag_scale == 0:
            return io.NodeOutput(model)

        dm = model.get_model_object("diffusion_model")
        if not hasattr(dm, "preprocess_text_embeds") or not hasattr(dm, "transformer_blocks"):
            raise RuntimeError("DA_Nag only supports LTX2 models.")

        device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        dtype = model.model.get_dtype_inference()

        model_clone = model.clone()
        dm = model_clone.get_model_object("diffusion_model")
        nag_context = _prepare_nag_context(dm, nag_cond, device, dtype, offload_device)

        for i, block in enumerate(dm.transformer_blocks):
            model_clone.add_object_patch(
                f"diffusion_model.transformer_blocks.{i}.attn2.forward",
                _make_nag_forward(block.attn2, nag_context, nag_scale, nag_alpha, nag_tau),
            )

        return io.NodeOutput(model_clone)
