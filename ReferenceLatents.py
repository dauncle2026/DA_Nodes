import node_helpers
from comfy_api.latest import io
from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
from nodes import VAEEncode

IMAGE_COUNT = 10


class ReferenceLatents(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Conditioning.Input("positive"),
            io.Conditioning.Input("negative", optional=True),
            io.Vae.Input("vae"),
            io.Combo.Input("upscale_method", options=ImageScaleToTotalPixels.upscale_methods, default="lanczos", optional=True),
            io.Float.Input("megapixels", default=1.0, min=0.01, max=16.0, step=0.01, optional=True),
            io.Int.Input("resolution_steps", default=1, min=1, max=256, optional=True, advanced=True),
        ]
        for i in range(1, IMAGE_COUNT + 1):
            name = f"image_{i}"
            inputs.append(io.Image.Input(name, optional=True))
        return io.Schema(
            node_id="DA_ReferenceLatents",
            display_name="Set Reference Latents",
            category="DA_Nodes",
            description="Scale, VAE-encode, and append each connected image as a reference latent. Empty image slots are skipped, so 1-N images can share one workflow.",
            search_aliases=["reference latent", "set reference latent", "flux2 reference", "multi reference"],
            inputs=inputs,
            outputs=[
                io.Conditioning.Output("positive", display_name="positive"),
                io.Conditioning.Output("negative", display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, positive, vae, upscale_method="lanczos", megapixels=1.0, resolution_steps=1, negative=None, **kwargs) -> io.NodeOutput:
        encoder = VAEEncode()
        ref_latents = []
        for i in range(1, IMAGE_COUNT + 1):
            image = kwargs.get(f"image_{i}")
            if image is None:
                continue
            pixels = ImageScaleToTotalPixels.execute(image, upscale_method, megapixels, resolution_steps)[0]
            encoded = encoder.encode(vae, pixels)[0]
            ref_latents.append(encoded["samples"])

        if ref_latents:
            values = {"reference_latents": ref_latents}
            positive = node_helpers.conditioning_set_values(positive, values, append=True)
            if negative is not None:
                negative = node_helpers.conditioning_set_values(negative, values, append=True)

        return io.NodeOutput(positive, negative)
