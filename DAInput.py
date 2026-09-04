import os

import folder_paths
from comfy_api.latest import io
from comfy_extras.nodes_audio import LoadAudio
from comfy_extras.nodes_resolution import AspectRatio
from comfy_extras.nodes_video import LoadVideo
from nodes import LoadImage

IMAGE_COUNT = 10
AUDIO_COUNT = 5
VIDEO_COUNT = 5
_FRAME_SLOTS = ("first_frame", "last_frame")


def _filename(value):
    if not value:
        return None
    name = str(value).strip()
    return name or None


def _slot_names(prefix, count):
    return [f"{prefix}_{i}" for i in range(1, count + 1)]


def _file_slots():
    return list(_FRAME_SLOTS) + _slot_names("image", IMAGE_COUNT) + _slot_names("audio", AUDIO_COUNT) + _slot_names("video", VIDEO_COUNT)


class DAInput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.String.Input("prompt", multiline=True, default="", tooltip="Text prompt. Connect to CLIP Text Encode."),
            io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True, tooltip="Random seed. Connect to KSampler."),
            io.Combo.Input("aspect_ratio", options=AspectRatio, default=AspectRatio.SQUARE, tooltip="Connect to Resolution Selector."),
            io.Float.Input("resolution", default=1.0, min=0.1, max=16.0, step=0.1, tooltip="Target megapixels. Connect to Resolution Selector. 1.0 ≈ 1024x1024 for square."),
            io.Float.Input("duration", default=5.0, min=0.0, step=0.01, tooltip="Duration in seconds."),
        ]
        outputs = [
            io.String.Output("prompt", display_name="prompt"),
            io.Int.Output("seed", display_name="seed"),
            io.Combo.Output("aspect_ratio", display_name="aspect_ratio", options=[v.value for v in AspectRatio]),
            io.Float.Output("resolution", display_name="resolution"),
            io.Float.Output("duration", display_name="duration"),
        ]
        for name in _FRAME_SLOTS:
            inputs.append(io.String.Input(name, optional=True, default="", tooltip="Filename in the input folder. Leave empty to skip."))
            outputs.append(io.Image.Output(name, display_name=name))
        for name in _slot_names("image", IMAGE_COUNT):
            inputs.append(io.String.Input(name, optional=True, default="", tooltip="Filename in the input folder. Leave empty to skip."))
            outputs.append(io.Image.Output(name, display_name=name))
        for name in _slot_names("audio", AUDIO_COUNT):
            inputs.append(io.String.Input(name, optional=True, default="", tooltip="Filename in the input folder. Leave empty to skip."))
            outputs.append(io.Audio.Output(name, display_name=name))
        for name in _slot_names("video", VIDEO_COUNT):
            inputs.append(io.String.Input(name, optional=True, default="", tooltip="Filename in the input folder. Decoded to frames and soundtrack. Leave empty to skip."))
            outputs.append(io.Image.Output(name, display_name=name, tooltip="Video frames"))
        for name in _slot_names("video_audio", VIDEO_COUNT):
            outputs.append(io.Audio.Output(name, display_name=name, tooltip="Soundtrack of the same-numbered video"))
        return io.Schema(
            node_id="DA_Input",
            display_name="DA Input",
            category="DA_Nodes",
            description="Workflow entry node: prompt, seed, aspect ratio, resolution, duration, and optional image/audio/video files from the input folder.",
            search_aliases=["DAInput", "workflow input", "load media", "prompt", "seed"],
            inputs=inputs,
            outputs=outputs,
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        for key in _file_slots():
            name = _filename(kwargs.get(key))
            if name is None:
                continue
            if not folder_paths.exists_annotated_filepath(name):
                return "Invalid file: {}".format(name)
        return True

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        parts = []
        for key in _file_slots():
            name = _filename(kwargs.get(key))
            if name is None:
                parts.append(f"{key}=")
                continue
            if folder_paths.exists_annotated_filepath(name):
                path = folder_paths.get_annotated_filepath(name)
                parts.append(f"{key}={name}:{os.path.getmtime(path)}:{os.path.getsize(path)}")
            else:
                parts.append(f"{key}={name}")
        return ",".join(parts)

    @classmethod
    def execute(cls, prompt, seed, aspect_ratio, resolution, duration, first_frame="", last_frame="", **kwargs) -> io.NodeOutput:
        kwargs["first_frame"] = first_frame
        kwargs["last_frame"] = last_frame
        results = [prompt, seed, aspect_ratio, resolution, duration]
        loader = LoadImage()
        for name in list(_FRAME_SLOTS) + _slot_names("image", IMAGE_COUNT):
            filename = _filename(kwargs.get(name))
            if filename is None:
                results.append(None)
            else:
                image, _mask = loader.load_image(filename)
                results.append(image)
        for name in _slot_names("audio", AUDIO_COUNT):
            filename = _filename(kwargs.get(name))
            if filename is None:
                results.append(None)
            else:
                results.append(LoadAudio.execute(filename)[0])
        video_images = []
        video_audios = []
        for name in _slot_names("video", VIDEO_COUNT):
            filename = _filename(kwargs.get(name))
            if filename is None:
                video_images.append(None)
                video_audios.append(None)
            else:
                components = LoadVideo.execute(filename)[0].get_components()
                video_images.append(components.images)
                video_audios.append(components.audio)
        results.extend(video_images)
        results.extend(video_audios)
        return io.NodeOutput(*results)
