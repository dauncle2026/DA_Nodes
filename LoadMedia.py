import os

import folder_paths
from comfy_api.latest import io
from comfy_extras.nodes_audio import LoadAudio
from comfy_extras.nodes_video import LoadVideo
from nodes import LoadImage

IMAGE_COUNT = 10
AUDIO_COUNT = 5
VIDEO_COUNT = 5


def _filename(value):
    if not value:
        return None
    name = str(value).strip()
    return name or None


def _slot_names(prefix, count):
    return [f"{prefix}_{i}" for i in range(1, count + 1)]


class LoadMedia(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = []
        outputs = []
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
            node_id="DA_LoadMedia",
            display_name="Load Media",
            category="DA_Nodes",
            description="Load images, audio, and video by filename from the input folder. Video slots decode to frames and a paired soundtrack. Empty slots output nothing and do not fail validation.",
            search_aliases=["load image", "load video", "load audio", "load media", "multiple images"],
            inputs=inputs,
            outputs=outputs,
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        for value in kwargs.values():
            name = _filename(value)
            if name is None:
                continue
            if not folder_paths.exists_annotated_filepath(name):
                return "Invalid file: {}".format(name)
        return True

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        parts = []
        for key in _slot_names("image", IMAGE_COUNT) + _slot_names("audio", AUDIO_COUNT) + _slot_names("video", VIDEO_COUNT):
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
    def execute(cls, **kwargs) -> io.NodeOutput:
        results = []
        loader = LoadImage()
        for name in _slot_names("image", IMAGE_COUNT):
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
