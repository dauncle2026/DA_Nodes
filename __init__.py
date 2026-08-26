import os
import sys
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

# ComfyUI loads custom node folders via spec_from_file_location without package
# context. Register this folder so sibling modules can use relative imports.
_DIR = os.path.dirname(os.path.abspath(__file__))
__path__ = [_DIR]
__package__ = "DA_Nodes"
sys.modules.setdefault("DA_Nodes", sys.modules[__name__])

from .LoadMedia import LoadMedia
from .ReferenceLatents import ReferenceLatents


class DANodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadMedia,
            ReferenceLatents,
        ]


async def comfy_entrypoint() -> DANodesExtension:
    return DANodesExtension()
