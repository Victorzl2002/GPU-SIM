"""
沙盒统一输出的 API 决策模型。
"""

from dataclasses import dataclass
from typing import Dict

from core.vgpu_model.resource_model.vgpu_resource import VGPUResource


@dataclass
class SandboxDecision:
    """记录每次限流的实际用量以及三门是否触发。"""

    task_id: str
    usage: VGPUResource
    limited: Dict[str, bool]
