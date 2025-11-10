"""
GLB 调度器的关键开关。
"""

from dataclasses import dataclass


@dataclass
class SchedulingConfig:
    """统一描述两级调度的节奏、权重与共享策略。"""

    glb_interval: int = 10
    lambda_weight: float = 0.6
    static_partition: bool = False
    enable_sharing: bool = True
    oversubscription: float = 1.0
