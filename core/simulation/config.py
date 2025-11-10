"""
仿真主循环的关键参数，例如 tick、时长与门控配置。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.isolation.policies import SandboxConfig
from core.scheduling.config import SchedulingConfig


@dataclass
class SimulationConfig:
    """统一描述主循环节奏、调度策略与沙盒能力。"""
    delta_t: float = 0.01  # seconds per tick
    duration: float = 600.0
    scheduling_interval: int = 5  # ticks
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
