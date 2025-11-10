"""
沙盒与 SLO 守护的配置项定义。
"""

from dataclasses import dataclass, field


@dataclass
class SLOGuardConfig:
    """描述 SLO 守护在算力门上的调参幅度与节奏。"""

    enabled: bool = True
    adjust_interval: int = 50
    max_boost: float = 1.2
    decay: float = 0.02


@dataclass
class SandboxConfig:
    """控制三门开关、令牌桶参数与算力上限等核心选项。"""

    enable_memory_gate: bool = True
    enable_bandwidth_gate: bool = True
    enable_compute_gate: bool = True
    slo_guard: SLOGuardConfig = field(default_factory=SLOGuardConfig)
    bandwidth_refill_rate: float = 1.0
    compute_ceiling: float = 1.0
