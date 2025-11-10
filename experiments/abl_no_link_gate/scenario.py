"""
A4：在 A3 基础上移除带宽令牌桶的消融实验。
"""

from core.isolation.policies import SandboxConfig, SLOGuardConfig
from core.scheduling.config import SchedulingConfig
from core.simulation.config import SimulationConfig

from experiments.base import ExperimentProfile


SCENARIO = ExperimentProfile(
    name="A4-no-link-gate",
    description="API 沙盒但关闭带宽令牌桶，评估链路门的重要性",
    simulation=SimulationConfig(
        delta_t=0.01,
        duration=320.0,
        scheduling_interval=4,
        scheduling=SchedulingConfig(static_partition=False, enable_sharing=True, oversubscription=1.15),
        sandbox=SandboxConfig(
            enable_memory_gate=True,
            enable_bandwidth_gate=False,
            enable_compute_gate=True,
            slo_guard=SLOGuardConfig(enabled=True, adjust_interval=30, max_boost=1.35, decay=0.03),
            bandwidth_refill_rate=1.4,
            compute_ceiling=1.25,
        ),
    ),
    num_tasks=60,
    arrival_mode="burst",
)
