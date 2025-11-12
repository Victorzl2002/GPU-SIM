"""
A5：关闭 SLO 守护反馈，仅保留 API 沙盒。
"""

from core.isolation.policies import SandboxConfig, SLOGuardConfig
from core.scheduling.config import SchedulingConfig
from core.simulation.config import SimulationConfig

from experiments.base import ExperimentProfile


SCENARIO = ExperimentProfile(
    name="A5-no-slo-guard",
    description="API 沙盒但关闭 SLO 守护反馈，观察尾部波动",
    simulation=SimulationConfig(
        delta_t=0.01,
        duration=320.0,
        scheduling_interval=4,
        scheduling=SchedulingConfig(static_partition=False, enable_sharing=True, oversubscription=1.05),
        sandbox=SandboxConfig(
            enable_memory_gate=True,
            enable_bandwidth_gate=True,
            enable_compute_gate=True,
            slo_guard=SLOGuardConfig(enabled=False),
            bandwidth_refill_rate=1.0,
            compute_ceiling=1.4,
        ),
    ),
    num_tasks=150,
    arrival_mode="burst",
)
