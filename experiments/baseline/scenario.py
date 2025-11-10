"""
A1：无 API 沙盒，仅 GLB 调度的基线。
"""

from core.isolation.policies import SandboxConfig, SLOGuardConfig
from core.scheduling.config import SchedulingConfig
from core.simulation.config import SimulationConfig

from experiments.base import ExperimentProfile


SCENARIO = ExperimentProfile(
    name="A1-baseline",
    description="无沙盒 + 共享池，仅测试GLB调度对利用率与干扰的影响",
    simulation=SimulationConfig(
        delta_t=0.02,
        duration=320.0,
        scheduling_interval=4,
        scheduling=SchedulingConfig(static_partition=False, enable_sharing=True, oversubscription=1.0),
        sandbox=SandboxConfig(
            enable_memory_gate=False,
            enable_bandwidth_gate=False,
            enable_compute_gate=False,
            slo_guard=SLOGuardConfig(enabled=False),
        ),
    ),
    num_tasks=48,
    arrival_mode="poisson",
)
