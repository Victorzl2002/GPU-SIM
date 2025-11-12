"""
A2：MIG 式硬切分，禁止共享与沙盒。
"""

from core.isolation.policies import SandboxConfig, SLOGuardConfig
from core.scheduling.config import SchedulingConfig
from core.simulation.config import SimulationConfig

from experiments.base import ExperimentProfile


SCENARIO = ExperimentProfile(
    name="A2-hard-split",
    description="静态分片（MIG近似），任务独占节点，评估利用率下限",
    simulation=SimulationConfig(
        delta_t=0.02,
        duration=320.0,
        scheduling_interval=6,
        scheduling=SchedulingConfig(static_partition=True, enable_sharing=False, oversubscription=1.0),
        sandbox=SandboxConfig(
            enable_memory_gate=False,
            enable_bandwidth_gate=False,
            enable_compute_gate=False,
            slo_guard=SLOGuardConfig(enabled=False),
        ),
    ),
    num_tasks=120,
    arrival_mode="burst",
)
