"""
A2P：ParvaGPU 风格固定分片 + 局部共享。
"""

from core.isolation.policies import SandboxConfig, SLOGuardConfig
from core.scheduling.config import SchedulingConfig
from core.simulation.config import SimulationConfig

from experiments.base import ExperimentProfile


SCENARIO = ExperimentProfile(
    name="A2P-parvagpu",
    description="固定切片 + 局部共享 (ParvaGPU 近似)，无API沙盒",
    simulation=SimulationConfig(
        delta_t=0.02,
        duration=320.0,
        scheduling_interval=6,
        scheduling=SchedulingConfig(static_partition=True, enable_sharing=True, oversubscription=0.95),
        sandbox=SandboxConfig(
            enable_memory_gate=True,
            enable_bandwidth_gate=False,
            enable_compute_gate=False,
            slo_guard=SLOGuardConfig(enabled=False),
        ),
    ),
    num_tasks=52,
    arrival_mode="wave",
)
