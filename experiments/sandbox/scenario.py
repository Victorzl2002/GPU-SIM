"""
A3：GLB + API 沙盒三门 + SLO 守护的完整方案。
"""

from core.isolation.policies import SandboxConfig, SLOGuardConfig
from core.scheduling.config import SchedulingConfig
from core.simulation.config import SimulationConfig

from experiments.base import ExperimentProfile


SCENARIO = ExperimentProfile(
    name="A3-sandbox",
    description="GLB 两级调度 + API 沙盒三门（显存/带宽/算力）+ SLO 守护闭环",
    simulation=SimulationConfig(
        delta_t=0.01,
        duration=320.0,
        scheduling_interval=4,
        scheduling=SchedulingConfig(static_partition=False, enable_sharing=True, oversubscription=1.05),
        sandbox=SandboxConfig(
            enable_memory_gate=True,
            enable_bandwidth_gate=True,
            enable_compute_gate=True,
            slo_guard=SLOGuardConfig(enabled=True, adjust_interval=30, max_boost=1.25, decay=0.03),
            bandwidth_refill_rate=1.0,
            compute_ceiling=1.2,
        ),
    ),
    num_tasks=150,
    arrival_mode="burst",
)
