"""
实验场景共用的构建/运行辅助模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

from core.cluster.gpu import GPUDevice
from core.cluster.node import ClusterNode
from core.simulation.config import SimulationConfig
from core.simulation.engine import SimulationEngine
from core.vgpu_model.resource_model.vgpu_resource import VGPUResource
from core.workload.generator import TaskProfile, WorkloadGenerator


def build_reference_cluster() -> List[ClusterNode]:
    """构造包含 A100 + 910B 的异构基准集群。"""
    def gpu(gpu_id: str, vendor: str, model: str, compute: float, memory: float, bandwidth: float) -> GPUDevice:
        return GPUDevice(
            gpu_id=gpu_id,
            vendor=vendor,
            model=model,
            capacity=VGPUResource(
                compute=compute,
                memory=memory,
                bandwidth=bandwidth,
                resource_id=gpu_id,
                vendor=vendor,
                model=model,
            ),
            link_bandwidth=900.0 if vendor == "nvidia" else 700.0,
        )

    nodes = [
        ClusterNode(
            node_id="nv-node-1",
            vendor="nvidia",
            gpus=[gpu("a100-01", "nvidia", "A100", 312, 80, 2039), gpu("a100-02", "nvidia", "A100", 312, 80, 2039)],
            link_bandwidth=900.0,
        ),
        ClusterNode(
            node_id="nv-node-2",
            vendor="nvidia",
            gpus=[gpu("a100-03", "nvidia", "A100", 312, 80, 2039)],
            link_bandwidth=900.0,
        ),
        ClusterNode(
            node_id="asc-node-1",
            vendor="huawei",
            gpus=[gpu("910b-01", "huawei", "Ascend910B", 280, 64, 1600), gpu("910b-02", "huawei", "Ascend910B", 280, 64, 1600)],
            link_bandwidth=720.0,
        ),
        ClusterNode(
            node_id="asc-node-2",
            vendor="huawei",
            gpus=[gpu("910b-03", "huawei", "Ascend910B", 280, 64, 1600)],
            link_bandwidth=720.0,
        ),
    ]
    return nodes


DEFAULT_PROFILES = [
    TaskProfile(
        name="llm-batch",
        demand=VGPUResource(compute=140, memory=48, bandwidth=800, resource_id="llm-batch", vendor="", model=""),
        workload=3200,   # 理想完成时间 ~23s
        deadline=35.0,
        compatibility={"nvidia", "huawei"},
        k_min=2,
        k_max=4,
    ),
    TaskProfile(
        name="cv-online",
        demand=VGPUResource(compute=80, memory=24, bandwidth=550, resource_id="cv-online", vendor="", model=""),
        workload=1600,   # 理想完成时间 20s
        deadline=28.0,
        compatibility={"nvidia"},
        k_min=1,
        k_max=2,
    ),
    TaskProfile(
        name="preprocess",
        demand=VGPUResource(compute=60, memory=28, bandwidth=380, resource_id="preprocess", vendor="", model=""),
        workload=1800,   # 理想完成时间 30s
        deadline=42.0,
        compatibility={"nvidia", "huawei"},
    ),
    TaskProfile(
        name="feature-etl",
        demand=VGPUResource(compute=40, memory=18, bandwidth=260, resource_id="feature-etl", vendor="", model=""),
        workload=900,    # 理想完成时间 22.5s
        deadline=30.0,
        compatibility={"huawei"},
    ),
]


@dataclass
class ExperimentProfile:
    """封装单个场景的集群、任务谱与仿真配置。"""
    name: str
    description: str
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    num_tasks: int = 48
    arrival_mode: str = "poisson"
    workload_profiles: List[TaskProfile] = field(default_factory=lambda: DEFAULT_PROFILES)
    cluster_factory: Callable[[], List[ClusterNode]] = build_reference_cluster

    def run(self, seed: int = 2025) -> Dict[str, Dict[str, float]]:
        """构造任务 → 运行仿真 → 返回指标摘要。"""
        generator = WorkloadGenerator(seed=seed)
        tasks = generator.generate(
            profiles=self.workload_profiles,
            num_tasks=self.num_tasks,
            duration=self.simulation.duration,
            arrival_mode=self.arrival_mode,
        )
        engine = SimulationEngine(self.cluster_factory(), tasks, self.simulation)
        return engine.run()
