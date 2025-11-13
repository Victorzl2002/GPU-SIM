"""
实验场景共用的构建/运行辅助模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional

from core.cluster.gpu import GPUDevice
from core.cluster.node import ClusterNode
from core.simulation.config import SimulationConfig
from core.simulation.engine import SimulationEngine
from core.vgpu_model.resource_model.vgpu_resource import VGPUResource
from core.workload.generator import TaskProfile, WorkloadGenerator


def build_reference_cluster() -> List[ClusterNode]:
    """构造包含 A100 + 910B 的异构基准集群。"""
    #创建单卡
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


# 预定义任务谱
DEFAULT_PROFILES = [
    TaskProfile(
        name="llm-batch",
        demand=VGPUResource(compute=160, memory=56, bandwidth=950, resource_id="llm-batch", vendor="", model=""),
        workload=4000,   # 理想完成时间 25s
        deadline=40.0,
        compatibility={"nvidia", "huawei"},
        k_min=2,
        k_max=4,
    ),
    TaskProfile(
        name="multimodal-online",
        demand=VGPUResource(compute=120, memory=42, bandwidth=720, resource_id="multimodal-online", vendor="", model=""),
        workload=3000,   # 理想完成时间 25s
        deadline=35.0,
        compatibility={"nvidia"},
        k_min=1,
        k_max=2,
    ),
    TaskProfile(
        name="preprocess-pipeline",
        demand=VGPUResource(compute=90, memory=30, bandwidth=480, resource_id="preprocess-pipeline", vendor="", model=""),
        workload=2700,   # 理想完成时间 30s
        deadline=45.0,
        compatibility={"nvidia", "huawei"},
    ),
    TaskProfile(
        name="feature-etl",
        demand=VGPUResource(compute=60, memory=24, bandwidth=360, resource_id="feature-etl", vendor="", model=""),
        workload=1800,    # 理想完成时间 30s
        deadline=42.0,
        compatibility={"huawei"},
    ),
    TaskProfile(
        name="llm-heavy",
        demand=VGPUResource(
            compute=210,
            memory=48,
            bandwidth=1100,
            resource_id="llm-heavy",
            vendor="",
            model="",
        ),
        workload=7200,    # 理想完成时间 30s
        deadline=60.0,
        compatibility={"nvidia"},
        k_min=3,
        k_max=4,
    ),
]


@dataclass
class ExperimentProfile:
    """封装单个场景的集群、任务谱与仿真配置。"""
    name: str
    description: str
    #仿真配置（基线对比配置修改）
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    num_tasks: int = 48
    arrival_mode: str = "poisson"
    #任务谱
    workload_profiles: List[TaskProfile] = field(default_factory=lambda: DEFAULT_PROFILES)
    #集群构造
    cluster_factory: Callable[[], List[ClusterNode]] = build_reference_cluster

    def run(
        self,
        seed: int = 2025,
        duration: Optional[float] = None,
        num_tasks: Optional[int] = None,
        arrival_mode: Optional[str] = None,
        return_tasks: bool = False,
    ):
        """构造任务 → 运行仿真 → 返回指标摘要。可覆盖负载参数以形成不同压力场景。"""
        generator = WorkloadGenerator(seed=seed)
        duration_value = duration if duration is not None else self.simulation.duration
        num_tasks_value = num_tasks if num_tasks is not None else self.num_tasks
        arrival_mode_value = arrival_mode if arrival_mode is not None else self.arrival_mode
        #任务生成
        tasks = generator.generate(
            profiles=self.workload_profiles,
            num_tasks=num_tasks_value,
            duration=duration_value,
            arrival_mode=arrival_mode_value,
        )
        # for task in tasks:
        #     print(task)
        sim_config = replace(self.simulation, duration=duration_value)
        engine = SimulationEngine(self.cluster_factory(), tasks, sim_config)
        summary = engine.run()
        if return_tasks:
            return summary, list(engine.metrics.completed_tasks)
        return summary
