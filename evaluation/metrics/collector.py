"""
仿真实验的指标采集与汇总。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Dict, List

from core.cluster.node import ClusterNode
from core.vgpu_model.resource_model.vgpu_resource import VGPUResource
from core.workload.task import Task


def _zero() -> Dict[str, float]:
    return {"compute": 0.0, "memory": 0.0, "bandwidth": 0.0}


@dataclass
class MetricCollector:
    """聚合节点使用率、SLO 完成度、干扰率与限流事件。"""
    nodes: List[ClusterNode]
    node_capacity: Dict[str, VGPUResource] = field(init=False)
    node_usage_samples: Dict[str, List[VGPUResource]] = field(init=False)
    limiter_events: Dict[str, int] = field(default_factory=_zero)
    completed_tasks: List[Task] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.node_capacity = {node.node_id: node.total_capacity() for node in self.nodes}
        self.node_usage_samples = {node.node_id: [] for node in self.nodes}

    def record_node_usage(self, node_id: str, usage: VGPUResource) -> None:
        self.node_usage_samples.setdefault(node_id, []).append(usage)

    def record_task_completion(self, task: Task) -> None:
        self.completed_tasks.append(task)

    def record_limiter(self, limiter_name: str) -> None:
        self.limiter_events[limiter_name] = self.limiter_events.get(limiter_name, 0) + 1

    def _utilization(self, samples: List[VGPUResource], capacity: VGPUResource) -> Dict[str, float]:
        if not samples:
            return {"compute": 0.0, "memory": 0.0, "bandwidth": 0.0}
        compute_avg = mean(s.compute for s in samples)
        memory_avg = mean(s.memory for s in samples)
        bandwidth_avg = mean(s.bandwidth for s in samples)
        return {
            "compute": compute_avg / capacity.compute if capacity.compute else 0.0,
            "memory": memory_avg / capacity.memory if capacity.memory else 0.0,
            "bandwidth": bandwidth_avg / capacity.bandwidth if capacity.bandwidth else 0.0,
        }

    def _stability(self, samples: List[VGPUResource], capacity: VGPUResource) -> Dict[str, float]:
        if len(samples) < 2:
            return {"compute": 0.0, "memory": 0.0, "bandwidth": 0.0}
        compute_ratio = [s.compute / capacity.compute for s in samples if capacity.compute]
        memory_ratio = [s.memory / capacity.memory for s in samples if capacity.memory]
        bandwidth_ratio = [s.bandwidth / capacity.bandwidth for s in samples if capacity.bandwidth]
        return {
            "compute": pstdev(compute_ratio) if len(compute_ratio) > 1 else 0.0,
            "memory": pstdev(memory_ratio) if len(memory_ratio) > 1 else 0.0,
            "bandwidth": pstdev(bandwidth_ratio) if len(bandwidth_ratio) > 1 else 0.0,
        }

    def summarize(self) -> Dict[str, Dict[str, float]]:
        """输出节点指标 + 全局 SLO/IR + 三门触发次数。"""
        node_metrics = {}
        for node_id, samples in self.node_usage_samples.items():
            capacity = self.node_capacity[node_id]
            node_metrics[node_id] = {
                "utilization": self._utilization(samples, capacity),
                "stability": self._stability(samples, capacity),
            }

        slo_met = [task.slo_met for task in self.completed_tasks if task.slo_met is not None]
        interference = [task.interference_ratio for task in self.completed_tasks if task.interference_ratio]

        summary = {
            "nodes": node_metrics,
            "slo_rate": sum(1 for met in slo_met if met) / len(slo_met) if slo_met else 0.0,
            "avg_interference": mean(interference) if interference else 0.0,
            "limiter_events": self.limiter_events,
        }
        return summary
