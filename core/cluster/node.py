"""
集群节点模型。

一个节点聚合多张同厂商 GPU，二级调度 (LRP + BRA) 以节点为基本粒度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.vgpu_model.resource_model.vgpu_resource import VGPUResource

from .gpu import GPUDevice


def _zero(vendor: str, model: str) -> VGPUResource:
    return VGPUResource(
        compute=0.0,
        memory=0.0,
        bandwidth=0.0,
        resource_id="zero",
        vendor=vendor,
        model=model,
    )


@dataclass
class ClusterNode:
    """表示同厂商 GPU 聚合而成的节点，负责追踪配额、使用率与链路信息。"""
    node_id: str
    vendor: str
    gpus: List[GPUDevice]
    link_bandwidth: float
    labels: Dict[str, str] = field(default_factory=dict)
    task_quotas: Dict[str, VGPUResource] = field(default_factory=dict)

    def total_capacity(self) -> VGPUResource:
        capacity = _zero(self.vendor, self.gpus[0].model if self.gpus else "")
        for gpu in self.gpus:
            capacity += gpu.effective_capacity()
        return capacity

    def current_usage(self) -> VGPUResource:
        usage = _zero(self.vendor, self.gpus[0].model if self.gpus else "")
        for quota in self.task_quotas.values():
            usage += quota
        return usage

    def available_capacity(self) -> VGPUResource:
        total = self.total_capacity()
        usage = self.current_usage()
        return total - usage

    def can_allocate(self, demand: VGPUResource) -> bool:
        available = self.available_capacity()
        return (
            available.compute >= demand.compute
            and available.memory >= demand.memory
            and available.bandwidth >= demand.bandwidth
        )

    def allocate(self, task_id: str, quota: VGPUResource) -> bool:
        if not self.can_allocate(quota):
            return False
        self.task_quotas[task_id] = quota
        return True

    def release(self, task_id: str) -> Optional[VGPUResource]:
        return self.task_quotas.pop(task_id, None)

    def utilization_ratio(self) -> Dict[str, float]:
        total = self.total_capacity()
        usage = self.current_usage()
        if total.compute == 0 or total.memory == 0 or total.bandwidth == 0:
            return {"compute": 0.0, "memory": 0.0, "bandwidth": 0.0}
        return {
            "compute": usage.compute / total.compute,
            "memory": usage.memory / total.memory,
            "bandwidth": usage.bandwidth / total.bandwidth,
        }

    def describe(self) -> Dict[str, float]:
        total = self.total_capacity()
        return {
            "node_id": self.node_id,
            "vendor": self.vendor,
            "link_bandwidth": self.link_bandwidth,
            "total_compute": total.compute,
            "total_memory": total.memory,
            "total_bandwidth": total.bandwidth,
            "active_tasks": len(self.task_quotas),
        }
