"""
一级 GLB：按跨厂商得分挑选最优平台/资源池。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from core.vgpu_model.normalization.cross_vendor_scorer import CrossVendorScorer
from core.vgpu_model.resource_model.vgpu_resource import VGPUResource

from core.cluster.node import ClusterNode


@dataclass
class VendorDecision:
    """记录首阶段选择结果：目标厂商得分及候选节点列表。"""

    vendor: str
    score: float
    candidate_nodes: List[ClusterNode]


class VendorSelector:
    """结合折算系数对不同厂商资源做 cost/perf 评估。"""

    def __init__(self, scorer: Optional[CrossVendorScorer] = None):
        self._scorer = scorer or CrossVendorScorer()

    def _aggregate_by_vendor(self, nodes: Iterable[ClusterNode]) -> Dict[str, VGPUResource]:
        """把同厂商节点的可用容量聚合成池，用于 cost 计算。"""
        aggregate: Dict[str, VGPUResource] = {}
        for node in nodes:
            total = aggregate.get(node.vendor)
            if total is None:
                total = node.available_capacity()
            else:
                total += node.available_capacity()
            aggregate[node.vendor] = total
        return aggregate

    def select(self, task_demand: VGPUResource, nodes: Iterable[ClusterNode], compatibility: Iterable[str]) -> Optional[VendorDecision]:
        """根据任务兼容性与得分挑选最优厂商，并返回对应节点列表。"""
        compatible_nodes = [
            node for node in nodes if node.vendor in compatibility
        ]
        if not compatible_nodes:
            return None

        vendor_totals = self._aggregate_by_vendor(compatible_nodes)
        best_vendor: Optional[str] = None
        best_score = float("-inf")
        for vendor, aggregate in vendor_totals.items():
            if (
                aggregate.compute <= 0
                or aggregate.memory <= 0
                or aggregate.bandwidth <= 0
            ):
                continue
            try:
                score = self._scorer.calculate_cross_vendor_score(task_demand, aggregate)
            except ValueError:
                continue
            if score > best_score:
                best_score = score
                best_vendor = vendor

        if best_vendor is None:
            return None

        candidate_nodes = [node for node in compatible_nodes if node.vendor == best_vendor]
        return VendorDecision(vendor=best_vendor, score=best_score, candidate_nodes=candidate_nodes)
