"""
二级 GLB：基于 LRP + BRA 的节点评分。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math

from core.vgpu_model.resource_model.vgpu_resource import VGPUResource
from core.cluster.node import ClusterNode


@dataclass
class NodeScore:
    """封装节点得分及组成项，便于调试。"""

    node: ClusterNode
    score: float
    lrp: float
    bra: float


class NodeSelector:
    """执行 LRP (剩余资源) 与 BRA (均衡性) 计算的选择器。"""

    def __init__(self, lambda_weight: float = 0.6):
        self.lambda_weight = lambda_weight

    def _lrp_score(self, node: ClusterNode, demand: VGPUResource) -> float:
        """LRP：放置后剩余越多得分越高，无法放置则返回 -inf。"""
        if not node.can_allocate(demand):
            return float("-inf")
        residual = node.available_capacity() - demand
        total = node.total_capacity()
        return (
            residual.compute / total.compute
            + residual.memory / total.memory
            + residual.bandwidth / total.bandwidth
        ) / 3

    def _bra_score(self, node: ClusterNode, demand: VGPUResource) -> float:
        """BRA：三维利用率越均衡波动越小，得分越高。"""
        total = node.total_capacity()
        usage = node.current_usage() + demand
        ratios = [
            usage.compute / total.compute,
            usage.memory / total.memory,
            usage.bandwidth / total.bandwidth,
        ]
        mean_ratio = sum(ratios) / 3
        variance = sum((r - mean_ratio) ** 2 for r in ratios) / 3
        return 1 - math.sqrt(max(variance, 0.0))

    def select(self, nodes: List[ClusterNode], demand: VGPUResource) -> Optional[NodeScore]:
        """遍历候选节点，返回综合得分最高的节点。"""
        best: Optional[NodeScore] = None
        for node in nodes:
            lrp = self._lrp_score(node, demand)
            if lrp == float("-inf"):
                continue
            bra = self._bra_score(node, demand)
            combined = self.lambda_weight * lrp + (1 - self.lambda_weight) * bra
            if best is None or combined > best.score:
                best = NodeScore(node=node, score=combined, lrp=lrp, bra=bra)
        return best
