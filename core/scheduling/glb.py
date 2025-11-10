"""
两级 GLB 调度器实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from core.cluster.node import ClusterNode
from core.vgpu_model.resource_model.vgpu_resource import VGPUResource
from core.workload.task import Task, TaskState

from .config import SchedulingConfig
from .node_selector import NodeSelector
from .vendor_selector import VendorSelector


@dataclass
class TaskAllocation:
    """记录一次分配的节点与静态配额。"""

    task_id: str
    node_id: str
    quota: VGPUResource


class GLBScheduler:
    """封装“选厂商 → 选节点 → 输出配额”的完整流程。"""

    def __init__(self, nodes: Iterable[ClusterNode], config: Optional[SchedulingConfig] = None):
        self.nodes: List[ClusterNode] = list(nodes)
        self.config = config or SchedulingConfig()
        self.vendor_selector = VendorSelector()
        self.node_selector = NodeSelector(lambda_weight=self.config.lambda_weight)
        self.node_index: Dict[str, ClusterNode] = {node.node_id: node for node in self.nodes}

    def _apply_oversubscription(self, demand: VGPUResource) -> VGPUResource:
        """根据配置放大需求，模拟静态配额上的超量共享。"""
        if self.config.oversubscription == 1.0:
            return demand
        return VGPUResource(
            compute=demand.compute * self.config.oversubscription,
            memory=demand.memory * self.config.oversubscription,
            bandwidth=demand.bandwidth * self.config.oversubscription,
            resource_id=demand.resource_id,
            vendor=demand.vendor,
            model=demand.model,
        )

    def allocate(self, task: Task) -> Optional[TaskAllocation]:
        """对单个任务执行两级选择，返回可用配额。"""
        vendor_decision = self.vendor_selector.select(task.demand, self.nodes, task.compatibility)
        if vendor_decision is None:
            return None

        demand = self._apply_oversubscription(task.demand)
        candidate = self.node_selector.select(vendor_decision.candidate_nodes, demand)
        if candidate is None:
            return None

        if self.config.static_partition and not self.config.enable_sharing and candidate.node.task_quotas:
            return None

        if not candidate.node.allocate(task.task_id, demand):
            return None

        return TaskAllocation(task_id=task.task_id, node_id= candidate.node.node_id, quota=demand)

    def schedule(self, tasks: List[Task]) -> List[TaskAllocation]:
        """批量尝试调度等待队列，返回成功分配的任务列表。"""
        allocations: List[TaskAllocation] = []
        for task in tasks:
            if task.state not in {TaskState.WAITING} or not task.compatibility:
                continue
            allocation = self.allocate(task)
            if allocation:
                allocations.append(allocation)
        return allocations

    def release(self, task: Task) -> None:
        """任务结束/掉线后释放节点配额。"""
        if task.node_id is None:
            return
        node = self.node_index.get(task.node_id)
        if node:
            node.release(task.task_id)
