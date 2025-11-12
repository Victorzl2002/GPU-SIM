"""
离散时间仿真引擎，实现“到达→调度→沙盒→推进→采样”的循环。
"""

from __future__ import annotations

from typing import Dict, List

from core.cluster.node import ClusterNode
from core.isolation import APISandbox
from core.workload.task import Task, TaskState
from core.scheduling.glb import GLBScheduler
from evaluation.metrics.collector import MetricCollector
from core.vgpu_model.resource_model.vgpu_resource import VGPUResource

from .config import SimulationConfig


class SimulationEngine:
    """封装仿真主循环，以及调度/沙盒/指标等子系统的编排。"""
    def __init__(self, nodes: List[ClusterNode], tasks: List[Task], config: SimulationConfig):
        self.nodes = nodes
        self.tasks = sorted(tasks, key=lambda t: t.arrival_time)
        self.config = config
        self.scheduler = GLBScheduler(nodes, config.scheduling)
        self.sandbox = APISandbox(config.sandbox)
        self.metrics = MetricCollector(nodes)
        self.node_lookup: Dict[str, ClusterNode] = {node.node_id: node for node in nodes}
        self.running_tasks: Dict[str, Task] = {}
        self.task_index = {task.task_id: task for task in tasks}

    def _zero_usage(self, node_id: str) -> VGPUResource:
        """构造零向量，用于在未取样节点上补数据。"""
        node = self.node_lookup[node_id]
        model = node.gpus[0].model if node.gpus else ""
        return VGPUResource(
            compute=0.0,
            memory=0.0,
            bandwidth=0.0,
            resource_id=f"{node_id}-usage",
            vendor=node.vendor,
            model=model,
        )

    def _slo_pressure(self, task: Task, current_time: float) -> bool:
        """判断任务是否逼近 deadline，驱动 SLO 守护。"""
        if task.start_time is None or task.deadline <= 0:
            return False
        elapsed = current_time - task.start_time
        remaining_budget = task.deadline - elapsed
        return remaining_budget <= max(task.deadline * 0.1, 0.05)

    def _record_usage(self, node_usage: Dict[str, VGPUResource]) -> None:
        """将每个 tick 的节点使用量写入指标收集器。"""
        for node in self.nodes:
            usage = node_usage.get(node.node_id, self._zero_usage(node.node_id))
            self.metrics.record_node_usage(node.node_id, usage)

    def run(self) -> Dict[str, Dict[str, float]]:
        """执行完整仿真并返回指标摘要。"""
        total_ticks = int(self.config.duration / self.config.delta_t)
        for tick in range(total_ticks):
            current_time = tick * self.config.delta_t
            ready = [
                task
                for task in self.tasks
                if task.state == TaskState.WAITING and task.is_arrived(current_time)
            ]

            if tick % self.config.scheduling_interval == 0 and ready:
                for allocation in self.scheduler.schedule(ready):
                    task = self.task_index[allocation.task_id]
                    task.assign(allocation.node_id, allocation.quota, current_time)
                    self.running_tasks[task.task_id] = task

            node_usage: Dict[str, VGPUResource] = {}
            for task in list(self.running_tasks.values()):
                if task.state not in {TaskState.RUNNING, TaskState.SCHEDULED} or not task.quota:
                    continue
                if task.node_id is None:
                    continue
                slo_pressure = self._slo_pressure(task, current_time)
                dynamic_demand = task.current_demand(current_time)
                decision = self.sandbox.apply(task, dynamic_demand, task.quota, self.config.delta_t, slo_pressure)
                for limiter_name, triggered in decision.limited.items():
                    if triggered:
                        task.record_limiter_event(limiter_name)
                        self.metrics.record_limiter(limiter_name)
                task.update_progress(decision.usage.compute, self.config.delta_t)
                node_usage.setdefault(task.node_id, self._zero_usage(task.node_id))
                node_usage[task.node_id] = node_usage[task.node_id] + decision.usage

                if task.state == TaskState.COMPLETED:
                    task.finalize(current_time)
                    self.metrics.record_task_completion(task)
                    self.scheduler.release(task)
                    self.sandbox.release(task)
                    self.running_tasks.pop(task.task_id, None)
                elif task.start_time and current_time - task.start_time > task.deadline * 1.5:
                    task.mark_dropped(current_time)
                    self.metrics.record_task_completion(task)
                    self.scheduler.release(task)
                    self.sandbox.release(task)
                    self.running_tasks.pop(task.task_id, None)

            self._record_usage(node_usage)

        # Finalize remaining tasks
        for task in self.running_tasks.values():
            task.mark_dropped(self.config.duration)
            self.metrics.record_task_completion(task)
            self.scheduler.release(task)
            self.sandbox.release(task)
        for task in self.tasks:
            if task.state == TaskState.WAITING:
                task.mark_dropped(self.config.duration)
                self.metrics.record_task_completion(task)

        return self.metrics.summarize()
