"""
离散仿真用到的任务/工作负载原语。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Set

from core.vgpu_model.resource_model.vgpu_resource import VGPUResource


class TaskState(Enum):
    WAITING = auto()
    SCHEDULED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    DROPPED = auto()


@dataclass
class Task:
    """模型化单个作业的资源需求、SLO、进度与限流统计。"""
    task_id: str
    demand: VGPUResource
    workload: float  # total work units to finish
    arrival_time: float
    deadline: float
    compatibility: Set[str]
    k_min: int
    k_max: int
    ideal_duration: float
    state: TaskState = TaskState.WAITING
    progress: float = 0.0
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    node_id: Optional[str] = None
    quota: Optional[VGPUResource] = None
    slo_met: Optional[bool] = None
    limiter_events: Dict[str, int] = field(default_factory=lambda: {"memory": 0, "bandwidth": 0, "compute": 0})

    def is_arrived(self, current_time: float) -> bool:
        """判断任务是否在当前时间步已经到达。"""
        return current_time >= self.arrival_time

    def assign(self, node_id: str, quota: VGPUResource, current_time: float) -> None:
        self.node_id = node_id
        self.quota = quota
        self.start_time = current_time
        self.state = TaskState.RUNNING

    def update_progress(self, compute_budget: float, delta_t: float) -> float:
        """按算力配额推进任务进度，返回剩余工作量。"""
        consumed_work = compute_budget * delta_t
        self.progress += consumed_work
        remaining = max(0.0, self.workload - self.progress)
        if remaining == 0:
            self.state = TaskState.COMPLETED
        return remaining

    def mark_dropped(self, current_time: float) -> None:
        self.state = TaskState.DROPPED
        self.completion_time = current_time
        self.slo_met = False

    def finalize(self, current_time: float) -> None:
        self.completion_time = current_time
        self.slo_met = (current_time - (self.start_time or current_time)) <= self.deadline

    def record_limiter_event(self, limiter_name: str) -> None:
        self.limiter_events[limiter_name] = self.limiter_events.get(limiter_name, 0) + 1

    @property
    def interference_ratio(self) -> Optional[float]:
        if not self.completion_time or not self.start_time or self.ideal_duration == 0:
            return None
        actual_duration = self.completion_time - self.start_time
        return actual_duration / self.ideal_duration
