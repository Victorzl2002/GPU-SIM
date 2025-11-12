"""
离散仿真用到的任务/工作负载原语。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Set
import math
import random

from core.vgpu_model.resource_model.vgpu_resource import VGPUResource


class TaskState(Enum):
    WAITING = auto()
    SCHEDULED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    DROPPED = auto()


@dataclass
class ResourceFluctuation:
    """描述任务在运行期的资源波动特性。"""

    compute_amp: float
    memory_amp: float
    bandwidth_amp: float
    period: float
    spike_probability: float
    spike_amp: float
    phase: float = field(default_factory=lambda: random.random())


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
    fluctuation: Optional[ResourceFluctuation] = None
    state: TaskState = TaskState.WAITING
    progress: float = 0.0
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    node_id: Optional[str] = None
    quota: Optional[VGPUResource] = None
    slo_met: Optional[bool] = None
    limiter_events: Dict[str, int] = field(default_factory=lambda: {"memory": 0, "bandwidth": 0, "compute": 0})

    def __post_init__(self) -> None:
        self._rng = random.Random(hash(self.task_id) & 0xFFFFFFFF)

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

    def current_demand(self, current_time: float) -> VGPUResource:
        """返回当前时间的动态需求向量。"""
        if not self.fluctuation or self.fluctuation.period <= 0:
            return self.demand

        def scale(base: float, amp: float) -> float:
            if amp <= 0:
                return base
            angle = 2 * math.pi * ((current_time / self.fluctuation.period) + self.fluctuation.phase)
            multiplier = 1 + amp * math.sin(angle)
            if self.fluctuation.spike_probability > 0 and self._rng.random() < self.fluctuation.spike_probability:
                multiplier += self.fluctuation.spike_amp
            return max(0.0, base * multiplier)

        return VGPUResource(
            compute=scale(self.demand.compute, self.fluctuation.compute_amp),
            memory=scale(self.demand.memory, self.fluctuation.memory_amp),
            bandwidth=scale(self.demand.bandwidth, self.fluctuation.bandwidth_amp),
            resource_id=self.demand.resource_id,
            vendor=self.demand.vendor,
            model=self.demand.model,
        )
