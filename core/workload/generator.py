"""
工作负载生成工具：根据实验定义批量生成带 SLO 的任务序列。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set
import math
import random

from core.vgpu_model.resource_model.vgpu_resource import VGPUResource

from .task import Task, ResourceFluctuation


@dataclass
class TaskProfile:
    """描述一类任务的三维需求、工作量与兼容性。"""

    name: str
    demand: VGPUResource
    workload: float
    deadline: float
    compatibility: Set[str]
    k_min: int = 1
    k_max: int = 1


class WorkloadGenerator:
    """用于生成 Poisson / Burst / Wave 等不同到达过程的任务列表。"""

    def __init__(self, seed: int = 2025):
        self._rng = random.Random(seed)

    def _sample_arrival(self, duration: float, mode: str) -> float:
        """按照模式采样到达时间，支持泊松/突发/波动等形态。"""
        if mode == "poisson":
            return self._rng.uniform(0, duration)
        if mode == "burst":
            if self._rng.random() < 0.6:
                return self._rng.uniform(0, duration * 0.1)
            return self._rng.uniform(duration * 0.1, duration)
        if mode == "poisson_burst":
            """大部分任务按照泊松分布随机到达，少部分集中在中段形成突发。"""
            if self._rng.random() < 0.7:
                return self._rng.uniform(0, duration)
            mid_start = duration * 0.45
            mid_end = duration * 0.55
            return self._rng.uniform(mid_start, mid_end)
        if mode == "wave":
            t = self._rng.uniform(0, 1)
            return duration * ((1 - math.cos(2 * math.pi * t)) / 2)
        return self._rng.uniform(0, duration)

    def generate(
        self,
        profiles: Sequence[TaskProfile],
        num_tasks: int,
        duration: float,
        arrival_mode: str = "poisson",
    ) -> List[Task]:
        """根据任务谱生成指定数量的 Task，默认按到达时间排序。"""
        tasks: List[Task] = []
        for idx in range(num_tasks):
            profile = profiles[idx % len(profiles)]
            arrival_time = self._sample_arrival(duration, arrival_mode)
            if "heavy" in profile.name:
                # Heavy 任务集中在仿真中段，制造短时突发。
                burst_span = max(duration * 0.05, 1e-3)
                mid_point = duration * 0.5
                start = max(0.0, mid_point - burst_span)
                end = min(duration, mid_point + burst_span)
                arrival_time = self._rng.uniform(start, end if end > start else start + 1e-6)
            ideal_duration = profile.workload / max(profile.demand.compute, 1e-6)*0.5
            fluctuation = ResourceFluctuation(
                compute_amp=self._rng.uniform(0.05, 0.2),
                memory_amp=self._rng.uniform(0.03, 0.15),
                bandwidth_amp=self._rng.uniform(0.03, 0.2),
                period=self._rng.uniform(10.0, 40.0),
                spike_probability=self._rng.uniform(0.005, 0.02),
                spike_amp=self._rng.uniform(0.1, 0.35),
            )
            task = Task(
                task_id=f"{profile.name}-{idx}",
                demand=VGPUResource(
                    compute=profile.demand.compute,
                    memory=profile.demand.memory,
                    bandwidth=profile.demand.bandwidth,
                    resource_id=f"task-{idx}",
                    vendor="",
                    model="",
                ),
                workload=profile.workload,
                arrival_time=arrival_time,
                deadline=profile.deadline,
                compatibility=set(profile.compatibility),
                k_min=profile.k_min,
                k_max=profile.k_max,
                ideal_duration=ideal_duration,
                fluctuation=fluctuation,
            )
            tasks.append(task)
        tasks.sort(key=lambda t: t.arrival_time)
        return tasks
