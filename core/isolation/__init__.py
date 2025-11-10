"""
API sandbox implementation (memory quota gate, bandwidth token bucket, compute throttle).
"""

from __future__ import annotations

from typing import Dict

from core.vgpu_model.resource_model.vgpu_resource import VGPUResource
from core.workload.task import Task

from .api_model import SandboxDecision
from .hooks import registry
from .limiters import BandwidthTokenBucket, ComputeThrottle, MemoryQuotaGate
from .policies import SandboxConfig


class APISandbox:
    """实现显存门、带宽令牌桶与算力节流协同的运行时沙盒。"""
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.memory_gate = MemoryQuotaGate()
        self.bandwidth_gate = BandwidthTokenBucket(config.bandwidth_refill_rate)
        self.compute_gate = ComputeThrottle(config.compute_ceiling)

    def apply(self, task: Task, quota: VGPUResource, delta_t: float, slo_pressure: bool) -> SandboxDecision:
        """执行一次限流决策，返回实际可用资源与限流标记。"""
        desired = task.demand
        limited = {"memory": False, "bandwidth": False, "compute": False}

        if self.config.enable_memory_gate:
            mem = self.memory_gate.apply(desired.memory, quota.memory)
            limited["memory"] = mem.limited
        else:
            mem = self.memory_gate.apply(desired.memory, desired.memory)
        if self.config.enable_bandwidth_gate:
            bw = self.bandwidth_gate.apply(task.task_id, desired.bandwidth, quota.bandwidth, delta_t)
            limited["bandwidth"] = bw.limited
        else:
            bw = self.bandwidth_gate.apply(task.task_id, desired.bandwidth, desired.bandwidth, delta_t)
        if self.config.enable_compute_gate:
            if self.config.slo_guard.enabled:
                if slo_pressure:
                    self.compute_gate.boost(task.task_id, amount=0.05, max_boost=self.config.slo_guard.max_boost)
                else:
                    self.compute_gate.decay(task.task_id, decay=self.config.slo_guard.decay)
            compute = self.compute_gate.apply(task.task_id, desired.compute, quota.compute)
            limited["compute"] = compute.limited
        else:
            compute = self.compute_gate.apply(task.task_id, desired.compute, desired.compute)

        usage = VGPUResource(
            compute=compute.value,
            memory=mem.value,
            bandwidth=bw.value,
            resource_id=quota.resource_id,
            vendor=quota.vendor,
            model=quota.model,
        )
        decision = SandboxDecision(task_id=task.task_id, usage=usage, limited=limited)
        registry.emit("sandbox_apply", decision=decision)
        return decision

    def release(self, task: Task) -> None:
        """任务结束后释放内部状态（令牌桶/算力提升）。"""
        self.bandwidth_gate.release(task.task_id)
        self.compute_gate.release(task.task_id)
