

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class LimiterOutput:
    """描述 limiter 输出的值以及是否触发限流。"""

    value: float
    limited: bool


class MemoryQuotaGate:
    """显存配额门，直接裁剪需求。"""

    def apply(self, demand: float, quota: float) -> LimiterOutput:
        allowed = min(demand, quota)
        return LimiterOutput(value=allowed, limited=allowed + 1e-9 < demand)


class BandwidthTokenBucket:
    """链路令牌桶，按照时间步充值/扣减，模拟带宽门。"""

    def __init__(self, refill_factor: float = 1.0):
        self.refill_factor = refill_factor
        self._tokens: Dict[str, float] = {}

    def apply(self, task_id: str, demand: float, quota: float, delta_t: float) -> LimiterOutput:
        bucket_size = max(quota * self.refill_factor, 1e-9)
        tokens = self._tokens.get(task_id, bucket_size)
        tokens = min(bucket_size, tokens + quota * delta_t)
        needed = demand * delta_t
        granted = min(needed, tokens)
        tokens -= granted
        self._tokens[task_id] = tokens
        allowed_bandwidth = granted / max(delta_t, 1e-9)
        return LimiterOutput(value=allowed_bandwidth, limited=allowed_bandwidth + 1e-9 < demand)

    def release(self, task_id: str) -> None:
        self._tokens.pop(task_id, None)


class ComputeThrottle:
    """算力节流门，可被 SLO 守护动态提升或衰减。"""

    def __init__(self, ceiling: float = 1.0):
        self.ceiling = ceiling
        self._boost: Dict[str, float] = {}

    def _scale(self, task_id: str) -> float:
        return self._boost.setdefault(task_id, 1.0)

    def apply(self, task_id: str, demand: float, quota: float) -> LimiterOutput:
        scale = min(self._scale(task_id), self.ceiling)
        allowed = min(demand, quota)
        return LimiterOutput(value=allowed, limited=allowed + 1e-9 < demand)

    def boost(self, task_id: str, amount: float, max_boost: float) -> None:
        current = self._scale(task_id)
        self._boost[task_id] = min(max_boost, current + amount)

    def decay(self, task_id: str, decay: float) -> None:
        current = self._scale(task_id)
        self._boost[task_id] = max(1.0, current - decay)

    def release(self, task_id: str) -> None:
        self._boost.pop(task_id, None)
