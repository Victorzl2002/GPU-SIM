"""
运行已定义场景的辅助入口。
"""

from __future__ import annotations

from typing import Dict

from experiments.base import ExperimentProfile
from experiments.abl_no_link_gate.scenario import SCENARIO as A4
from experiments.abl_no_slo_guard.scenario import SCENARIO as A5
from experiments.baseline.scenario import SCENARIO as A1
from experiments.hard_split.scenario import SCENARIO as A2
from experiments.parvagpu.scenario import SCENARIO as A2P
from experiments.sandbox.scenario import SCENARIO as A3


SCENARIOS: Dict[str, ExperimentProfile] = {
    A1.name: A1,
    A2.name: A2,
    A2P.name: A2P,
    A3.name: A3,
    A4.name: A4,
    A5.name: A5,
}


def run(name: str, seed: int = 2025, return_tasks: bool = False, **overrides):
    """按名称查找场景并运行仿真，方便脚本化调用。"""
    scenario = SCENARIOS.get(name)
    if scenario is None:
        raise KeyError(f"未知场景: {name}")
    return scenario.run(seed=seed, return_tasks=return_tasks, **overrides)
