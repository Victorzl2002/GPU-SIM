#!/usr/bin/env python3
"""
Utilities shared by parameter experiment scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from core.workload.task import Task, TaskState
from run_experiment import limited_ratio_float, ratio_above


@dataclass
class RunMetrics:
    """Normalized metrics extracted from a single simulation run."""

    scenario: str
    load: int
    seed: int
    slo_rate: float
    avg_interference: float
    drop_rate: float
    drop_count: int
    total_tasks: int
    limiter_events: float
    limited_ratio: float
    ir_gt_1: float
    ir_gt_1_25: float
    ir_gt_1_5: float


def summarize_run(
    scenario: str,
    load: int,
    seed: int,
    tasks: Iterable[Task],
    summary: Dict[str, float],
) -> RunMetrics:
    """Collect SLO/drop/IR stats from a completed run."""

    task_list = list(tasks)
    total_tasks = len(task_list) or 1
    drop_count = sum(1 for task in task_list if getattr(task, "state", None) == TaskState.DROPPED)
    drop_rate = drop_count / total_tasks

    limiter_events = sum(summary.get("limiter_events", {}).values())
    limited_ratio = limited_ratio_float(task_list)

    ir_gt_1 = ratio_above(task_list, 1.0)
    ir_gt_1_25 = ratio_above(task_list, 1.25)
    ir_gt_1_5 = ratio_above(task_list, 1.5)

    return RunMetrics(
        scenario=scenario,
        load=load,
        seed=seed,
        slo_rate=summary.get("slo_rate", 0.0),
        avg_interference=summary.get("avg_interference", 0.0),
        drop_rate=drop_rate,
        drop_count=drop_count,
        total_tasks=total_tasks,
        limiter_events=limiter_events,
        limited_ratio=limited_ratio,
        ir_gt_1=ir_gt_1,
        ir_gt_1_25=ir_gt_1_25,
        ir_gt_1_5=ir_gt_1_5,
    )


def average_metric(rows: List[RunMetrics], field: str) -> float:
    if not rows:
        return 0.0
    return sum(getattr(row, field) for row in rows) / len(rows)
