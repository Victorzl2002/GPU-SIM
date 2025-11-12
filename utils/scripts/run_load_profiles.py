#!/usr/bin/env python3
"""
Run all scenarios under both normal and stress loads to contrast baseline and sandbox behavior.
"""

from __future__ import annotations

import argparse
from typing import Dict, Any

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.runner import SCENARIOS, run

LOAD_PROFILES = {
    "normal": {
        "duration": 320.0,
        "num_tasks": 120,
        "arrival_mode": "poisson",
    },
    "stress": {
        "duration": 200.0,
        "num_tasks": 220,
        "arrival_mode": "burst",
    },
}


def format_summary(name: str, load: str, summary: Dict[str, Any]) -> str:
    duration_p95 = summary.get("duration_p95")
    ir_p95 = summary.get("ir_p95")
    limiter = summary.get("limiter_events", {})
    duration_text = f"{duration_p95:.1f}" if duration_p95 is not None else "N/A"
    ir_text = f"{ir_p95:.3f}" if ir_p95 is not None else "N/A"
    return (
        f"{name} [{load}]  "
        f"SLO={summary.get('slo_rate', 0):.3f}  "
        f"IR(avg)={summary.get('avg_interference', 0):.3f}  "
        f"IR(p95)={ir_text}  "
        f"Dur(p95)={duration_text}  "
        f"Limiter={limiter}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scenarios under normal and stress loads.")
    parser.add_argument("--seed", type=int, default=21, help="Random seed for workload generation.")
    args = parser.parse_args()

    for name in SCENARIOS:
        for load_name, overrides in LOAD_PROFILES.items():
            summary = run(
                name,
                seed=args.seed,
                duration=overrides["duration"],
                num_tasks=overrides["num_tasks"],
                arrival_mode=overrides["arrival_mode"],
            )
            print(format_summary(name, load_name, summary))


if __name__ == "__main__":
    main()
