#!/usr/bin/env python3
"""
Plot per-task IR distribution for baseline vs sandbox under stress load.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.runner import run  # noqa: E402


def collect_ir(tasks) -> List[float]:
    return sorted(
        ir for task in tasks if (ir := task.interference_ratio) is not None
    )


def limited_ratio(tasks) -> Tuple[int, int]:
    limited = sum(1 for task in tasks if sum(task.limiter_events.values()) > 0)
    return limited, len(tasks)


def build_cdf(values: List[float]):
    if not values:
        return [0], [0]
    x, y = [], []
    n = len(values)
    for idx, val in enumerate(values):
        x.append(val)
        y.append((idx + 1) / n)
    return x, y


def summarize(name: str, params: dict, seed: int):
    summary, tasks = run(name, seed=seed, return_tasks=True, **params)
    ir_values = collect_ir(tasks)
    limited, total = limited_ratio(tasks)
    return summary, ir_values, limited, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot IR CDF for baseline vs sandbox.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--duration", type=float, default=200.0)
    parser.add_argument("--num_tasks", type=int, default=220)
    parser.add_argument("--arrival_mode", default="burst")
    parser.add_argument("--output", default="ir_distribution.png")
    args = parser.parse_args()

    stress_params = {
        "duration": args.duration,
        "num_tasks": args.num_tasks,
        "arrival_mode": args.arrival_mode,
    }

    scenarios = {
        "A1-baseline": "#4F81BD",
        "A3-sandbox": "#C0504D",
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    text_lines = []

    for name, color in scenarios.items():
        summary, ir_values, limited, total = summarize(name, stress_params, args.seed)
        x, y = build_cdf(ir_values)
        ax.step(x, y, where="post", label=f"{name}", color=color)
        text_lines.append(
            f"{name}: SLO={summary['slo_rate']:.3f}, "
            f"IR(avg)={summary['avg_interference']:.3f}, "
            f"IR(p95)={summary.get('ir_p95','N/A')}, "
            f"Limiter tasks={limited}/{total}"
        )

    ax.set_xlabel("Interference Ratio")
    ax.set_ylabel("CDF")
    ax.set_title("IR CDF under stress load")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.text(
        0.02,
        0.02,
        "\n".join(text_lines),
        fontsize=9,
        va="bottom",
    )

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
