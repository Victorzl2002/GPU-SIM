#!/usr/bin/env python3
"""
Compare per-task IR distribution between baseline and sandbox under stress load.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.runner import run  # noqa: E402


def bucket_ir(tasks, bucket_size=0.1):
    counter = Counter()
    for t in tasks:
        ir = t.interference_ratio
        if ir is None:
            continue
        bucket = round(ir / bucket_size) * bucket_size
        counter[bucket] += 1
    return counter


def ratio_above(tasks, threshold):
    total = 0
    hit = 0
    for t in tasks:
        ir = t.interference_ratio
        if ir is None:
            continue
        total += 1
        if ir > threshold:
            hit += 1
    return (hit / total) if total else 0.0


def limited_ratio(tasks):
    total = len(tasks) or 1
    limited = sum(1 for t in tasks if sum(t.limiter_events.values()) > 0)
    return limited / total


def summarize(name, stress_params, seed):
    summary, tasks = run(name, seed=seed, return_tasks=True, **stress_params)
    bucket = bucket_ir(tasks)
    print(f"=== {name} ===")
    print(
        f"SLO={summary['slo_rate']:.3f}  IR(avg)={summary['avg_interference']:.3f}  "
        f"IR(p95)={summary.get('ir_p95','N/A')}  SLO_violation={summary.get('slo_violations')}"
    )
    print(
        f"IR>1.5={ratio_above(tasks,1.5):.2%}  IR>2.0={ratio_above(tasks,2.0):.2%}  "
        f"Limited tasks={limited_ratio(tasks):.2%}"
    )
    print("Limiter events:", summary.get("limiter_events"))
    print("IR buckets:")
    for key in sorted(bucket):
        print(f"  {key:.1f}: {bucket[key]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare per-task IR distribution between baseline and sandbox.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--duration", type=float, default=200.0)
    parser.add_argument("--num_tasks", type=int, default=220)
    parser.add_argument("--arrival_mode", default="burst")
    args = parser.parse_args()

    stress_params = {
        "duration": args.duration,
        "num_tasks": args.num_tasks,
        "arrival_mode": args.arrival_mode,
    }

    for scenario in ["A1-baseline", "A3-sandbox"]:
        summarize(scenario, stress_params, args.seed)


if __name__ == "__main__":
    main()
