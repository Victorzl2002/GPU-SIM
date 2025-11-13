#!/usr/bin/env python3
"""
扫描 reports/<dir>/*.json 提取 SLO & GPU 利用率并绘制折线图。
使用方式：python scripts/plot_gpu_slo.py --root reports --pattern 'a1-a3-task*' --scenarios A1-baseline A3-sandbox
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def collect_metrics(root: Path, pattern: str, scenarios: List[str]):
    entries = []
    for sub in sorted(root.glob(pattern)):
        if not sub.is_dir():
            continue
        task_count = sub.name.split("task")[-1]
        for name in scenarios:
            json_path = sub / f"{name}_report.json"
            if not json_path.exists():
                continue
            data = json.loads(json_path.read_text())
            summary = data.get("summary", {})
            nodes = summary.get("nodes", {})
            util_avg = sum(node.get("utilization", {}).get("compute", 0.0) for node in nodes.values()) / (len(nodes) or 1)
            entries.append({
                "dir": sub.name,
                "tasks": int(''.join(filter(str.isdigit, task_count)) or 0),
                "scenario": name,
                "slo_rate": summary.get("slo_rate", 0.0),
                "ir_p95": summary.get("ir_p95", 0.0),
                "ir_over_1_25": summary.get("ir_over_1_25_ratio", summary.get("ir_over_1_5_ratio", 0.0)),
                "gpu_util": util_avg,
            })
    entries.sort(key=lambda x: (x["tasks"], x["scenario"]))
    return entries


def plot_metric(entries: List[Dict], metric: str, ylabel: str, output: Path):
    plt.figure(figsize=(6, 4))
    scenario_names = sorted({e["scenario"] for e in entries})
    for scenario in scenario_names:
        xs = [e["tasks"] for e in entries if e["scenario"] == scenario]
        ys = [e[metric] for e in entries if e["scenario"] == scenario]
        plt.plot(xs, ys, marker="o", label=scenario)
    plt.xlabel("Number of tasks")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs load")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="reports")
    parser.add_argument("--pattern", default="a1-a3-task*")
    parser.add_argument("--scenarios", nargs="+", default=["A1-baseline", "A3-sandbox"])
    parser.add_argument("--out", default="reports/summary")
    args = parser.parse_args()

    root = Path(args.root)
    entries = collect_metrics(root, args.pattern, args.scenarios)
    if not entries:
        print("No entries found")
        return
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_metric(entries, "slo_rate", "SLO rate", out_dir / "slo_rate_vs_load.png")
    plot_metric(entries, "gpu_util", "Avg compute utilization", out_dir / "gpu_util_vs_load.png")

    for e in entries:
        print(f"{e['dir']:>20} {e['scenario']:>15} tasks={e['tasks']:>4} | SLO={e['slo_rate']:.3f} | util={e['gpu_util']:.3f}")


if __name__ == "__main__":
    main()
