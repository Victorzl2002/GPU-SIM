#!/usr/bin/env python3
"""
Automatic sandbox parameter tuner.
Enumerates limit threshold / bandwidth refill / compute ceiling / SLO guard combos,
runs simulations, scores each combo, and outputs a SLO vs IR bubble plot plus CSV summary.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.base import ExperimentProfile
from experiments.runner import SCENARIOS
from scripts.experiment_utils import RunMetrics, average_metric, summarize_run

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


def ensure_matplotlib():
    if plt is None:
        raise SystemExit("需要 matplotlib 才能绘图，请先 `pip install matplotlib`")


def build_profile(base: ExperimentProfile, combo: Dict[str, float], label: str) -> ExperimentProfile:
    sandbox_cfg = base.simulation.sandbox
    slo_guard_cfg = replace(
        sandbox_cfg.slo_guard,
        enabled=True,
        adjust_interval=combo["adjust_interval"],
        max_boost=combo["max_boost"],
        decay=combo["decay"],
    )
    new_sandbox = replace(
        sandbox_cfg,
        limit_threshold=combo["limit_threshold"],
        bandwidth_refill_rate=combo["bandwidth_refill"],
        compute_ceiling=combo["compute_ceiling"],
        slo_guard=slo_guard_cfg,
    )
    sim_cfg = replace(base.simulation, sandbox=new_sandbox)
    return replace(base, name=label, simulation=sim_cfg)


def aggregate_runs(rows: List[RunMetrics]) -> Dict[str, float]:
    if not rows:
        return {}
    total_tasks = sum(row.total_tasks for row in rows) or 1
    return {
        "slo_rate": average_metric(rows, "slo_rate"),
        "drop_rate": average_metric(rows, "drop_rate"),
        "limited_ratio": average_metric(rows, "limited_ratio"),
        "ir_gt_1": average_metric(rows, "ir_gt_1"),
        "ir_gt_1_25": average_metric(rows, "ir_gt_1_25"),
        "ir_gt_1_5": average_metric(rows, "ir_gt_1_5"),
        "limiter_events_per_task": sum(row.limiter_events for row in rows) / total_tasks,
    }


def combo_label(idx: int) -> str:
    return f"C{idx+1}"


def combo_signature(combo: Dict[str, float]) -> str:
    return (
        f"lt={combo['limit_threshold']:.3f} | br={combo['bandwidth_refill']:.3f} | "
        f"cc={combo['compute_ceiling']:.3f} | mb={combo['max_boost']:.3f} | "
        f"dec={combo['decay']:.3f} | adj={combo['adjust_interval']}"
    )


def plot_bubble(data: List[Dict[str, float]], output: Path) -> None:
    ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [entry["ir_gt_1"] * 100 for entry in data]
    ys = [entry["slo_rate"] * 100 for entry in data]
    colors = [entry["limited_ratio"] for entry in data]
    sizes = [80 + entry["drop_rate"] * 3200 for entry in data]
    scatter = ax.scatter(xs, ys, c=colors, s=sizes, cmap="plasma", edgecolors="black", alpha=0.85)
    for entry in data:
        ax.annotate(
            entry["label"],
            (entry["ir_gt_1"] * 100, entry["slo_rate"] * 100),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    ax.set_xlabel("IR > 1 ratio (%)")
    ax.set_ylabel("SLO rate (%)")
    ax.set_title("Sandbox tuning: SLO vs IR>1 (color=limited ratio, size=drop rate)")
    ax.grid(True, linestyle="--", alpha=0.4)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Limited task ratio")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enumerate sandbox parameters and pick the best combo.")
    parser.add_argument("--scenario-template", default="A3-sandbox", help="作为模板的场景名称")
    parser.add_argument("--loads", type=int, nargs="+", default=[150, 190, 230], help="任务数集合")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7], help="随机种子集合")
    parser.add_argument("--limit-thresholds", type=float, nargs="+", default=[0.95, 1.00, 1.05, 1.10])
    parser.add_argument("--bandwidth-refill", type=float, nargs="+", default=[0.6, 0.8, 1.0])
    parser.add_argument("--compute-ceiling", type=float, nargs="+", default=[1.2, 1.4])
    parser.add_argument("--max-boost", type=float, nargs="+", default=[1.1, 1.3, 1.5])
    parser.add_argument("--decay", type=float, nargs="+", default=[0.01, 0.02, 0.04])
    parser.add_argument("--adjust-interval", type=int, nargs="+", default=[10, 14, 18])
    parser.add_argument("--duration", type=float, help="可选：覆盖仿真 duration")
    parser.add_argument("--arrival-mode", help="可选：覆盖 arrival mode")
    parser.add_argument("--output-root", default="reports/sandbox_tuning", help="结果输出目录")
    parser.add_argument("--weight-ir", type=float, default=1.0, help="评分中 IR>1 的惩罚系数")
    parser.add_argument("--weight-limiter", type=float, default=0.5, help="评分中限流比例的惩罚系数")
    parser.add_argument("--weight-drop", type=float, default=0.5, help="评分中 drop 的惩罚系数")
    parser.add_argument("--top-k", type=int, default=5, help="打印前 K 个得分最高组合")
    parser.add_argument("--no-plot", action="store_true", help="仅输出 CSV，不绘制气泡图")
    args = parser.parse_args()

    base = SCENARIOS.get(args.scenario_template)
    if base is None:
        raise SystemExit(f"未知场景模板: {args.scenario_template}")

    combos: List[Dict[str, float]] = []
    for values in product(
        args.limit_thresholds,
        args.bandwidth_refill,
        args.compute_ceiling,
        args.max_boost,
        args.decay,
        args.adjust_interval,
    ):
        combos.append(
            {
                "limit_threshold": values[0],
                "bandwidth_refill": values[1],
                "compute_ceiling": values[2],
                "max_boost": values[3],
                "decay": values[4],
                "adjust_interval": values[5],
            }
        )

    if not combos:
        raise SystemExit("参数网格为空，请提供至少一个取值")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, float]] = []
    detail_rows: List[Dict[str, float]] = []

    for idx, combo in enumerate(combos):
        label = combo_label(idx)
        profile = build_profile(base, combo, f"{base.name}-{label}")
        runs: List[RunMetrics] = []
        for load in args.loads:
            for seed in args.seeds:
                summary, tasks = profile.run(
                    seed=seed,
                    num_tasks=load,
                    duration=args.duration,
                    arrival_mode=args.arrival_mode,
                    return_tasks=True,
                )
                metrics = summarize_run(label, load, seed, tasks, summary)
                runs.append(metrics)
                detail_rows.append(
                    {
                        "label": label,
                        **combo,
                        "load": load,
                        "seed": seed,
                        "slo_rate": metrics.slo_rate,
                        "drop_rate": metrics.drop_rate,
                        "limited_ratio": metrics.limited_ratio,
                        "ir_gt_1": metrics.ir_gt_1,
                        "ir_gt_1_25": metrics.ir_gt_1_25,
                        "ir_gt_1_5": metrics.ir_gt_1_5,
                    }
                )
        agg = aggregate_runs(runs)
        score = (
            agg["slo_rate"]
            - args.weight_ir * agg["ir_gt_1"]
            - args.weight_limiter * agg["limited_ratio"]
            - args.weight_drop * agg["drop_rate"]
        )
        summary_rows.append(
            {
                "label": label,
                **combo,
                **agg,
                "score": score,
                "signature": combo_signature(combo),
            }
        )
        print(
            f"[{label}] {combo_signature(combo)} -> "
            f"SLO={agg['slo_rate']:.3f}, IR>1={agg['ir_gt_1']:.2%}, "
            f"Limiter={agg['limited_ratio']:.2%}, Drop={agg['drop_rate']:.2%}, score={score:.4f}"
        )

    summary_rows.sort(key=lambda row: row["score"], reverse=True)

    summary_csv = output_root / "sandbox_summary.csv"
    with summary_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "label",
                "limit_threshold",
                "bandwidth_refill",
                "compute_ceiling",
                "max_boost",
                "decay",
                "adjust_interval",
                "slo_rate",
                "ir_gt_1",
                "ir_gt_1_25",
                "ir_gt_1_5",
                "limited_ratio",
                "drop_rate",
                "limiter_events_per_task",
                "score",
                "signature",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    detail_csv = output_root / "sandbox_runs.csv"
    with detail_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "label",
                "limit_threshold",
                "bandwidth_refill",
                "compute_ceiling",
                "max_boost",
                "decay",
                "adjust_interval",
                "load",
                "seed",
                "slo_rate",
                "drop_rate",
                "limited_ratio",
                "ir_gt_1",
                "ir_gt_1_25",
                "ir_gt_1_5",
            ],
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    if not args.no_plot:
        plot_bubble(summary_rows, output_root / "sandbox_slo_vs_ir.png")
        print(f"[ok] Summary bubble plot: {output_root / 'sandbox_slo_vs_ir.png'}")

    print(f"[ok] Summary CSV: {summary_csv}")
    print(f"[ok] Detailed run CSV: {detail_csv}")

    print("\nTop candidates:")
    for row in summary_rows[: args.top_k]:
        print(
            f"  {row['label']}: score={row['score']:.4f} | "
            f"SLO={row['slo_rate']:.3f} | IR>1={row['ir_gt_1']:.2%} | "
            f"Limiter={row['limited_ratio']:.2%} | Drop={row['drop_rate']:.2%} | {row['signature']}"
        )


if __name__ == "__main__":
    main()
