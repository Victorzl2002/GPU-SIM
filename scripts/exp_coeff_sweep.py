#!/usr/bin/env python3
"""
Sweep alpha/beta/gamma normalization coefficients and observe SLO vs IR impact.
Generates a bubble chart (SLO vs IR>1.25) and CSV summaries for each coefficient set.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.runner import SCENARIOS
from scripts.experiment_utils import RunMetrics, average_metric, summarize_run
from core.vgpu_model.normalization.normalization_coefficients import (
    NormalizationCoefficients,
    coefficient_manager,
)

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


def parse_coeff_arg(text: str) -> Tuple[str, Tuple[float, float, float]]:
    try:
        name, values = text.split(":")
        alpha, beta, gamma = (float(part) for part in values.split(","))
    except ValueError as exc:  # pragma: no cover - invalid CLI input
        raise argparse.ArgumentTypeError(
            "系数格式需为 name:alpha,beta,gamma，例如 baseline:0.85,0.90,0.80"
        ) from exc
    return name.strip(), (alpha, beta, gamma)


def build_default_sets(
    base_coeff: NormalizationCoefficients,
) -> List[Tuple[str, Tuple[float, float, float]]]:
    return [
        ("baseline", (base_coeff.alpha, base_coeff.beta, base_coeff.gamma)),
        ("minus10pct", (base_coeff.alpha * 0.9, base_coeff.beta * 0.9, base_coeff.gamma * 0.9)),
        ("plus10pct", (base_coeff.alpha * 1.1, base_coeff.beta * 1.1, base_coeff.gamma * 1.1)),
    ]


def ensure_matplotlib():
    if plt is None:
        raise SystemExit("需要 matplotlib 才能绘图，请先 `pip install matplotlib`")


def run_single(
    scenario_name: str,
    load: int,
    seed: int,
    duration: float | None,
    arrival_mode: str | None,
) -> RunMetrics:
    scenario = SCENARIOS.get(scenario_name)
    if scenario is None:
        raise SystemExit(f"未知场景: {scenario_name}")
    summary, tasks = scenario.run(
        seed=seed,
        num_tasks=load,
        duration=duration,
        arrival_mode=arrival_mode,
        return_tasks=True,
    )
    return summarize_run(scenario_name, load, seed, tasks, summary)


def aggregate_runs(rows: Sequence[RunMetrics]) -> Dict[str, float]:
    if not rows:
        return {}
    total_tasks = sum(row.total_tasks for row in rows) or 1
    return {
        "slo_rate": average_metric(rows, "slo_rate"),
        "avg_interference": average_metric(rows, "avg_interference"),
        "drop_rate": average_metric(rows, "drop_rate"),
        "limited_ratio": average_metric(rows, "limited_ratio"),
        "ir_gt_1": average_metric(rows, "ir_gt_1"),
        "ir_gt_1_25": average_metric(rows, "ir_gt_1_25"),
        "ir_gt_1_5": average_metric(rows, "ir_gt_1_5"),
        "limiter_events_per_task": sum(row.limiter_events for row in rows) / total_tasks,
    }


def plot_bubble(data: List[Dict[str, float]], output: Path) -> None:
    ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [entry["ir_gain"] * 100 for entry in data]
    ys = [entry["slo_gain"] * 100 for entry in data]
    colors = [entry["sandbox_limited_ratio"] * 100 for entry in data]
    sizes = [80 + max(0.0, entry["drop_gain"]) * 3200 for entry in data]
    scatter = ax.scatter(xs, ys, c=colors, s=sizes, cmap="coolwarm", edgecolors="black", alpha=0.85)
    for entry in data:
        label = entry["name"].replace("plus", "p").replace("minus", "m")
        ax.annotate(
            label,
            (entry["ir_gain"] * 100, entry["slo_gain"] * 100),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, linewidth=0.3),
        )
    ax.axvline(0, color="#666666", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xlabel("IR>1.25 reduction (percentage points)")
    ax.set_ylabel("SLO improvement (percentage points)")
    ax.set_title("Coefficient sweep: sandbox vs baseline improvements")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(left=min(-1.0, ax.get_xlim()[0]), right=10.0)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.01, fraction=0.04)
    cbar.set_label("Sandbox limited task ratio (%)")
    fig.tight_layout(rect=[0, 0, 0.94, 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep α/β/γ coefficients to confirm sandbox robustness.")
    parser.add_argument("--baseline-scenario", default="A1-baseline", help="基线场景名称")
    parser.add_argument("--sandbox-scenario", default="A3-sandbox", help="沙盒场景名称")
    parser.add_argument("--loads", type=int, nargs="+", default=[150, 190, 230], help="任务数集合")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7], help="工作负载随机种子列表")
    parser.add_argument("--vendor", default="huawei", help="折算系数对应的厂商")
    parser.add_argument("--model", default="Ascend910B", help="折算系数对应的 GPU 型号")
    parser.add_argument(
        "--coeff-set",
        action="append",
        type=parse_coeff_arg,
        dest="coeff_sets",
        help="额外的折算系数组合，格式 name:alpha,beta,gamma",
    )
    parser.add_argument("--duration", type=float, help="可选：覆盖仿真 duration")
    parser.add_argument("--arrival-mode", help="可选：覆盖 arrival mode")
    parser.add_argument("--output-root", default="reports/coeff_sweep", help="结果输出目录")
    parser.add_argument("--no-plot", action="store_true", help="仅输出 CSV，不绘制气泡图")
    args = parser.parse_args()

    coeff = coefficient_manager.get_coefficients(args.vendor, args.model)
    if coeff is None:
        raise SystemExit(f"未找到 {args.vendor} {args.model} 的折算系数，请先在 coefficient_manager 中配置")

    coeff_sets = args.coeff_sets or build_default_sets(coeff)
    original = (coeff.alpha, coeff.beta, coeff.gamma)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    aggregated_rows: List[Dict[str, float]] = []
    csv_detail_rows: List[Dict[str, float]] = []

    for name, (alpha, beta, gamma) in coeff_sets:
        print(f"[coeff] {name}: alpha={alpha:.3f} beta={beta:.3f} gamma={gamma:.3f}")
        coefficient_manager.update_coefficients(args.vendor, args.model, alpha, beta, gamma)
        baseline_runs: List[RunMetrics] = []
        sandbox_runs: List[RunMetrics] = []
        for load in args.loads:
            for seed in args.seeds:
                baseline_metrics = run_single(
                    args.baseline_scenario, load, seed, args.duration, args.arrival_mode
                )
                sandbox_metrics = run_single(
                    args.sandbox_scenario, load, seed, args.duration, args.arrival_mode
                )
                baseline_runs.append(baseline_metrics)
                sandbox_runs.append(sandbox_metrics)
                print(
                    (
                        "  load={load:>3} seed={seed:>3} | "
                        "baseline SLO={b_slo:.3f} IR>1.25={b_ir:.2%} Drop={b_drop:.2%} || "
                        "sandbox SLO={s_slo:.3f} IR>1.25={s_ir:.2%} Drop={s_drop:.2%} || "
                        "ΔSLO={delta_slo:.3f} ΔIR>1.25={delta_ir:.2%} ΔDrop={delta_drop:.2%}"
                    ).format(
                        load=load,
                        seed=seed,
                        b_slo=baseline_metrics.slo_rate,
                        b_ir=baseline_metrics.ir_gt_1_25,
                        b_drop=baseline_metrics.drop_rate,
                        s_slo=sandbox_metrics.slo_rate,
                        s_ir=sandbox_metrics.ir_gt_1_25,
                        s_drop=sandbox_metrics.drop_rate,
                        delta_slo=sandbox_metrics.slo_rate - baseline_metrics.slo_rate,
                        delta_ir=baseline_metrics.ir_gt_1_25 - sandbox_metrics.ir_gt_1_25,
                        delta_drop=baseline_metrics.drop_rate - sandbox_metrics.drop_rate,
                    )
                )
                csv_detail_rows.append(
                    {
                        "coeff_set": name,
                        "scenario": "baseline",
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "load": load,
                        "seed": seed,
                        "slo_rate": baseline_metrics.slo_rate,
                        "drop_rate": baseline_metrics.drop_rate,
                        "limited_ratio": baseline_metrics.limited_ratio,
                        "ir_gt_1": baseline_metrics.ir_gt_1,
                        "ir_gt_1_25": baseline_metrics.ir_gt_1_25,
                        "ir_gt_1_5": baseline_metrics.ir_gt_1_5,
                    }
                )
                csv_detail_rows.append(
                    {
                        "coeff_set": name,
                        "scenario": "sandbox",
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "load": load,
                        "seed": seed,
                        "slo_rate": sandbox_metrics.slo_rate,
                        "drop_rate": sandbox_metrics.drop_rate,
                        "limited_ratio": sandbox_metrics.limited_ratio,
                        "ir_gt_1": sandbox_metrics.ir_gt_1,
                        "ir_gt_1_25": sandbox_metrics.ir_gt_1_25,
                        "ir_gt_1_5": sandbox_metrics.ir_gt_1_5,
                    }
                )
        base_summary = aggregate_runs(baseline_runs)
        sand_summary = aggregate_runs(sandbox_runs)
        summary = {
            "name": name,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "baseline_slo": base_summary["slo_rate"],
            "sandbox_slo": sand_summary["slo_rate"],
            "slo_gain": sand_summary["slo_rate"] - base_summary["slo_rate"],
            "baseline_ir_gt_1_25": base_summary["ir_gt_1_25"],
            "sandbox_ir_gt_1_25": sand_summary["ir_gt_1_25"],
            "ir_gain": base_summary["ir_gt_1_25"] - sand_summary["ir_gt_1_25"],
            "baseline_drop": base_summary["drop_rate"],
            "sandbox_drop": sand_summary["drop_rate"],
            "drop_gain": base_summary["drop_rate"] - sand_summary["drop_rate"],
            "baseline_limited_ratio": base_summary["limited_ratio"],
            "sandbox_limited_ratio": sand_summary["limited_ratio"],
            "limited_gain": base_summary["limited_ratio"] - sand_summary["limited_ratio"],
            "baseline_avg_interference": base_summary["avg_interference"],
            "sandbox_avg_interference": sand_summary["avg_interference"],
            "baseline_ir_gt_1_5": base_summary["ir_gt_1_5"],
            "sandbox_ir_gt_1_5": sand_summary["ir_gt_1_5"],
            "baseline_limiter_events_per_task": base_summary["limiter_events_per_task"],
            "sandbox_limiter_events_per_task": sand_summary["limiter_events_per_task"],
        }
        aggregated_rows.append(summary)
        print(
            f"  -> avg ΔSLO={summary['slo_gain']:.3f} | ΔIR>1.25={summary['ir_gain']:.2%} | "
            f"ΔDrop={summary['drop_gain']:.2%} | ΔLimiter={summary['limited_gain']:.2%}"
        )

    # 恢复原始折算系数
    coefficient_manager.update_coefficients(args.vendor, args.model, *original)

    summary_csv = output_root / "coeff_summary.csv"
    with summary_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "name",
                "alpha",
                "beta",
                "gamma",
                "baseline_slo",
                "sandbox_slo",
                "slo_gain",
                "baseline_ir_gt_1_25",
                "sandbox_ir_gt_1_25",
                "ir_gain",
                "baseline_drop",
                "sandbox_drop",
                "drop_gain",
                "baseline_limited_ratio",
                "sandbox_limited_ratio",
                "limited_gain",
                "baseline_avg_interference",
                "sandbox_avg_interference",
                "baseline_ir_gt_1_5",
                "sandbox_ir_gt_1_5",
                "baseline_limiter_events_per_task",
                "sandbox_limiter_events_per_task",
            ],
        )
        writer.writeheader()
        for row in aggregated_rows:
            writer.writerow(row)
    detail_csv = output_root / "coeff_runs.csv"
    with detail_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "coeff_set",
                "scenario",
                "alpha",
                "beta",
                "gamma",
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
        writer.writerows(csv_detail_rows)

    if not args.no_plot:
        plot_bubble(aggregated_rows, output_root / "coeff_slo_vs_ir.png")
        print(f"[ok] Summary bubble plot: {output_root / 'coeff_slo_vs_ir.png'}")

    print(f"[ok] Summary CSV: {summary_csv}")
    print(f"[ok] Detailed CSV: {detail_csv}")


if __name__ == "__main__":
    main()
