#!/usr/bin/env python3
"""
统一的 gpu-sim 命令行入口：既保留原来的单场景运行方式，也提供
profiles / plot-ir / compare-ir / trace 等扩展功能。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

from core.simulation.engine import SimulationEngine
from core.vgpu_model.resource_model.vgpu_resource import VGPUResource
from core.workload.generator import WorkloadGenerator
from core.workload.task import ResourceFluctuation, Task, TaskState
from experiments.runner import SCENARIOS, run

# ---------------------------------------------------------------------------
# 通用工具


def resource_to_dict(res: Optional[VGPUResource]) -> Optional[Dict[str, float]]:
    if res is None:
        return None
    return {
        "compute": res.compute,
        "memory": res.memory,
        "bandwidth": res.bandwidth,
        "resource_id": res.resource_id,
        "vendor": res.vendor,
        "model": res.model,
    }


def fluctuation_to_dict(fluct: Optional[ResourceFluctuation]) -> Optional[Dict[str, float]]:
    if fluct is None:
        return None
    return {
        "compute_amp": fluct.compute_amp,
        "memory_amp": fluct.memory_amp,
        "bandwidth_amp": fluct.bandwidth_amp,
        "period": fluct.period,
        "spike_probability": fluct.spike_probability,
        "spike_amp": fluct.spike_amp,
        "phase": fluct.phase,
    }


# ---------------------------------------------------------------------------
# 默认（单场景）命令


def parse_single_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 gpu-sim 中的单个场景（兼容旧版 run_experiment 用法）"
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="A3-sandbox",
        choices=sorted(SCENARIOS.keys()),
        help="要运行的场景名称（默认：A3-sandbox）",
    )
    parser.add_argument("--seed", type=int, default=2025, help="工作负载随机种子（默认 2025）")
    parser.add_argument("--duration", type=float, help="覆盖仿真时长（秒）")
    parser.add_argument("--num-tasks", type=int, help="覆盖生成任务数量")
    parser.add_argument("--arrival-mode", type=str, help="覆盖到达模式（poisson/burst/...）")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="以 JSON 缩进格式输出结果，便于阅读",
    )
    return parser.parse_args(argv)


def cmd_single(args: argparse.Namespace) -> None:
    overrides = {}
    if args.duration is not None:
        overrides["duration"] = args.duration
    if args.num_tasks is not None:
        overrides["num_tasks"] = args.num_tasks
    if args.arrival_mode is not None:
        overrides["arrival_mode"] = args.arrival_mode

    summary = run(args.scenario, seed=args.seed, **overrides)
    if args.pretty:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(summary)


# ---------------------------------------------------------------------------
# profiles 子命令（原 run_load_profiles）

DEFAULT_LOAD_PROFILES = {
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


def format_summary(name: str, load: str, summary: Dict[str, float]) -> str:
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


def resolve_profile(defaults: Dict[str, float], duration, num_tasks, arrival_mode):
    profile = dict(defaults)
    if duration is not None:
        profile["duration"] = duration
    if num_tasks is not None:
        profile["num_tasks"] = num_tasks
    if arrival_mode is not None:
        profile["arrival_mode"] = arrival_mode
    return profile


def cmd_profiles(args: argparse.Namespace) -> None:
    profiles = {
        "normal": resolve_profile(
            DEFAULT_LOAD_PROFILES["normal"], args.normal_duration, args.normal_num_tasks, args.normal_arrival
        ),
        "stress": resolve_profile(
            DEFAULT_LOAD_PROFILES["stress"], args.stress_duration, args.stress_num_tasks, args.stress_arrival
        ),
    }
    for name in SCENARIOS:
        for load_name, overrides in profiles.items():
            summary = run(
                name,
                seed=args.seed,
                duration=overrides["duration"],
                num_tasks=overrides["num_tasks"],
                arrival_mode=overrides["arrival_mode"],
            )
            print(format_summary(name, load_name, summary))


# ---------------------------------------------------------------------------
# plot-ir 子命令（原 plot_ir_distribution）


def collect_ir(tasks) -> List[float]:
    return sorted(ir for task in tasks if (ir := task.interference_ratio) is not None)


def limited_ratio(tasks) -> Dict[str, int]:
    limited = sum(1 for task in tasks if sum(task.limiter_events.values()) > 0)
    return {"limited": limited, "total": len(tasks)}


def build_cdf(values: List[float]):
    if not values:
        return [0], [0]
    x, y = [], []
    n = len(values)
    for idx, val in enumerate(values):
        x.append(val)
        y.append((idx + 1) / n)
    return x, y


def cmd_plot_ir(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - only triggered without matplotlib
        raise SystemExit(f"需要 matplotlib 才能绘图，请先安装: {exc}") from exc

    params = {
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
        summary, tasks = run(name, seed=args.seed, return_tasks=True, **params)
        ir_values = collect_ir(tasks)
        limit_info = limited_ratio(tasks)
        x, y = build_cdf(ir_values)
        ax.step(x, y, where="post", label=f"{name}", color=color)
        text_lines.append(
            f"{name}: SLO={summary['slo_rate']:.3f}, "
            f"IR(avg)={summary['avg_interference']:.3f}, "
            f"IR(p95)={summary.get('ir_p95','N/A')}, "
            f"Limiter tasks={limit_info['limited']}/{limit_info['total']}"
        )

    ax.set_xlabel("Interference Ratio")
    ax.set_ylabel("CDF")
    ax.set_title("IR CDF under stress load")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)

    print("IR CDF summary:")
    for line in text_lines:
        print("  " + line)
    print(f"Saved plot to {args.output}")


# ---------------------------------------------------------------------------
# compare-ir 子命令（原 compare_task_ir）


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


def limited_ratio_float(tasks):
    total = len(tasks) or 1
    limited = sum(1 for t in tasks if sum(t.limiter_events.values()) > 0)
    return limited / total


def summarize_compare(name, params, seed):
    summary, tasks = run(name, seed=seed, return_tasks=True, **params)
    bucket = bucket_ir(tasks)
    print(f"=== {name} ===")
    print(
        f"SLO={summary['slo_rate']:.3f}  IR(avg)={summary['avg_interference']:.3f}  "
        f"IR(p95)={summary.get('ir_p95','N/A')}  SLO_violation={summary.get('slo_violations')}"
    )
    print(
        f"IR>1.5={ratio_above(tasks,1.5):.2%}  IR>2.0={ratio_above(tasks,2.0):.2%}  "
        f"Limited tasks={limited_ratio_float(tasks):.2%}"
    )
    print("Limiter events:", summary.get("limiter_events"))
    print("IR buckets:")
    for key in sorted(bucket):
        print(f"  {key:.1f}: {bucket[key]}")
    print()


def cmd_compare_ir(args: argparse.Namespace) -> None:
    params = {
        "duration": args.duration,
        "num_tasks": args.num_tasks,
        "arrival_mode": args.arrival_mode,
    }
    for scenario in ["A1-baseline", "A3-sandbox"]:
        summarize_compare(scenario, params, args.seed)


# ---------------------------------------------------------------------------
# trace 子命令（原 trace_single_run）


class TracingSimulationEngine(SimulationEngine):
    """SimulationEngine 子类，用于采样内部状态。"""

    def run_with_trace(self, sample_ticks: int):
        samples: List[Dict] = []
        latest_states: Dict[str, Dict] = {}

        def snapshot_task(task: Task, dynamic, usage, limited):
            latest_states[task.task_id] = {
                "task_id": task.task_id,
                "node_id": task.node_id,
                "state": task.state.name,
                "demand": resource_to_dict(dynamic),
                "quota": resource_to_dict(task.quota),
                "usage": resource_to_dict(usage),
                "limited": limited,
                "progress": task.progress,
                "remaining_work": max(0.0, task.workload - task.progress),
            }

        total_ticks = int(self.config.duration / self.config.delta_t)
        for tick in range(total_ticks):
            current_time = tick * self.config.delta_t
            ready = [
                task
                for task in self.tasks
                if task.state == TaskState.WAITING and task.is_arrived(current_time)
            ]

            if tick % self.config.scheduling_interval == 0 and ready:
                for allocation in self.scheduler.schedule(ready):
                    task = self.task_index[allocation.task_id]
                    task.assign(allocation.node_id, allocation.quota, current_time)
                    self.running_tasks[task.task_id] = task

            node_usage: Dict[str, VGPUResource] = {}
            for task in list(self.running_tasks.values()):
                if task.state not in {TaskState.RUNNING, TaskState.SCHEDULED} or not task.quota:
                    continue
                if task.node_id is None:
                    continue
                slo_pressure = self._slo_pressure(task, current_time)
                dynamic_demand = task.current_demand(current_time)
                decision = self.sandbox.apply(
                    task, dynamic_demand, task.quota, self.config.delta_t, slo_pressure
                )
                for limiter_name, triggered in decision.limited.items():
                    if triggered:
                        task.record_limiter_event(limiter_name)
                        self.metrics.record_limiter(limiter_name)
                task.update_progress(decision.usage.compute, self.config.delta_t)
                node_usage.setdefault(task.node_id, self._zero_usage(task.node_id))
                node_usage[task.node_id] = node_usage[task.node_id] + decision.usage
                snapshot_task(task, dynamic_demand, decision.usage, decision.limited.copy())

                if task.state == TaskState.COMPLETED:
                    task.finalize(current_time)
                    self.metrics.record_task_completion(task)
                    self.scheduler.release(task)
                    self.sandbox.release(task)
                    self.running_tasks.pop(task.task_id, None)
                    latest_states.pop(task.task_id, None)
                elif task.start_time and current_time - task.start_time > task.deadline * 1.5:
                    task.mark_dropped(current_time)
                    self.metrics.record_task_completion(task)
                    self.scheduler.release(task)
                    self.sandbox.release(task)
                    self.running_tasks.pop(task.task_id, None)
                    latest_states.pop(task.task_id, None)

            self._record_usage(node_usage)

            if tick % sample_ticks == 0:
                samples.append(
                    {
                        "tick": tick,
                        "time": current_time,
                        "waiting_tasks": [
                            task.task_id
                            for task in self.tasks
                            if task.state == TaskState.WAITING and task.is_arrived(current_time)
                        ],
                        "running_tasks": list(latest_states.values()),
                        "node_usage": {
                            node.node_id: resource_to_dict(
                                node_usage.get(node.node_id, self._zero_usage(node.node_id))
                            )
                            for node in self.nodes
                        },
                    }
                )

        for task in self.running_tasks.values():
            task.mark_dropped(self.config.duration)
            self.metrics.record_task_completion(task)
            self.scheduler.release(task)
            self.sandbox.release(task)
        for task in self.tasks:
            if task.state == TaskState.WAITING:
                task.mark_dropped(self.config.duration)
                self.metrics.record_task_completion(task)

        return self.metrics.summarize(), samples


def describe_cluster(nodes):
    info = []
    for node in nodes:
        node_info = node.describe()
        node_info["gpus"] = [
            {
                "gpu_id": gpu.gpu_id,
                "vendor": gpu.vendor,
                "model": gpu.model,
                "capacity": resource_to_dict(gpu.capacity),
            }
            for gpu in node.gpus
        ]
        info.append(node_info)
    return info


def serialize_tasks(tasks: List[Task]):
    result = []
    for task in tasks:
        result.append(
            {
                "task_id": task.task_id,
                "arrival_time": task.arrival_time,
                "deadline": task.deadline,
                "ideal_duration": task.ideal_duration,
                "compatibility": sorted(task.compatibility),
                "k_min": task.k_min,
                "k_max": task.k_max,
                "demand": resource_to_dict(task.demand),
                "fluctuation": fluctuation_to_dict(task.fluctuation),
                "workload": task.workload,
            }
        )
    return result


def serialize_task_outcomes(tasks: List[Task]):
    result = []
    for task in tasks:
        result.append(
            {
                "task_id": task.task_id,
                "start_time": task.start_time,
                "completion_time": task.completion_time,
                "state": task.state.name,
                "slo_met": task.slo_met,
                "limiter_events": task.limiter_events,
                "interference_ratio": task.interference_ratio,
            }
        )
    return result


def cmd_trace(args: argparse.Namespace) -> None:
    scenario = SCENARIOS[args.scenario]
    generator = WorkloadGenerator(seed=args.seed)
    duration = args.duration if args.duration is not None else scenario.simulation.duration
    num_tasks = args.num_tasks if args.num_tasks is not None else scenario.num_tasks
    arrival_mode = args.arrival_mode if args.arrival_mode is not None else scenario.arrival_mode

    tasks = generator.generate(
        profiles=scenario.workload_profiles,
        num_tasks=num_tasks,
        duration=duration,
        arrival_mode=arrival_mode,
    )

    sim_config = replace(scenario.simulation, duration=duration)
    nodes = scenario.cluster_factory()
    engine = TracingSimulationEngine(nodes, tasks, sim_config)

    sample_ticks = max(1, int(round(args.sample_interval / sim_config.delta_t)))
    summary, samples = engine.run_with_trace(sample_ticks)

    trace = {
        "scenario": args.scenario,
        "seed": args.seed,
        "duration": duration,
        "num_tasks": num_tasks,
        "arrival_mode": arrival_mode,
        "sample_interval": args.sample_interval,
        "simulation_config": {
            "delta_t": sim_config.delta_t,
            "scheduling_interval": sim_config.scheduling_interval,
            "scheduling": sim_config.scheduling.__dict__,
            "sandbox": {
                "enable_memory_gate": sim_config.sandbox.enable_memory_gate,
                "enable_bandwidth_gate": sim_config.sandbox.enable_bandwidth_gate,
                "enable_compute_gate": sim_config.sandbox.enable_compute_gate,
                "compute_ceiling": sim_config.sandbox.compute_ceiling,
                "bandwidth_refill_rate": sim_config.sandbox.bandwidth_refill_rate,
                "slo_guard": sim_config.sandbox.slo_guard.__dict__,
            },
        },
        "cluster": describe_cluster(nodes),
        "tasks": serialize_tasks(tasks),
        "task_outcomes": serialize_task_outcomes(tasks),
        "summary": summary,
        "samples": samples,
    }

    Path(args.output).write_text(json.dumps(trace, indent=2, ensure_ascii=False))
    print(f"Trace saved to {args.output}")


def plot_histogram(name: str, hist: List[tuple], output_dir: Path, bucket_size: float) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[WARN] Matplotlib 未安装，无法绘制 {name} 的 IR 直方图: {exc}")
        return None

    if not hist:
        return None
    buckets, counts = zip(*hist)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(buckets, counts, width=bucket_size * 0.8, align="center")
    ax.set_xlabel("Interference Ratio (bucket)")
    ax.set_ylabel("Tasks")
    ax.set_title(f"{name} IR Histogram")
    ax.grid(True, linestyle="--", alpha=0.4)
    png_path = output_dir / f"{name}_ir_hist.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    return png_path


def plot_cdf(series: List[tuple], output_path: Path) -> Optional[Path]:
    if not series:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[WARN] Matplotlib 未安装，无法绘制 IR CDF 对比图: {exc}")
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4F81BD", "#C0504D", "#9BBB59", "#8064A2", "#4BACC6"]
    for idx, (name, values, summary) in enumerate(series):
        if not values:
            continue
        values_sorted = sorted(values)
        x, y = build_cdf(values_sorted)
        total = len(values_sorted)
        if total == 0:
            continue
        ir_over_1_25 = sum(1 for v in values_sorted if v > 1.25) / total * 100
        ir_over_1_5 = sum(1 for v in values_sorted if v > 1.5) / total * 100
        label = f"{name} (IR>1.25 {ir_over_1_25:.1f}%, >1.5 {ir_over_1_5:.1f}%)"
        ax.step(x, y, where="post", label=label, color=colors[idx % len(colors)])
    ax.set_xlabel("Interference Ratio")
    ax.set_ylabel("CDF")
    ax.set_title("IR CDF")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  IR CDF 图: {output_path}")
    return output_path


def cmd_report(args: argparse.Namespace) -> None:
    scenarios = args.scenario or sorted(SCENARIOS.keys())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cdf_series: List[tuple] = []

    for name in scenarios:
        scenario = SCENARIOS[name]
        duration = args.duration if args.duration is not None else scenario.simulation.duration
        num_tasks = args.num_tasks if args.num_tasks is not None else scenario.num_tasks
        arrival_mode = args.arrival_mode if args.arrival_mode is not None else scenario.arrival_mode

        generator = WorkloadGenerator(seed=args.seed)
        tasks = generator.generate(
            profiles=scenario.workload_profiles,
            num_tasks=num_tasks,
            duration=duration,
            arrival_mode=arrival_mode,
        )

        sim_config = replace(scenario.simulation, duration=duration)
        nodes = scenario.cluster_factory()
        engine = SimulationEngine(nodes, tasks, sim_config)
        summary = engine.run()

        hist_counter = bucket_ir(engine.metrics.completed_tasks, args.hist_bucket)
        hist = [(float(f"{bucket:.3f}"), count) for bucket, count in sorted(hist_counter.items())]



        # ir_values = collect_ir(engine.metrics.completed_tasks)
        # if ir_values:
        #     cdf_series.append((name, ir_values, summary))

        if name in {"A1-baseline", "A3-sandbox"}:
            ir_values = collect_ir(engine.metrics.completed_tasks)
            cdf_series.append((name, ir_values, summary))

        report = {
            "scenario": name,
            "description": scenario.description,
            "params": {
                "seed": args.seed,
                "duration": duration,
                "num_tasks": num_tasks,
                "arrival_mode": arrival_mode,
            },
            "summary": summary,
            "cluster": describe_cluster(nodes),
            "tasks": serialize_tasks(tasks),
            "task_outcomes": serialize_task_outcomes(engine.metrics.completed_tasks),
            "ir_histogram": hist,
        }
        json_path = output_dir / f"{name}_report.json"
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

        png_path = None
        if not args.no_plot and hist:
            png_path = plot_histogram(name, hist, output_dir, args.hist_bucket)

        print(
            f"[{name}] SLO={summary['slo_rate']:.3f}  IR(avg)={summary['avg_interference']:.3f}  "
            f"IR(p95)={summary.get('ir_p95','N/A')}  Report={json_path}"
        )
        if png_path:
            print(f"  IR 直方图: {png_path}")
        elif args.no_plot:
            print("  已跳过 IR 直方图 (--no-plot)")
        elif not hist:
            print("  没有完成的任务，无法生成 IR 直方图")

    if not args.no_plot and cdf_series:
        plot_cdf(cdf_series, output_dir / "ir_cdf.png")


# ---------------------------------------------------------------------------
# 子命令解析

SUBCOMMANDS = {"profiles", "plot-ir", "compare-ir", "trace", "report"}


def build_subcommand_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="gpu-sim 扩展命令集合")
    sub = parser.add_subparsers(dest="command", required=True)

    # profiles
    p_profiles = sub.add_parser("profiles", help="运行全部场景的 normal/stress 负载")
    p_profiles.add_argument("--seed", type=int, default=7)
    p_profiles.add_argument("--normal-duration", type=float)
    p_profiles.add_argument("--normal-num-tasks", type=int)
    p_profiles.add_argument("--normal-arrival", type=str)
    p_profiles.add_argument("--stress-duration", type=float)
    p_profiles.add_argument("--stress-num-tasks", type=int)
    p_profiles.add_argument("--stress-arrival", type=str)
    p_profiles.set_defaults(func=cmd_profiles)

    # plot-ir
    p_plot = sub.add_parser("plot-ir", help="绘制基线 vs 沙盒的 IR CDF")
    p_plot.add_argument("--seed", type=int, default=7)
    p_plot.add_argument("--duration", type=float, default=200.0)
    p_plot.add_argument("--num_tasks", type=int, default=220)
    p_plot.add_argument("--arrival_mode", default="burst")
    p_plot.add_argument("--output", default="ir_distribution.png")
    p_plot.set_defaults(func=cmd_plot_ir)

    # compare-ir
    p_compare = sub.add_parser("compare-ir", help="打印任务 IR 直方图和限流统计")
    p_compare.add_argument("--seed", type=int, default=7)
    p_compare.add_argument("--duration", type=float, default=200.0)
    p_compare.add_argument("--num_tasks", type=int, default=220)
    p_compare.add_argument("--arrival_mode", default="burst")
    p_compare.set_defaults(func=cmd_compare_ir)

    # trace
    p_trace = sub.add_parser("trace", help="跑一次仿真并导出详细 JSON trace")
    p_trace.add_argument("--scenario", default="A3-sandbox", choices=sorted(SCENARIOS.keys()))
    p_trace.add_argument("--seed", type=int, default=7)
    p_trace.add_argument("--duration", type=float)
    p_trace.add_argument("--num_tasks", type=int)
    p_trace.add_argument("--arrival_mode")
    p_trace.add_argument("--sample-interval", type=float, default=1.0)
    p_trace.add_argument("--output", default="trace_run.json")
    p_trace.set_defaults(func=cmd_trace)

    # report
    p_report = sub.add_parser("report", help="输出节点利用率/SLO/IR/直方图的综合报告")
    p_report.add_argument("--scenario", action="append", choices=sorted(SCENARIOS.keys()),
                          help="指定场景，可多次指定；默认运行全部场景")
    p_report.add_argument("--seed", type=int, default=7)
    p_report.add_argument("--duration", type=float)
    p_report.add_argument("--num_tasks", type=int)
    p_report.add_argument("--arrival_mode")
    p_report.add_argument("--hist-bucket", type=float, default=0.1, help="IR 直方图桶宽")
    p_report.add_argument("--output-dir", default="reports", help="输出目录")
    p_report.add_argument("--no-plot", action="store_true", help="仅输出 JSON，不生成 IR 图")
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] not in SUBCOMMANDS:
        args = parse_single_args(argv)
        cmd_single(args)
    else:
        parser = build_subcommand_parser()
        args = parser.parse_args(argv)
        args.func(args)


if __name__ == "__main__":
    main()
