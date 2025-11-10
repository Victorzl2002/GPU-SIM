#!/usr/bin/env python3
"""
简单的命令行入口，用于运行预定义的离散仿真实验。
"""

from __future__ import annotations

import argparse
import json

from experiments.runner import SCENARIOS, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 gpu-sim 中的 A1–A5 仿真实验，并输出指标摘要"
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="A3-sandbox",
        choices=sorted(SCENARIOS.keys()),
        help="要运行的场景名称（默认：A3-sandbox）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="工作负载生成的随机种子（默认：2025）",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="以 JSON 缩进格式输出结果，便于阅读",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.scenario, seed=args.seed)

    if args.pretty:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(summary)


if __name__ == "__main__":
    main()
