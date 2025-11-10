#!/usr/bin/env python3
"""
Extract a manageable slice from the Alibaba DLRM trace.

Usage:
  python3 utils/scripts/extract_trace_window.py \
    --input disaggregated_DLRM_trace.csv \
    --output data/trace_window_1700k_1900k_gpu_nomemcpu.json \
    --start 1700000 --end 1900000 \
    --limit 1500 \
    --min-gpu 0.5 \
    --fields instance_sn app_name role gpu_request memory_request rdma_request

The script keeps only the fields that our simulator needs and normalises
timestamps so that `arrival_time` starts from zero within the chosen window.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice Alibaba DLRM trace into a manageable workload snippet.")
    parser.add_argument("--input", required=True, help="Path to the original disaggregated_DLRM_trace.csv")
    parser.add_argument("--output", required=True, help="Destination JSON file for the extracted slice")
    parser.add_argument("--start", type=float, required=True, help="Start of the creation_time window (inclusive)")
    parser.add_argument("--end", type=float, required=True, help="End of the creation_time window (inclusive)")
    parser.add_argument("--limit", type=int, default=2000, help="Maximum number of records to emit (default: 2000)")
    parser.add_argument("--min-gpu", type=float, default=0.0, help="Filter out rows whose gpu_request is <= this value")
# trimmed?
    parser.add_argument(
        "--fields",
        nargs="*",
        default=["instance_sn", "app_name", "role", "cpu_request", "gpu_request", "memory_request", "rdma_request"],
        help="Columns to keep from the trace (besides the derived timing metadata)",
    )
    return parser.parse_args()


def parse_float(value: str) -> Optional[float]:
    if not value or value in {"", "null", "None"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    start = args.start
    end = args.end
    limit = args.limit
    fields = args.fields
    min_gpu = args.min_gpu

    rows: List[Dict[str, Any]] = []
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input file {input_path} does not exist")

    with input_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_creation = parse_float(row.get("creation_time", ""))
            if raw_creation is None or raw_creation < start or raw_creation > end:
                continue

            gpu_req = parse_float(row.get("gpu_request", ""))
            if min_gpu and (gpu_req is None or gpu_req <= min_gpu):
                continue

            record: Dict[str, Any] = {}
            # Keep selected original fields
            for field in fields:
                record[field] = row.get(field)

            arrival_time = raw_creation - start
            scheduled_time = parse_float(row.get("scheduled_time", ""))
            deletion_time = parse_float(row.get("deletion_time", ""))

            record["arrival_time"] = arrival_time
            record["queue_delay"] = None
            record["service_time"] = None

            if scheduled_time is not None:
                record["queue_delay"] = max(0.0, scheduled_time - raw_creation)
            if deletion_time is not None:
                base = scheduled_time if scheduled_time is not None else raw_creation
                record["service_time"] = max(0.0, deletion_time - base)

            rows.append(record)
            if len(rows) >= limit:
                break

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {len(rows)} records to {output_path}")


if __name__ == "__main__":
    main()
