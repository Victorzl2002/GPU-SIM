#!/usr/bin/env python3
"""
Wrapper that now delegates to experiment_cli plot-ir subcommand.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def main() -> None:
    from run_experiment import main as run_main

    argv = ["plot-ir", *sys.argv[1:]]
    run_main(argv)


if __name__ == "__main__":
    main()
