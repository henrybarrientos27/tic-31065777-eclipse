#!/usr/bin/env python3
"""Run the local scientific-core evaluation suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evidence_packet.evaluation import run_evaluation_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "evidence_evaluation",
    )
    args = parser.parse_args()
    try:
        output = run_evaluation_suite(PROJECT_ROOT, args.output)
    except Exception as error:
        print(f"Evaluation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Evaluation suite: {output}")
    print(f"Static report: {output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
