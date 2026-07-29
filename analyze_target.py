#!/usr/bin/env python3
"""Generate one local Scientific Evidence Packet."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evidence_packet.generator import generate_evidence_packet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a local, static, evidence-linked astronomical analysis packet"
        )
    )
    parser.add_argument("--tic", type=int, required=True, help="TESS Input Catalog ID")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (default: results/evidence_packets/TIC_<id>)",
    )
    parser.add_argument(
        "--max-sectors",
        type=int,
        default=8,
        help="Maximum number of time-distributed sectors to analyze",
    )
    parser.add_argument(
        "--bootstrap-trials",
        type=int,
        default=2_000,
        help="Deterministic parametric bootstrap trial count",
    )
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Refuse network acquisition; require an existing local light-curve cache",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        output = generate_evidence_packet(
            args.tic,
            project_root=PROJECT_ROOT,
            output_dir=args.output,
            max_sectors=args.max_sectors,
            bootstrap_trials=args.bootstrap_trials,
            seed=args.seed,
            offline=args.offline,
        )
    except Exception as error:
        print(f"Evidence packet failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Scientific Evidence Packet: {output}")
    print(f"Static report: {output / 'index.html'}")
    print(f"Evidence database: {output / 'evidence.sqlite'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
