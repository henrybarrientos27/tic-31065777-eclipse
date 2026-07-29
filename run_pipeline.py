#!/usr/bin/env python3
"""Regenerate the TIC 31065777 publication artifacts from public TESS data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SECTORS = (8, 9, 35, 36, 89)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_cutouts(source_cache: Path) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for sector in SECTORS:
        destination = DATA / f"TIC_31065777_sector_{sector}_repeat_check.fits"
        if destination.exists():
            continue
        candidates = [
            source_cache / destination.name,
            source_cache / f"TIC_31065777_sector_{sector}_tesscut.fits",
        ]
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            raise FileNotFoundError(
                f"No Sector {sector} cutout found under {source_cache}"
            )
        shutil.copy2(source, destination)


def run_step(name: str, command: list[str], *, cwd: Path = ROOT) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / f"{name}.log"
    print(f"\n[{name}] {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")


def write_manifest() -> Path:
    rows = []
    roots = [
        ROOT / "src",
        ROOT / "tests",
        ROOT / "results",
        ROOT / "manuscript",
        ROOT / "data",
    ]
    excluded_suffixes = {".aux", ".bbl", ".blg", ".log", ".out", ".zip"}
    for base in roots:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or "vendor" in path.parts
                or path.suffix in excluded_suffixes
                or path.name == "release_manifest.json"
            ):
                continue
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    for name in [
        ".gitignore",
        "CITATION.cff",
        "LICENSE",
        "LICENSE-CONTENT",
        "LITERATURE_SCREEN.md",
        "README.md",
        "analyze_target.py",
        "evaluate_pipeline.py",
        "requirements.txt",
        "run_pipeline.py",
    ]:
        path = ROOT / name
        if path.is_file():
            rows.append(
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    rows.sort(key=lambda row: row["path"])
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "files": rows,
    }
    path = RESULTS / "release_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-cache",
        type=Path,
        help="Optional directory containing previously downloaded TESSCut FITS files",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require both TESSCut files and the normalized QLP cache to exist",
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.source_cache:
        stage_cutouts(args.source_cache.resolve())

    python = sys.executable
    run_step("01_joint_raw_pixel_fit", [python, "src/joint_raw_pixel_fit.py"])
    run_step("02_robust_validation", [python, "src/robust_validate.py"])
    run_step("03_centroid_localization", [python, "src/localize_events.py"])

    packet_command = [python, "analyze_target.py", "--tic", "31065777"]
    if args.offline:
        packet_command.append("--offline")
    run_step("04_evidence_packet", packet_command)
    run_step("05_publication_figure", [python, "src/make_figure.py"])

    manuscript_dir = ROOT / "manuscript"
    shutil.copy2(
        RESULTS / "publication" / "TIC_31065777_RNAAS_FIGURE.pdf",
        manuscript_dir / "TIC_31065777_RNAAS_FIGURE.pdf",
    )
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        raise RuntimeError(
            "Tectonic is required to compile the manuscript; see README.md"
        )
    run_step(
        "06_manuscript",
        [tectonic, "--keep-intermediates", "--keep-logs", "tic_31065777_rnaas.tex"],
        cwd=manuscript_dir,
    )
    run_step(
        "07_tests",
        [python, "-m", "unittest", "discover", "-s", "tests", "-v"],
    )
    if not args.skip_evaluation:
        run_step("08_engineering_evaluation", [python, "evaluate_pipeline.py"])

    manifest = write_manifest()
    print(f"\nPublication pipeline complete. Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
