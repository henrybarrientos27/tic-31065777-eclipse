"""Resumable OMEGA v2 target-analysis orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import threading
import time
from typing import Iterable

import numpy as np
import pandas as pd

from .catalogs import villanova_catalog_status
from .detection import detect_dip_events, link_repeating_events, summarize_target
from .lightcurves import download_target_lightcurves
from .plotting import plot_candidate


@contextmanager
def _target_timeout(seconds: float | None):
    """Interrupt one stalled archive target without losing the campaign.

    SIGALRM is available on macOS/Linux and works here because campaign scans
    run on the main thread. On other platforms/threads the guard safely becomes
    a no-op rather than breaking target analysis.
    """
    if (
        seconds is None
        or seconds <= 0
        or not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    timeout_started = time.monotonic()

    def handle_timeout(signum, frame):
        raise TimeoutError(
            f"Target exceeded the {float(seconds):.0f}-second archive timeout"
        )

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        previous_delay, previous_interval = previous_timer
        if previous_delay > 0:
            remaining = max(
                1e-6, previous_delay - (time.monotonic() - timeout_started)
            )
            signal.setitimer(
                signal.ITIMER_REAL, remaining, previous_interval
            )


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _portable_path(path: Path, project_root: Path) -> str:
    """Store project files relative to the repository when possible."""
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


@dataclass
class OmegaPaths:
    project_root: Path
    run_name: str

    def __post_init__(self):
        self.data_root = self.project_root / "data" / "omega_v2"
        self.result_root = self.project_root / "results" / "omega_v2" / self.run_name
        self.plot_root = self.project_root / "plots" / "omega_v2" / self.run_name
        self.lightcurve_cache = self.data_root / "mast_cache"
        self.tables = self.result_root / "tables"
        self.events = self.result_root / "events"
        self.products = self.result_root / "products"
        self.repeats = self.result_root / "repeats"
        self.checkpoints = self.result_root / "checkpoints"
        for path in [
            self.data_root,
            self.result_root,
            self.plot_root,
            self.lightcurve_cache,
            self.tables,
            self.events,
            self.products,
            self.repeats,
            self.checkpoints,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def analyze_target(
    tic_id: int,
    paths: OmegaPaths,
    max_sectors: int = 8,
    resume: bool = True,
    metadata: dict | None = None,
    force_plot: bool = False,
) -> dict:
    tic_id = int(tic_id)
    checkpoint_path = paths.checkpoints / f"TIC_{tic_id}.json"
    if resume and checkpoint_path.exists():
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))

    metadata = dict(metadata or {})
    started = time.time()

    try:
        lightcurve_path = paths.tables / f"TIC_{tic_id}_lightcurve.parquet"
        product_path = paths.products / f"TIC_{tic_id}_products.csv"

        if resume and lightcurve_path.exists():
            lightcurve = pd.read_parquet(lightcurve_path)
            product_rows = (
                pd.read_csv(product_path).to_dict(orient="records")
                if product_path.exists()
                else []
            )
        else:
            lightcurve, product_rows = download_target_lightcurves(
                tic_id,
                download_dir=paths.lightcurve_cache,
                max_sectors=max_sectors,
            )
            pd.DataFrame(product_rows).to_csv(product_path, index=False)
            if len(lightcurve):
                lightcurve.to_parquet(lightcurve_path, index=False)

        if len(lightcurve) == 0:
            result = {
                "tic_id": tic_id,
                "status": "no_tess_lightcurve",
                "classification": "not_analyzed",
                "omega_score": -999.0,
                "points": 0,
                "sectors": 0,
                "elapsed_seconds": time.time() - started,
                **metadata,
            }
        else:
            events = detect_dip_events(lightcurve)
            events_path = paths.events / f"TIC_{tic_id}_events.csv"
            events.to_csv(events_path, index=False)

            repeat = link_repeating_events(events, lightcurve) if len(events) >= 2 else None
            repeat_path = paths.repeats / f"TIC_{tic_id}_repeat.json"
            repeat_path.write_text(
                json.dumps(repeat, indent=2, default=_json_default),
                encoding="utf-8",
            )

            catalog_status = {"villanova_known": False, "villanova_url": ""}
            if len(events):
                try:
                    catalog_status = villanova_catalog_status(tic_id)
                except Exception as error:
                    catalog_status = {
                        "villanova_known": False,
                        "villanova_url": "",
                        "villanova_error": f"{type(error).__name__}: {error}",
                    }

            result = summarize_target(
                tic_id,
                lightcurve=lightcurve,
                events=events,
                repeat=repeat,
                villanova_known=bool(catalog_status["villanova_known"]),
            )
            result.update(catalog_status)
            result.update(metadata)
            result["elapsed_seconds"] = time.time() - started
            result["event_file"] = _portable_path(events_path, paths.project_root)
            result["lightcurve_file"] = _portable_path(lightcurve_path, paths.project_root)
            result["repeat_file"] = _portable_path(repeat_path, paths.project_root)

            if force_plot or result["classification"] != "no_significant_events":
                plot_path = paths.plot_root / f"TIC_{tic_id}_omega_v2.png"
                plot_candidate(
                    tic_id,
                    lightcurve=lightcurve,
                    events=events,
                    summary=result,
                    repeat=repeat,
                    output_path=plot_path,
                )
                result["plot_file"] = _portable_path(plot_path, paths.project_root)

    except Exception as error:
        result = {
            "tic_id": tic_id,
            "status": "failed",
            "classification": "failed",
            "omega_score": -999.0,
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.time() - started,
            **metadata,
        }

    checkpoint_path.write_text(
        json.dumps(result, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return result


def aggregate_results(paths: OmegaPaths) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(paths.checkpoints.glob("TIC_*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "omega_score" in frame.columns:
        frame = frame.sort_values("omega_score", ascending=False)
    frame.to_csv(paths.result_root / "omega_v2_leaderboard.csv", index=False)
    return frame.reset_index(drop=True)


def scan_targets(
    targets: pd.DataFrame,
    paths: OmegaPaths,
    max_sectors: int = 8,
    resume: bool = True,
    delay_seconds: float = 0.2,
    target_timeout_seconds: float = 180.0,
) -> pd.DataFrame:
    total = len(targets)
    for position, row in targets.reset_index(drop=True).iterrows():
        tic_id = int(row["tic_id"])
        print(f"\n[{position + 1}/{total}] TIC {tic_id}")
        with _target_timeout(target_timeout_seconds):
            result = analyze_target(
                tic_id,
                paths=paths,
                max_sectors=max_sectors,
                resume=resume,
                metadata=row.to_dict(),
            )
        print(
            f"  {result.get('status')} | {result.get('classification')} | "
            f"score={result.get('omega_score')} | "
            f"sectors={result.get('sectors', 0)} | "
            f"period={result.get('period_days', np.nan)}"
        )
        time.sleep(max(0.0, delay_seconds))
    return aggregate_results(paths)


def display_leaderboard(frame: pd.DataFrame, rows: int = 20) -> str:
    if len(frame) == 0:
        return "No results."
    columns = [
        "tic_id",
        "classification",
        "morphology_hint",
        "omega_score",
        "detection_confidence_score",
        "sectors",
        "event_count",
        "primary_depth_percent",
        "secondary_to_primary_ratio",
        "unlinked_event_count",
        "period_days",
        "repeat_positive_windows",
        "period_negative_windows",
        "villanova_known",
    ]
    columns = [column for column in columns if column in frame.columns]
    return frame[columns].head(rows).to_string(index=False)
