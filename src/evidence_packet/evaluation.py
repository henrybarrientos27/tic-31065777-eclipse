"""Deterministic evaluation suite for the evidence-packet signal core."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
from typing import Any

from astropy.timeseries import BoxLeastSquares
import numpy as np
import pandas as pd

from omega_v2.detection import detect_dip_events, link_repeating_events
from omega_v2.lightcurves import robust_sigma

from .analysis import harmonic_period_agreement, phase_distance


EVALUATION_SEED = 20260719


def _synthetic_frame(
    *,
    model: str,
    seed: int,
    depth: float = 0.0,
    period: float = 5.137,
    duration_hours: float = 3.0,
    corrupted: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    frames = []
    corruption_rows: list[dict] = []
    for sector, start in enumerate((0.0, 55.0, 110.0), start=1):
        time = start + np.arange(0.0, 24.0, 30.0 / 1440.0)
        flux = 1.0 + rng.normal(0.0, 0.0008, len(time))
        if model in {"transit", "eclipse", "injection"}:
            distance = np.abs(phase_distance(time, 1.35, period))
            inside = distance <= 0.5 * duration_hours / 24.0
            flux[inside] -= depth
            if model == "eclipse":
                secondary = np.abs(np.abs(phase_distance(time, 1.35, period)) - 0.5 * period)
                flux[secondary <= 0.35 * duration_hours / 24.0] -= 0.25 * depth
        elif model == "variable":
            flux += 0.008 * np.sin(2.0 * np.pi * time / period)
            flux += 0.002 * np.sin(4.0 * np.pi * time / period + 0.4)
        elif model != "normal":
            raise ValueError(f"Unknown synthetic model: {model}")

        if corrupted:
            missing = rng.choice(len(time), size=max(8, len(time) // 35), replace=False)
            flux_missing = missing[: len(missing) // 2]
            time_missing = missing[len(missing) // 2 :]
            for index in flux_missing:
                corruption_rows.append(
                    {
                        "sector": sector,
                        "source_row": int(index),
                        "corruption": "nonfinite_flux",
                        "original_time": float(time[index]),
                        "original_flux": float(flux[index]),
                        "corrupted_time": float(time[index]),
                        "corrupted_flux": np.nan,
                        "retained_after_finite_filter": False,
                    }
                )
            flux[flux_missing] = np.nan
            for index in time_missing:
                corruption_rows.append(
                    {
                        "sector": sector,
                        "source_row": int(index),
                        "corruption": "nonfinite_time",
                        "original_time": float(time[index]),
                        "original_flux": float(flux[index]),
                        "corrupted_time": np.nan,
                        "corrupted_flux": float(flux[index]),
                        "retained_after_finite_filter": False,
                    }
                )
            time[time_missing] = np.nan
            finite = np.flatnonzero(np.isfinite(time) & np.isfinite(flux))
            outliers = rng.choice(finite, size=12, replace=False)
            offsets = rng.choice([-1, 1], len(outliers)) * 0.04
            for index, offset in zip(outliers, offsets, strict=True):
                corruption_rows.append(
                    {
                        "sector": sector,
                        "source_row": int(index),
                        "corruption": "additive_flux_outlier",
                        "original_time": float(time[index]),
                        "original_flux": float(flux[index]),
                        "corrupted_time": float(time[index]),
                        "corrupted_flux": float(flux[index] + offset),
                        "retained_after_finite_filter": True,
                    }
                )
            flux[outliers] += offsets

        valid = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[valid], flux[valid]
        median = float(np.median(flux))
        normalized = flux / median
        noise = robust_sigma(normalized - 1.0)
        frames.append(
            pd.DataFrame(
                {
                    "tic_id": -seed,
                    "sector": sector,
                    "author": "SYNTHETIC_LOCAL_FIXTURE",
                    "segment_id": 0,
                    "time": time,
                    "flux": normalized,
                    "residual_flux": normalized - 1.0,
                    "noise": noise,
                    "cadence_days": 30.0 / 1440.0,
                    "edge_distance_days": np.minimum(time - time[0], time[-1] - time),
                }
            )
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(corruption_rows)


def _omega_primary(frame: pd.DataFrame, max_period: float = 20.0) -> dict:
    events = detect_dip_events(frame)
    repeat = (
        link_repeating_events(events, frame, min_period=0.75, max_period=max_period)
        if len(events) >= 2
        else None
    )
    detected = bool(
        repeat
        and repeat.get("positive_windows", 0) >= 3
        and repeat.get("matched_sector_count", 0) >= 2
        and repeat.get("negative_windows", 99) <= 1
    )
    return {
        "detected": detected,
        "period_days": float(repeat["period_days"]) if repeat else None,
        "event_count": int(len(events)),
        "positive_windows": int(repeat.get("positive_windows", 0)) if repeat else 0,
        "negative_windows": int(repeat.get("negative_windows", 0)) if repeat else 0,
    }


def _bls_replication(frame: pd.DataFrame, max_period: float = 20.0) -> dict:
    """Independent BoxLeastSquares search with no OMEGA event inputs."""
    finite = np.isfinite(frame["time"]) & np.isfinite(frame["flux"])
    time = frame.loc[finite, "time"].to_numpy(float)
    flux = frame.loc[finite, "flux"].to_numpy(float)
    scatter = robust_sigma(flux - np.median(flux))
    if not np.isfinite(scatter) or scatter <= 0:
        scatter = float(np.std(flux))
    uncertainties = np.full(len(flux), scatter)
    periods = np.linspace(0.75, max_period, 1_200)
    durations = np.asarray([1.0, 2.0, 3.0, 6.0, 10.0]) / 24.0
    model = BoxLeastSquares(time, flux, dy=uncertainties)
    result = model.power(periods, durations, objective="snr")
    index = int(np.nanargmax(result.power))
    period = float(result.period[index])
    transit_time = float(result.transit_time[index])
    duration = float(result.duration[index])
    depth = float(result.depth[index])
    depth_error = float(result.depth_err[index])
    snr = depth / depth_error if depth_error > 0 else float("nan")

    predicted = np.abs(phase_distance(time, transit_time, period)) <= 0.5 * duration
    cycles = np.rint((time[predicted] - transit_time) / period).astype(int)
    observed_transits = int(len(np.unique(cycles)))
    detected = bool(
        np.isfinite(snr)
        and snr >= 8.0
        and depth >= 0.0025
        and observed_transits >= 3
    )
    return {
        "detected": detected,
        "period_days": period,
        "depth_percent": 100.0 * depth,
        "depth_snr": float(snr),
        "observed_transits": observed_transits,
    }


def _metrics(results: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    truth = results["truth_positive"].astype(bool)
    prediction = results[prediction_column].astype(bool)
    tp = int(np.sum(truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    fp = int(np.sum(~truth & prediction))
    tn = int(np.sum(~truth & ~prediction))
    return {
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "true_positive_rate": float(tp / (tp + fn)) if tp + fn else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "cases": int(len(results)),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_cached(project_root: Path, tic_id: int) -> Path | None:
    paths = list(
        (project_root / "results" / "omega_v2").glob(
            f"*/tables/TIC_{tic_id}_lightcurve.parquet"
        )
    )
    return paths[0] if paths else None


def run_evaluation_suite(project_root: Path, output_dir: Path) -> Path:
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "suite_manifest.csv",
        "case_results.csv",
        "corruption_manifest.csv",
        "metrics.json",
        "recovery_by_signal_strength.csv",
        "implementation_agreement.csv",
        "reproducibility.json",
        "artifact_manifest.csv",
        "index.html",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()

    specifications = [
        ("normal_white_1", "known_normal", "normal", False, 0.0, 5.137, False),
        ("normal_white_2", "known_normal", "normal", False, 0.0, 3.411, False),
        ("sinusoidal_variable", "known_variable", "variable", False, 0.0, 4.21, False),
        ("sinusoidal_variable_fast", "known_variable", "variable", False, 0.0, 1.73, False),
        ("known_transit", "known_transit_or_eclipse", "transit", True, 0.008, 5.137, False),
        ("known_eclipse", "known_transit_or_eclipse", "eclipse", True, 0.035, 3.29, False),
        ("corrupted_normal", "deliberately_corrupted", "normal", False, 0.0, 5.137, True),
        ("corrupted_eclipse", "deliberately_corrupted", "eclipse", True, 0.025, 4.17, True),
    ]
    for depth in (0.0025, 0.005, 0.01, 0.02):
        for injection_trial in range(3):
            specifications.append(
                (
                    f"injection_{100 * depth:.2f}pct_trial_{injection_trial + 1}",
                    "synthetic_injection",
                    "injection",
                    True,
                    depth,
                    5.137,
                    False,
                )
            )

    manifest_rows = []
    result_rows = []
    corruption_tables = []
    for case_index, (
        case_id,
        category,
        model,
        truth_positive,
        depth,
        true_period,
        corrupted,
    ) in enumerate(specifications):
        seed = EVALUATION_SEED + case_index
        frame, corruption = _synthetic_frame(
            model=model,
            seed=seed,
            depth=depth,
            period=true_period,
            corrupted=corrupted,
        )
        if len(corruption):
            corruption.insert(0, "case_id", case_id)
            corruption_tables.append(corruption)
        primary = _omega_primary(frame)
        replication = _bls_replication(frame)
        manifest_rows.append(
            {
                "case_id": case_id,
                "category": category,
                "truth_positive": truth_positive,
                "truth_period_days": true_period if truth_positive else np.nan,
                "signal_depth_percent": 100.0 * depth,
                "corrupted": corrupted,
                "truth_basis": "deterministic analytic fixture; ground truth known by construction",
                "seed": seed,
                "points": len(frame),
            }
        )
        result_rows.append(
            {
                "case_id": case_id,
                "category": category,
                "truth_positive": truth_positive,
                "truth_period_days": true_period if truth_positive else np.nan,
                "signal_depth_percent": 100.0 * depth,
                "primary_detected": primary["detected"],
                "primary_period_days": primary["period_days"],
                "primary_event_count": primary["event_count"],
                "replication_detected": replication["detected"],
                "replication_period_days": replication["period_days"],
                "replication_depth_snr": replication["depth_snr"],
                "period_agreement": harmonic_period_agreement(
                    primary["period_days"], replication["period_days"]
                ),
            }
        )

    # Include the packet's real, locally cached repeat target as an empirical
    # regression case. It is additional to, not a substitute for, known-truth
    # analytic fixtures.
    empirical_path = _find_cached(project_root, 31065777)
    if empirical_path:
        empirical = pd.read_parquet(empirical_path)
        primary = _omega_primary(empirical, max_period=100.0)
        replication = _bls_replication(empirical, max_period=100.0)
        manifest_rows.append(
            {
                "case_id": "TIC_31065777_empirical_repeat",
                "category": "empirical_public_tess_regression",
                "truth_positive": True,
                "truth_period_days": 40.5717302229,
                "signal_depth_percent": 6.207177,
                "corrupted": False,
                "truth_basis": "existing raw-pixel five-event validation in this repository",
                "seed": np.nan,
                "points": len(empirical),
            }
        )
        result_rows.append(
            {
                "case_id": "TIC_31065777_empirical_repeat",
                "category": "empirical_public_tess_regression",
                "truth_positive": True,
                "truth_period_days": 40.5717302229,
                "signal_depth_percent": 6.207177,
                "primary_detected": primary["detected"],
                "primary_period_days": primary["period_days"],
                "primary_event_count": primary["event_count"],
                "replication_detected": replication["detected"],
                "replication_period_days": replication["period_days"],
                "replication_depth_snr": replication["depth_snr"],
                "period_agreement": harmonic_period_agreement(
                    primary["period_days"], replication["period_days"]
                ),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    results = pd.DataFrame(result_rows)
    corruption_manifest = (
        pd.concat(corruption_tables, ignore_index=True)
        if corruption_tables
        else pd.DataFrame(
            columns=[
                "case_id",
                "sector",
                "source_row",
                "corruption",
                "original_time",
                "original_flux",
                "corrupted_time",
                "corrupted_flux",
                "retained_after_finite_filter",
            ]
        )
    )
    manifest.to_csv(output_dir / "suite_manifest.csv", index=False)
    results.to_csv(output_dir / "case_results.csv", index=False)
    corruption_manifest.to_csv(output_dir / "corruption_manifest.csv", index=False)

    headline_scope = results[
        results["category"] != "empirical_public_tess_regression"
    ].copy()
    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": EVALUATION_SEED,
        "scope": (
            "Small deterministic engineering evaluation. Rates are not estimates "
            "of performance on the astronomical population."
        ),
        "primary_omega": _metrics(headline_scope, "primary_detected"),
        "replication_bls": _metrics(headline_scope, "replication_detected"),
        "implementation_agreement_rate": float(
            np.mean(
                headline_scope["primary_detected"].astype(bool)
                == headline_scope["replication_detected"].astype(bool)
            )
        ),
        "period_agreement_rate_when_both_detect": None,
        "empirical_public_tess_regression": None,
        "by_category": {},
    }
    for category, category_rows in headline_scope.groupby("category"):
        metrics["by_category"][str(category)] = {
            "primary_omega": _metrics(category_rows, "primary_detected"),
            "replication_bls": _metrics(category_rows, "replication_detected"),
        }
    both = headline_scope[
        headline_scope["primary_detected"].astype(bool)
        & headline_scope["replication_detected"].astype(bool)
    ]
    if len(both):
        metrics["period_agreement_rate_when_both_detect"] = float(
            both["period_agreement"].mean()
        )
    empirical_rows = results[
        results["category"] == "empirical_public_tess_regression"
    ]
    if len(empirical_rows):
        empirical_row = empirical_rows.iloc[0]
        metrics["empirical_public_tess_regression"] = {
            "case_id": str(empirical_row["case_id"]),
            "primary_detected": bool(empirical_row["primary_detected"]),
            "replication_detected": bool(empirical_row["replication_detected"]),
            "period_agreement": bool(empirical_row["period_agreement"]),
            "interpretation": (
                "This empirical disagreement is retained as a visible weakness "
                "and is excluded from known-truth headline rates."
            ),
        }
    _write_json(output_dir / "metrics.json", metrics)

    injections = results[results["category"] == "synthetic_injection"].copy()
    injections["period_agreement_when_both_detect"] = injections[
        "period_agreement"
    ].where(
        injections["primary_detected"].astype(bool)
        & injections["replication_detected"].astype(bool)
    )
    recovery = (
        injections.groupby("signal_depth_percent", as_index=False)
        .agg(
            tests=("case_id", "size"),
            primary_recovery_rate=("primary_detected", "mean"),
            replication_recovery_rate=("replication_detected", "mean"),
            implementation_period_agreement_rate=(
                "period_agreement_when_both_detect",
                "mean",
            ),
        )
        .sort_values("signal_depth_percent")
    )
    recovery.to_csv(output_dir / "recovery_by_signal_strength.csv", index=False)

    agreement = results[
        [
            "case_id",
            "primary_detected",
            "replication_detected",
            "primary_period_days",
            "replication_period_days",
            "period_agreement",
        ]
    ].copy()
    agreement["classification_agreement"] = (
        agreement["primary_detected"] == agreement["replication_detected"]
    )
    agreement.to_csv(output_dir / "implementation_agreement.csv", index=False)

    primary_metrics = metrics["primary_omega"]
    replication_metrics = metrics["replication_bls"]
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Scientific core evaluation</title><style>
body{{font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#152238}}
h1,h2{{color:#173e72}} .warning{{border-left:5px solid #b26700;padding:12px;background:#fff7e8}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.metric{{border:1px solid #d8dfe8;border-radius:8px;padding:14px}} .metric strong{{display:block;font-size:26px;color:#2457a7}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d8dfe8;padding:7px;text-align:left}}th{{background:#edf2f8}}
.scroll{{overflow-x:auto}}</style></head><body>
<h1>Scientific core evaluation</h1>
<p class="warning">{html.escape(metrics['scope'])}</p>
<h2>Headline rates</h2><div class="grid">
<div class="metric"><strong>{100*primary_metrics['true_positive_rate']:.1f}%</strong>OMEGA true-positive rate</div>
<div class="metric"><strong>{100*primary_metrics['false_positive_rate']:.1f}%</strong>OMEGA false-positive rate</div>
<div class="metric"><strong>{100*replication_metrics['true_positive_rate']:.1f}%</strong>BLS true-positive rate</div>
<div class="metric"><strong>{100*replication_metrics['false_positive_rate']:.1f}%</strong>BLS false-positive rate</div>
<div class="metric"><strong>{100*metrics['implementation_agreement_rate']:.1f}%</strong>classification agreement</div>
</div>
<h2>Recovery by signal strength</h2><div class="scroll">{recovery.to_html(index=False, border=0)}</div>
<h2>All cases</h2><div class="scroll">{results.to_html(index=False, border=0, na_rep='—')}</div>
<h2>Deliberately corrupted observations</h2>
<p>Every injected nonfinite value and retained outlier is recorded in
<code>corruption_manifest.csv</code> with its original and corrupted values.</p>
<p>Known normal, variable, transit, and eclipse cases are deterministic analytic
fixtures with truth known by construction. The empirical TIC regression case uses
locally cached public TESS data and is excluded from headline rates. The
headline false-positive denominator contains {primary_metrics['false_positives'] + primary_metrics['true_negatives']}
negative fixtures; these rates are engineering checks, not population estimates.</p>
</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")

    source_files = [
        Path(__file__),
        Path(__file__).with_name("analysis.py"),
        project_root / "src" / "omega_v2" / "detection.py",
        project_root / "src" / "omega_v2" / "lightcurves.py",
        project_root / "evaluate_pipeline.py",
    ]
    reproducibility = {
        "seed": EVALUATION_SEED,
        "generated_utc": metrics["generated_utc"],
        "python_truth_scope": (
            "Deterministic analytic fixtures plus an optional locally cached "
            "public-TESS regression case"
        ),
        "code_files": {
            str(path.relative_to(project_root)): {"sha256": _sha256(path)}
            for path in source_files
            if path.exists()
        },
    }
    _write_json(output_dir / "reproducibility.json", reproducibility)
    artifact_rows = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "artifact_manifest.csv":
            continue
        artifact_rows.append(
            {
                "relative_path": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    pd.DataFrame(artifact_rows).to_csv(
        output_dir / "artifact_manifest.csv", index=False
    )
    return output_dir
