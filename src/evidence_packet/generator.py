"""Generate a completely local Scientific Evidence Packet."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import sys
from typing import Any

from astropy.io import fits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omega_v2.detection import (
    detect_dip_events,
    link_repeating_events,
    summarize_target,
)
from omega_v2.lightcurves import _split_indices
from omega_v2.pipeline import OmegaPaths, analyze_target as run_omega_target

from .analysis import (
    bootstrap_ephemeris,
    exact_cycle_permutation_test,
    harmonic_period_agreement,
    independent_lightcurve_replication,
    injection_summary,
    leave_one_event_out,
    phase_distance,
    robust_replication_ephemeris,
    targeted_injection_tests,
    weighted_ephemeris,
)


SCHEMA_VERSION = "1.1.0"
DEFAULT_BOOTSTRAP_TRIALS = 2_000
DEFAULT_SEED = 20260719


@dataclass
class SourceProduct:
    sector: int
    path: str
    sha256: str
    bytes: int
    origin: str
    telescope: str
    camera: int | None
    ccd: int | None
    object_name: str
    ra_deg: float | None
    dec_deg: float | None
    flux_column: str
    raw_points: int


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_clean_json(item) for item in value.tolist()]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_clean_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path | str, project_root: Path) -> str:
    """Represent local inputs without publishing a user's absolute home path."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        return f"<external-local-file>/{resolved.name}"


def _portable_frame_paths(frame: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    portable = frame.copy()
    prefix = str(project_root.resolve()) + os.sep
    for column in portable.columns:
        if portable[column].dtype != object:
            continue
        portable[column] = portable[column].map(
            lambda value: (
                value.replace(prefix, "")
                if isinstance(value, str)
                else value
            )
        )
    return portable


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in [
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "astropy",
        "lightkurve",
        "pyarrow",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _best_cached_lightcurve(project_root: Path, tic_id: int) -> Path | None:
    candidates = list(
        (project_root / "results" / "omega_v2").glob(
            f"*/tables/TIC_{tic_id}_lightcurve.parquet"
        )
    )
    readable: list[tuple[int, Path]] = []
    for path in candidates:
        try:
            metadata = pd.read_parquet(path, columns=["time"])
            readable.append((len(metadata), path))
        except Exception:
            continue
    return max(readable, default=(0, None), key=lambda item: item[0])[1]


def _load_primary_inputs(
    project_root: Path,
    tic_id: int,
    *,
    max_sectors: int,
    offline: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict | None, dict]:
    cached = _best_cached_lightcurve(project_root, tic_id)
    acquisition = "reused_local_parquet_cache"
    if cached is None:
        if offline:
            raise FileNotFoundError(
                f"No cached normalized light curve exists for TIC {tic_id}; "
                "rerun without --offline once to download public MAST data."
            )
        work_paths = OmegaPaths(project_root, f"evidence_packet_work_{tic_id}")
        result = run_omega_target(
            tic_id,
            paths=work_paths,
            max_sectors=max_sectors,
            resume=True,
            force_plot=False,
        )
        if result.get("status") != "ok":
            raise RuntimeError(
                f"OMEGA acquisition failed: {result.get('error', result.get('status'))}"
            )
        cached = Path(result["lightcurve_file"])
        if not cached.is_absolute():
            cached = project_root / cached
        acquisition = "downloaded_public_mast_then_cached"

    lightcurve = pd.read_parquet(cached).sort_values("time").reset_index(drop=True)
    if max_sectors and lightcurve["sector"].nunique() > max_sectors:
        sectors = sorted(int(value) for value in lightcurve["sector"].unique())
        indices = sorted(
            set(
                int(round(position))
                for position in np.linspace(0, len(sectors) - 1, max_sectors)
            )
        )
        selected = {sectors[index] for index in indices}
        lightcurve = lightcurve[lightcurve["sector"].isin(selected)].copy()

    events = detect_dip_events(lightcurve)
    repeat = link_repeating_events(events, lightcurve) if len(events) >= 2 else None
    summary = summarize_target(
        tic_id,
        lightcurve=lightcurve,
        events=events,
        repeat=repeat,
        villanova_known=False,
    )
    summary.update(
        {
            "status": "ok",
            "source_lightcurve": _portable_path(cached, project_root),
            "acquisition": acquisition,
        }
    )
    return lightcurve, events, repeat, summary


def _find_source_fits(project_root: Path, tic_id: int, sector: int) -> Path | None:
    padded = f"{tic_id:016d}"
    pattern = f"*s{sector:04d}-{padded}*fits"
    roots = [
        project_root / "data" / "omega_v2" / "mast_cache",
        project_root / "data" / "mastDownload",
    ]
    matches: list[Path] = []
    for root in roots:
        if root.exists():
            matches.extend(root.rglob(pattern))
    qlp = [path for path in matches if "hlsp_qlp" in path.name.lower()]
    return sorted(qlp or matches)[0] if (qlp or matches) else None


def _preferred_flux_column(names: list[str]) -> str:
    for name in [
        "KSPSAP_FLUX",
        "DET_FLUX",
        "PDCSAP_FLUX",
        "SAP_FLUX",
        "FLUX",
    ]:
        if name in names:
            return name
    raise KeyError("No recognized flux column in source FITS")


def _audit_sources(
    project_root: Path,
    tic_id: int,
    lightcurve: pd.DataFrame,
) -> tuple[list[SourceProduct], pd.DataFrame, pd.DataFrame]:
    products: list[SourceProduct] = []
    disposition_rows: list[dict] = []
    qc_rows: list[dict] = []

    for sector, retained_sector in lightcurve.groupby("sector", sort=True):
        sector = int(sector)
        source_path = _find_source_fits(project_root, tic_id, sector)
        retained_keys = set(np.round(retained_sector["time"].to_numpy(float), 8))
        if source_path is None:
            qc_rows.append(
                {
                    "sector": sector,
                    "raw_points": None,
                    "retained_points": int(len(retained_sector)),
                    "removed_points": None,
                    "source_audit_complete": False,
                    "note": "Original FITS product was not found in the local cache.",
                }
            )
            continue

        with fits.open(source_path, memmap=True) as hdul:
            header = hdul[0].header
            table = hdul[1].data
            names = list(table.columns.names)
            flux_column = _preferred_flux_column(names)
            times = np.asarray(table["TIME"], dtype=float)
            flux = np.asarray(table[flux_column], dtype=float)
            quality = (
                np.nan_to_num(np.asarray(table["QUALITY"], dtype=float), nan=1).astype(
                    np.int64
                )
                if "QUALITY" in names
                else np.zeros(len(times), dtype=np.int64)
            )
            cadence = (
                np.asarray(table["CADENCENO"])
                if "CADENCENO" in names
                else np.arange(len(times))
            )

            products.append(
                SourceProduct(
                    sector=sector,
                    path=_portable_path(source_path, project_root),
                    sha256=_sha256(source_path),
                    bytes=int(source_path.stat().st_size),
                    origin=str(header.get("ORIGIN", "")),
                    telescope=str(header.get("TELESCOP", "TESS")),
                    camera=int(header["CAMERA"]) if header.get("CAMERA") is not None else None,
                    ccd=int(header["CCD"]) if header.get("CCD") is not None else None,
                    object_name=str(header.get("OBJECT", f"TIC {tic_id}")),
                    ra_deg=float(header["RA_OBJ"]) if header.get("RA_OBJ") is not None else None,
                    dec_deg=float(header["DEC_OBJ"]) if header.get("DEC_OBJ") is not None else None,
                    flux_column=flux_column,
                    raw_points=int(len(times)),
                )
            )

        eligible = np.isfinite(times) & np.isfinite(flux) & (quality == 0)
        eligible_rows = np.flatnonzero(eligible)
        eligible_rows = eligible_rows[np.argsort(times[eligible_rows])]
        preprocessing_reason: dict[int, tuple[str, str, int | None]] = {}
        for segment_id, segment_positions in enumerate(
            _split_indices(times[eligible_rows])
        ):
            raw_rows = eligible_rows[segment_positions]
            if len(raw_rows) < 30:
                for raw_row in raw_rows:
                    preprocessing_reason[int(raw_row)] = (
                        "short_continuous_segment_lt_30_points",
                        "CLEAN-005",
                        int(segment_id),
                    )
                continue
            median_flux = float(np.median(flux[raw_rows]))
            if not np.isfinite(median_flux) or median_flux == 0:
                for raw_row in raw_rows:
                    preprocessing_reason[int(raw_row)] = (
                        "invalid_segment_median",
                        "CLEAN-004",
                        int(segment_id),
                    )
                continue
            for raw_row in raw_rows:
                preprocessing_reason[int(raw_row)] = (
                    "eligible_after_source_cleaning",
                    "CLEAN-004",
                    int(segment_id),
                )

        retained_count = 0
        reason_counts: dict[str, int] = {}
        for row_number, (time_value, flux_value, flag, cadence_number) in enumerate(
            zip(times, flux, quality, cadence, strict=True)
        ):
            key = round(float(time_value), 8) if np.isfinite(time_value) else None
            retained = key in retained_keys
            preprocessing = preprocessing_reason.get(int(row_number))
            segment_id = preprocessing[2] if preprocessing else None
            if retained:
                reason = "retained"
                decision_rule_id = "CLEAN-004"
                retained_count += 1
            elif not np.isfinite(time_value):
                reason = "nonfinite_time"
                decision_rule_id = "CLEAN-002"
            elif not np.isfinite(flux_value):
                reason = f"nonfinite_{flux_column.lower()}"
                decision_rule_id = "CLEAN-002"
            elif int(flag) != 0:
                reason = f"quality_flag_{int(flag)}"
                decision_rule_id = "CLEAN-002"
            elif preprocessing and preprocessing[0] != "eligible_after_source_cleaning":
                reason = preprocessing[0]
                decision_rule_id = preprocessing[1]
            else:
                reason = "eligible_source_row_missing_from_normalized_cache"
                decision_rule_id = "AUDIT-001"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            disposition_rows.append(
                {
                    "sector": sector,
                    "source_row": int(row_number),
                    "cadence_number": int(cadence_number),
                    "time_btjd": float(time_value) if np.isfinite(time_value) else np.nan,
                    "source_flux": (
                        float(flux_value) if np.isfinite(flux_value) else np.nan
                    ),
                    "source_flux_column": flux_column,
                    "quality_flag": int(flag),
                    "quality_flag_binary": f"0b{format(int(flag), '032b')}",
                    "continuous_segment_id": segment_id,
                    "disposition": "retained" if retained else "removed",
                    "reason": reason,
                    "decision_rule_id": decision_rule_id,
                    "source_file": _portable_path(source_path, project_root),
                }
            )
        qc_rows.append(
            {
                "sector": sector,
                "raw_points": int(len(times)),
                "retained_points": int(retained_count),
                "removed_points": int(len(times) - retained_count),
                "retention_fraction": float(retained_count / max(1, len(times))),
                "nonzero_quality_points": int(np.sum(quality != 0)),
                "nonfinite_time_points": int(np.sum(~np.isfinite(times))),
                "nonfinite_flux_points": int(np.sum(~np.isfinite(flux))),
                "cache_mismatch_points": int(
                    reason_counts.get(
                        "eligible_source_row_missing_from_normalized_cache", 0
                    )
                ),
                "source_audit_complete": True,
                "note": json.dumps(reason_counts, sort_keys=True),
            }
        )

    return products, pd.DataFrame(disposition_rows), pd.DataFrame(qc_rows)


def _measurement_events(
    project_root: Path,
    tic_id: int,
    events: pd.DataFrame,
    repeat: dict | None,
) -> tuple[pd.DataFrame, str]:
    legacy_path = project_root / "results" / f"TIC_{tic_id}_robust_event_fits.csv"
    if legacy_path.exists():
        raw = pd.read_csv(legacy_path)
        if "radius_pixels" in raw:
            radii = raw["radius_pixels"].astype(float)
            selected_radius = float(radii.iloc[(radii - 1.5).abs().argmin()])
            raw = raw[np.isclose(radii, selected_radius)].copy()
        required = {"sector", "cycle", "center", "center_error_days"}
        if required.issubset(raw.columns) and len(raw) >= 3:
            return (
                _portable_frame_paths(
                    raw.sort_values("cycle").reset_index(drop=True),
                    project_root,
                ),
                _portable_path(legacy_path, project_root),
            )

    if repeat and repeat.get("matched_events"):
        raw = pd.DataFrame(repeat["matched_events"]).copy()
        raw = raw.rename(columns={"time": "center"})
        cadence_by_sector = (
            events.groupby("sector")["duration_hours"].median().to_dict()
            if len(events)
            else {}
        )
        raw["center_error_days"] = [
            max(1.0 / 1440.0, float(cadence_by_sector.get(sector, 1.0)) / 48.0)
            for sector in raw["sector"]
        ]
        return (
            _portable_frame_paths(
                raw.sort_values("cycle").reset_index(drop=True),
                project_root,
            ),
            "OMEGA matched events",
        )
    raise RuntimeError("At least three repeat-event measurements are required")


def _read_joint_duration(project_root: Path, tic_id: int, fallback_hours: float) -> float:
    candidates = [
        (
            project_root / "results" / f"TIC_{tic_id}_joint_raw_pixel_fit_summary.txt",
            ("Duration hours:",),
        ),
        (
            project_root / "results" / f"TIC_{tic_id}_joint_raw_pixel_fit_terminal.txt",
            ("Total eclipse duration:",),
        ),
    ]
    for path, prefixes in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(prefixes):
                try:
                    return float(line.split(":", 1)[1].split()[0])
                except (ValueError, IndexError):
                    pass
    return float(fallback_hours)


def _period_search_table(
    project_root: Path,
    tic_id: int,
    repeat: dict | None,
    fit: dict,
    fitted_events: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "rank": 1,
            "method": "OMEGA event-link search",
            "period_days": float(repeat["period_days"]) if repeat else np.nan,
            "score": float(repeat.get("period_score", np.nan)) if repeat else np.nan,
            "status": "selected_primary" if repeat else "no_repeat_solution",
            "origin_artifact": "data/primary_repeat.json",
            "selection_basis": (
                "Best OMEGA event-link solution after penalizing predicted "
                "eclipses in covered flat windows."
            ),
        },
        {
            "rank": 1,
            "method": "weighted raw-event ephemeris",
            "period_days": float(fit["period_days"]),
            "score": np.nan,
            "status": "selected_conservative_measurement",
            "origin_artifact": "data/signal_measurements.json",
            "selection_basis": (
                "Weighted fit to all packet-local fitted event centers and "
                "explicit integer cycle assignments."
            ),
        },
    ]
    ordered_events = fitted_events.sort_values("cycle").reset_index(drop=True)
    for rank in range(1, len(ordered_events)):
        earlier = ordered_events.iloc[rank - 1]
        later = ordered_events.iloc[rank]
        cycle_delta = int(later["cycle"] - earlier["cycle"])
        if cycle_delta <= 0:
            continue
        time_delta = float(later["center"] - earlier["center"])
        rows.append(
            {
                "rank": rank,
                "method": "adjacent fitted-event cycle spacing",
                "period_days": time_delta / cycle_delta,
                "score": np.nan,
                "status": "direct_cycle_spacing_check",
                "origin_artifact": "data/fitted_events.csv",
                "selection_basis": (
                    f"Sector {int(earlier['sector'])} to "
                    f"{int(later['sector'])}: {time_delta:.9f} d across "
                    f"{cycle_delta} assigned cycle(s)."
                ),
            }
        )
    legacy = project_root / "results" / f"TIC_{tic_id}_top_periods.csv"
    if legacy.exists():
        table = pd.read_csv(legacy).head(25)
        for rank, row in enumerate(table.to_dict(orient="records"), start=1):
            rows.append(
                {
                    "rank": rank,
                    "method": "legacy Box Least Squares search",
                    "period_days": row.get("period_days"),
                    "score": row.get("power"),
                    "status": "alternative_alias",
                    "origin_artifact": f"data/{legacy.name}",
                    "selection_basis": (
                        "Legacy broad-search maximum retained as search history; "
                        "not adopted without five-event cycle and coverage support."
                    ),
                }
            )
    aliases = project_root / "results" / f"TIC_{tic_id}_alias_elimination.csv"
    if aliases.exists():
        table = pd.read_csv(aliases).head(30)
        for rank, row in enumerate(table.to_dict(orient="records"), start=1):
            rows.append(
                {
                    "rank": rank,
                    "method": "legacy coverage-based alias test",
                    "period_days": row.get("period_days"),
                    "score": row.get("score"),
                    "status": (
                        "legacy_partial_support_not_adopted"
                        if bool(row.get("supported", False))
                        else "legacy_coverage_rejected"
                    ),
                    "origin_artifact": f"data/{aliases.name}",
                    "selection_basis": (
                        f"Legacy coverage test aligned "
                        f"{int(row.get('known_events_aligned', 0))} known events; "
                        "the conservative period requires consistency with all "
                        "five fitted event centers, including consecutive cycles."
                    ),
                }
            )
    return pd.DataFrame(rows)


def _instrument_comparisons(
    products: list[SourceProduct],
    qc: pd.DataFrame,
    measurements: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for product in products:
        event_rows = measurements[measurements["sector"].astype(int) == product.sector]
        qc_row = qc[qc["sector"].astype(int) == product.sector]
        rows.append(
            {
                "sector": product.sector,
                "camera": product.camera,
                "ccd": product.ccd,
                "filter_or_bandpass": "TESS broad optical bandpass",
                "pipeline_author": product.origin or "QLP",
                "retained_points": (
                    int(qc_row.iloc[0]["retained_points"]) if len(qc_row) else None
                ),
                "cadence_minutes": (
                    float(event_rows["cadence_minutes"].median())
                    if len(event_rows) and "cadence_minutes" in event_rows
                    else np.nan
                ),
                "event_measured": bool(len(event_rows)),
                "depth_percent": (
                    float(event_rows["depth_percent"].median())
                    if len(event_rows) and "depth_percent" in event_rows
                    else np.nan
                ),
                "depth_error_percent": (
                    float(event_rows["depth_error_percent"].median())
                    if len(event_rows) and "depth_error_percent" in event_rows
                    else np.nan
                ),
                "noise_percent": (
                    float(event_rows["noise_percent"].median())
                    if len(event_rows) and "noise_percent" in event_rows
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _copy_optional_controls(project_root: Path, tic_id: int, data_dir: Path) -> dict:
    names = {
        "blank_apertures": f"TIC_{tic_id}_blank_apertures.csv",
        "temporal_controls": f"TIC_{tic_id}_temporal_controls.csv",
        "aperture_robustness": f"TIC_{tic_id}_aperture_robustness.csv",
        "secondary_scan": f"TIC_{tic_id}_all_phase_secondary_scan.csv",
        "catalog_snapshot": f"TIC_{tic_id}_catalog_check.json",
        "legacy_bls_periods": f"TIC_{tic_id}_top_periods.csv",
        "legacy_alias_elimination": f"TIC_{tic_id}_alias_elimination.csv",
    }
    copied = {}
    for key, name in names.items():
        source = project_root / "results" / name
        if source.exists():
            destination = data_dir / name
            shutil.copy2(source, destination)
            copied[key] = str(destination.relative_to(data_dir.parent))
    return copied


def _conventional_explanations(
    measurements: dict,
    controls: dict,
) -> list[dict]:
    depth = measurements.get("median_depth_percent")
    return [
        {
            "explanation": "ordinary eclipsing stellar or substellar companion",
            "assessment": "favored",
            "reason": (
                f"The repeatable, approximately {depth:.2f}% eclipse-like depth is "
                "consistent with an occulting companion; photometry alone does not "
                "establish its mass."
            ),
            "evidence": ["data/signal_measurements.json#/median_depth_percent"],
        },
        {
            "explanation": "background or neighboring eclipsing binary",
            "assessment": "reduced but not eliminated",
            "reason": (
                "Raw-pixel aperture and blank-position controls were available."
                if "blank_apertures" in controls
                else "No packet-local blank-aperture control was available."
            ),
            "evidence": [controls.get("blank_apertures", "data/unresolved_weaknesses.json")],
        },
        {
            "explanation": "quality-flagged spacecraft or reduction artifact",
            "assessment": "reduced",
            "reason": (
                "The cadence ledger records quality flags and the measured events "
                "are drawn from retained cadences across multiple sectors."
            ),
            "evidence": ["data/cadence_disposition.csv", "data/quality_control.csv"],
        },
        {
            "explanation": "period alias or harmonic",
            "assessment": "reduced but search-history dependent",
            "reason": (
                "Consecutive-cycle events and coverage tests support the reported "
                "period, while the period-search table preserves competing aliases."
            ),
            "evidence": ["data/period_search.csv", "data/held_out_validation.csv"],
        },
        {
            "explanation": "stellar rotation or pulsation",
            "assessment": "disfavored, not physically ruled out",
            "reason": (
                "The measured signal is localized in phase and repeats as discrete "
                "dips rather than being classified from a sinusoidal model."
            ),
            "evidence": ["data/primary_events.csv", "plots/phase_fold.png"],
        },
        {
            "explanation": "planetary transit",
            "assessment": "not established",
            "reason": (
                "Transit-like morphology alone does not identify a planet; dilution, "
                "stellar radius uncertainty, and companion mass remain unresolved."
            ),
            "evidence": [
                "data/signal_measurements.json",
                controls.get("catalog_snapshot", "data/unresolved_weaknesses.json"),
            ],
        },
    ]


def _weaknesses(
    products: list[SourceProduct],
    measurement_count: int,
    controls: dict,
) -> list[dict]:
    cameras = {product.camera for product in products}
    weaknesses = [
        {
            "severity": "high",
            "weakness": "No radial-velocity or spectroscopic mass measurement is present.",
            "consequence": "Photometry cannot determine whether the companion is planetary, substellar, or stellar.",
        },
        {
            "severity": "high",
            "weakness": "The permutation test is post-selection.",
            "consequence": "Its p-value is a coherence diagnostic, not a global false-alarm probability.",
        },
        {
            "severity": "medium",
            "weakness": f"Only {measurement_count} fitted eclipse events constrain the conservative ephemeris.",
            "consequence": "Rare systematics or one influential event can be underrepresented despite leave-one-out checks.",
        },
        {
            "severity": "medium",
            "weakness": "Primary and replication fits reuse the same astronomical observations.",
            "consequence": "Code-path agreement is not independent observational confirmation.",
        },
        {
            "severity": "medium",
            "weakness": (
                "Only one TESS camera is represented."
                if len(cameras) <= 1
                else "Camera coverage is unbalanced across sectors."
            ),
            "consequence": "A true cross-camera consistency test is not available.",
        },
        {
            "severity": "medium",
            "weakness": "Targeted injections use a known injected ephemeris.",
            "consequence": "They measure local sensitivity, not blind-search completeness.",
        },
        {
            "severity": "low",
            "weakness": (
                "Catalog responses are snapshots and may become stale."
                if "catalog_snapshot" in controls
                else "No catalog snapshot was available for this packet."
            ),
            "consequence": "Catalog absence must never be used as proof of novelty.",
        },
    ]
    return weaknesses


def _claims(
    tic_id: int,
    measurements: dict,
    bootstrap: dict,
    held_out: pd.DataFrame,
    replication: dict,
) -> list[dict]:
    max_held = float(np.max(np.abs(held_out["prediction_error_minutes"])))
    agreement_seconds = float(
        abs(measurements["period_days"] - replication["period_days"]) * 86400.0
    )
    return [
        {
            "claim_id": "M-001",
            "claim_class": "measurement",
            "statement": (
                f"TIC {tic_id} has a fitted period of "
                f"{measurements['period_days']:.9f} days."
            ),
            "evidence": [
                "data/signal_measurements.json#/period_days",
                "data/fitted_events.csv",
            ],
        },
        {
            "claim_id": "M-002",
            "claim_class": "measurement",
            "statement": (
                f"The median fitted eclipse depth is "
                f"{measurements['median_depth_percent']:.3f}% across "
                f"{measurements['event_count']} fitted events."
            ),
            "evidence": [
                "data/signal_measurements.json#/median_depth_percent",
                "data/fitted_events.csv#depth_percent",
            ],
        },
        {
            "claim_id": "M-003",
            "claim_class": "measurement",
            "statement": (
                f"The parametric bootstrap 95% period interval is "
                f"{bootstrap['period_95_percent_low_days']:.9f}–"
                f"{bootstrap['period_95_percent_high_days']:.9f} days."
            ),
            "evidence": [
                "data/significance_tests.json#/bootstrap",
                "data/period_bootstrap.csv",
            ],
        },
        {
            "claim_id": "M-004",
            "claim_class": "measurement",
            "statement": (
                f"The largest held-out event prediction error is {max_held:.2f} minutes."
            ),
            "evidence": ["data/held_out_validation.csv"],
        },
        {
            "claim_id": "M-005",
            "claim_class": "measurement",
            "statement": (
                "The separate median-pairwise-slope implementation differs from "
                f"the primary period by {agreement_seconds:.2f} seconds."
            ),
            "evidence": ["data/replication.json"],
        },
        {
            "claim_id": "M-006",
            "claim_class": "measurement",
            "statement": (
                "A separate Astropy BLS implementation, operating on the "
                "normalized light curve without the primary event centers, "
                f"measures a {replication['lightcurve_bls']['depth_percent']:.3f}% "
                "phase-coherent dip in the preselected period neighborhood."
            ),
            "evidence": [
                "data/replication.json#/lightcurve_bls",
                "data/replication_period_search.csv#depth_percent",
            ],
        },
        {
            "claim_id": "I-001",
            "claim_class": "interpretation",
            "statement": (
                "The stored measurements support a recurring eclipse-like signal "
                "in the public TESS observations."
            ),
            "evidence": [
                "data/primary_events.csv",
                "data/period_search.csv",
                "data/instrument_comparisons.csv",
            ],
        },
        {
            "claim_id": "I-002",
            "claim_class": "interpretation",
            "statement": (
                "An ordinary eclipsing companion is the leading conventional "
                "explanation; the packet does not establish an anomalous object."
            ),
            "evidence": [
                "data/signal_measurements.json",
                "data/conventional_explanations.json",
                "data/unresolved_weaknesses.json",
            ],
        },
        {
            "claim_id": "S-001",
            "claim_class": "speculation",
            "statement": (
                "The companion class could be refined by spectroscopy or additional "
                "independent time-series photometry."
            ),
            "evidence": ["data/unresolved_weaknesses.json"],
        },
    ]


def _numeric_value_count(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(np.isfinite(float(value)))
    if isinstance(value, dict):
        return sum(_numeric_value_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_numeric_value_count(item) for item in value)
    return 0


def _claim_evidence_table(output_dir: Path, claims: list[dict]) -> pd.DataFrame:
    """Resolve every claim reference and fail on broken evidence links."""
    rows: list[dict] = []
    for claim in claims:
        references = claim.get("evidence", [])
        if not references:
            raise ValueError(f"Claim {claim['claim_id']} has no evidence references")
        for reference in references:
            relative_text, separator, selector = str(reference).partition("#")
            relative = Path(relative_text)
            safe_relative = (
                not relative.is_absolute()
                and ".." not in relative.parts
                and relative_text != ""
            )
            path = output_dir / relative if safe_relative else output_dir / "__invalid__"
            exists = bool(safe_relative and path.is_file())
            selector_valid = exists
            numeric_count = 0
            detail = ""
            try:
                if exists and path.suffix.lower() == ".json":
                    value: Any = json.loads(path.read_text(encoding="utf-8"))
                    if separator and selector:
                        if not selector.startswith("/"):
                            raise KeyError("JSON selectors must be JSON pointers")
                        for token in selector.lstrip("/").split("/"):
                            token = token.replace("~1", "/").replace("~0", "~")
                            if isinstance(value, list):
                                value = value[int(token)]
                            else:
                                value = value[token]
                    numeric_count = _numeric_value_count(value)
                elif exists and path.suffix.lower() == ".csv":
                    frame = pd.read_csv(path)
                    if separator and selector:
                        if selector not in frame.columns:
                            raise KeyError(f"CSV column {selector!r} does not exist")
                        frame = frame[[selector]]
                    numeric_count = int(
                        sum(
                            pd.to_numeric(frame[column], errors="coerce")
                            .notna()
                            .sum()
                            for column in frame.columns
                        )
                    )
                elif exists:
                    numeric_count = 0
            except (ValueError, KeyError, IndexError, TypeError) as error:
                selector_valid = False
                detail = f"{type(error).__name__}: {error}"
            rows.append(
                {
                    "claim_id": claim["claim_id"],
                    "claim_class": claim["claim_class"],
                    "evidence_reference": str(reference),
                    "relative_path": relative_text,
                    "selector": selector if separator else "",
                    "path_is_packet_local": safe_relative,
                    "artifact_exists": exists,
                    "selector_valid": selector_valid,
                    "numeric_value_count": numeric_count,
                    "validation_detail": detail,
                }
            )

    table = pd.DataFrame(rows)
    invalid = table[
        ~table["path_is_packet_local"].astype(bool)
        | ~table["artifact_exists"].astype(bool)
        | ~table["selector_valid"].astype(bool)
    ]
    if len(invalid):
        details = invalid[
            ["claim_id", "evidence_reference", "validation_detail"]
        ].to_dict(orient="records")
        raise ValueError(f"Broken claim evidence references: {details}")

    numerical_claim_classes = {"measurement", "interpretation"}
    for claim in claims:
        if claim["claim_class"] not in numerical_claim_classes:
            continue
        linked = table[table["claim_id"] == claim["claim_id"]]
        if int(linked["numeric_value_count"].sum()) <= 0:
            raise ValueError(
                f"Claim {claim['claim_id']} has no stored numerical evidence"
            )
    return table


def _plot_packet(
    lightcurve: pd.DataFrame,
    measurements_table: pd.DataFrame,
    measurements: dict,
    bootstrap_table: pd.DataFrame,
    injections: pd.DataFrame,
    plot_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    times = lightcurve["time"].to_numpy(float)
    flux = lightcurve["flux"].to_numpy(float)

    figure, axis = plt.subplots(figsize=(11, 4))
    step = max(1, len(lightcurve) // 25_000)
    scatter = axis.scatter(
        times[::step],
        flux[::step],
        c=lightcurve["sector"].to_numpy()[::step],
        s=3,
        alpha=0.6,
        cmap="viridis",
    )
    axis.set(xlabel="Time (BTJD)", ylabel="Normalized flux", title="TESS timeline")
    figure.colorbar(scatter, ax=axis, label="Sector")
    figure.tight_layout()
    figure.savefig(plot_dir / "timeline.png", dpi=160)
    plt.close(figure)

    period = float(measurements["period_days"])
    epoch = float(measurements["epoch_btjd"])
    phase_hours = 24.0 * phase_distance(times, epoch, period)
    window = np.abs(phase_hours) <= 12.0
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.scatter(phase_hours[window], flux[window], s=5, alpha=0.35, color="#2457a7")
    bins = np.linspace(-12, 12, 80)
    indices = np.digitize(phase_hours[window], bins)
    centers, medians = [], []
    for index in range(1, len(bins)):
        selected = indices == index
        if np.sum(selected) >= 3:
            centers.append(0.5 * (bins[index - 1] + bins[index]))
            medians.append(float(np.median(flux[window][selected])))
    axis.plot(centers, medians, color="#d33f49", lw=2, label="bin median")
    axis.set(
        xlabel="Hours from fitted eclipse center",
        ylabel="Normalized flux",
        title="Phase-folded eclipse",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "phase_fold.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    if "depth_error_percent" in measurements_table:
        axis.errorbar(
            measurements_table["sector"],
            measurements_table["depth_percent"],
            yerr=measurements_table["depth_error_percent"],
            fmt="o",
            capsize=3,
            color="#2457a7",
        )
    else:
        axis.scatter(
            measurements_table["sector"], measurements_table["depth_percent"]
        )
    axis.axhline(
        measurements["median_depth_percent"], color="#d33f49", ls="--", label="median"
    )
    axis.set(xlabel="TESS sector", ylabel="Depth (%)", title="Depth by sector")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "sector_depths.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(bootstrap_table["period_days"], bins=40, color="#2457a7", alpha=0.85)
    axis.axvline(period, color="#d33f49", lw=2, label="weighted fit")
    axis.set(
        xlabel="Period (days)",
        ylabel="Bootstrap trials",
        title="Parametric period bootstrap",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "period_bootstrap.png", dpi=160)
    plt.close(figure)

    summary = injection_summary(injections)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        summary["injected_depth_percent"],
        100.0 * summary["recovery_rate"],
        marker="o",
        color="#2457a7",
    )
    axis.set(
        xlabel="Injected depth (%)",
        ylabel="Recovery rate (%)",
        ylim=(-3, 103),
        title="Targeted synthetic-signal recovery",
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "injection_recovery.png", dpi=160)
    plt.close(figure)


def _table_html(frame: pd.DataFrame, rows: int = 12) -> str:
    display = frame.head(rows).copy()
    return display.to_html(index=False, border=0, classes="data-table", na_rep="—")


def _render_html(
    output_dir: Path,
    tic_id: int,
    summary: dict,
    claims: list[dict],
    measurements: dict,
    qc: pd.DataFrame,
    cleaning_decisions: pd.DataFrame,
    period_search: pd.DataFrame,
    comparisons: pd.DataFrame,
    significance: dict,
    injection_results: pd.DataFrame,
    injection_results_summary: pd.DataFrame,
    held_out: pd.DataFrame,
    replication: dict,
    claim_evidence: pd.DataFrame,
    explanations: list[dict],
    weaknesses: list[dict],
    artifact_manifest: pd.DataFrame,
) -> None:
    claim_groups = {
        kind: [claim for claim in claims if claim["claim_class"] == kind]
        for kind in ("measurement", "interpretation", "speculation")
    }

    def claim_cards(kind: str) -> str:
        cards = []
        for claim in claim_groups[kind]:
            links = " · ".join(
                (
                    f'<a href="{html.escape(reference.partition("#")[0], quote=True)}">'
                    f"<code>{html.escape(reference)}</code></a>"
                )
                for reference in claim["evidence"]
            )
            cards.append(
                f'<article class="claim {kind}"><div class="tag">{kind}</div>'
                f"<p>{html.escape(claim['statement'])}</p>"
                f'<div class="evidence">Evidence: {links}</div></article>'
            )
        return "\n".join(cards)

    explanation_rows = pd.DataFrame(explanations)
    weakness_rows = pd.DataFrame(weaknesses)
    bls = replication["lightcurve_bls"]
    bootstrap = significance["bootstrap"]
    permutation = significance["permutation"]
    generated = datetime.now(timezone.utc).isoformat()
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TIC {tic_id} Scientific Evidence Packet</title>
<style>
:root {{ --ink:#132034; --muted:#5e6c80; --paper:#f6f3ec; --card:#fff;
--blue:#2457a7; --green:#17734b; --amber:#a85f00; --red:#a5303b; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--paper);
font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
header {{ padding:48px max(5vw,24px); color:white; background:#122743; }}
header h1 {{ margin:0 0 8px; font-size:clamp(30px,5vw,56px); }} header p {{ max-width:900px; color:#dce7f8; }}
main {{ width:min(1180px,92vw); margin:32px auto 80px; }} section {{ margin:38px 0; }}
h2 {{ border-bottom:2px solid #ccd5df; padding-bottom:8px; }} h3 {{ margin-top:28px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; }}
.metric,.claim,.panel {{ background:var(--card); border:1px solid #dce1e7; border-radius:10px; padding:16px; }}
.metric strong {{ display:block; font-size:25px; color:var(--blue); }} .metric span,.evidence,.muted {{ color:var(--muted); font-size:13px; }}
.tag {{ display:inline-block; font-size:11px; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }}
.measurement {{ border-left:5px solid var(--blue); }} .interpretation {{ border-left:5px solid var(--green); }}
.speculation {{ border-left:5px solid var(--amber); }} .claim p {{ margin:8px 0; }}
.warning {{ border-left:5px solid var(--red); background:#fff5f5; padding:14px 18px; }}
.plots {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:18px; }}
.plots figure {{ margin:0; padding:10px; background:white; border:1px solid #dce1e7; border-radius:10px; }}
.plots img {{ width:100%; height:auto; }} .plots figcaption {{ color:var(--muted); padding:4px 8px; }}
.data-table {{ width:100%; border-collapse:collapse; background:white; font-size:13px; overflow-wrap:anywhere; }}
.data-table th,.data-table td {{ padding:7px 9px; border:1px solid #dce1e7; text-align:left; }}
.data-table th {{ background:#eaf0f7; }} .scroll {{ overflow-x:auto; }}
code {{ background:#eef1f5; padding:1px 4px; border-radius:3px;
overflow-wrap:anywhere; white-space:normal; }}
a {{ color:var(--blue); }}
footer {{ color:var(--muted); margin-top:50px; font-size:13px; }}
</style>
</head>
<body>
<header><h1>Scientific Evidence Packet</h1>
<p>TIC {tic_id} · schema {SCHEMA_VERSION} · generated {html.escape(generated)}.
This report separates stored measurements from interpretation and speculation.
It is an auditable analysis product, not a peer-reviewed classification.</p></header>
<main>
<section>
<h2>Executive result</h2>
<div class="grid">
<div class="metric"><strong>{measurements['period_days']:.9f} d</strong><span>weighted raw-event period</span></div>
<div class="metric"><strong>{measurements['median_depth_percent']:.3f}%</strong><span>median fitted depth</span></div>
<div class="metric"><strong>{measurements['event_count']}</strong><span>fitted events</span></div>
<div class="metric"><strong>{replication['period_difference_seconds']:.2f} s</strong><span>primary–replication period difference</span></div>
<div class="metric"><strong>{bls['depth_percent']:.3f}%</strong><span>separate light-curve BLS depth</span></div>
</div>
<p class="warning"><strong>Conservative conclusion:</strong> recurring eclipse-like
photometry is measured. An ordinary eclipsing companion is favored. The packet
does not establish a planet, novelty, artificial origin, or new physics.</p>
</section>
<section><h2>Claims and evidence</h2>
<h3>Measurements</h3><div class="grid">{claim_cards('measurement')}</div>
<h3>Interpretations</h3><div class="grid">{claim_cards('interpretation')}</div>
<h3>Speculation</h3><div class="grid">{claim_cards('speculation')}</div>
<h3>Validated claim-to-evidence graph</h3>
<p>Every reference below was checked during generation. Measurement and
interpretation claims must resolve to at least one stored numeric value.</p>
<div class="scroll">{_table_html(claim_evidence, 100)}</div></section>
<section><h2>Plots</h2><div class="plots">
<figure><img src="plots/timeline.png" alt="TESS timeline"><figcaption>Normalized public TESS light curve by sector.</figcaption></figure>
<figure><img src="plots/phase_fold.png" alt="Phase-folded eclipse"><figcaption>Data folded on the conservative ephemeris.</figcaption></figure>
<figure><img src="plots/sector_depths.png" alt="Sector depths"><figcaption>Fitted depth by sector.</figcaption></figure>
<figure><img src="plots/period_bootstrap.png" alt="Period bootstrap"><figcaption>Measurement-error bootstrap; not a discovery FAP.</figcaption></figure>
<figure><img src="plots/injection_recovery.png" alt="Injection recovery"><figcaption>Known-ephemeris targeted injection sensitivity.</figcaption></figure>
</div></section>
<section><h2>Data provenance and quality control</h2>
<p>Source FITS files are content-hashed in <code>data/provenance.json</code>.
Every raw cadence found in those files has a retained/removed disposition in
<code>data/cadence_disposition.csv</code>.</p><div class="scroll">{_table_html(qc)}</div></section>
<section><h2>Cleaning decisions</h2>
<p>The decision IDs are joined to every removed cadence in
<code>data/cadence_disposition.csv</code>.</p>
<div class="scroll">{_table_html(cleaning_decisions, 30)}</div></section>
<section><h2>Period searches and alias history</h2>
<p>Legacy broad-search maxima and partial-support aliases are preserved as
search history. Only the solution consistent with all five fitted event centers,
including consecutive cycles, is adopted as the conservative measurement.</p>
<div class="scroll">{_table_html(period_search, 80)}</div></section>
<section><h2>Camera, CCD, filter, and sector comparisons</h2>
<div class="scroll">{_table_html(comparisons, 30)}</div></section>
<section><h2>Bootstrap and permutation diagnostics</h2>
<div class="grid">
<div class="metric"><strong>{bootstrap['period_95_percent_low_days']:.9f}–{bootstrap['period_95_percent_high_days']:.9f} d</strong><span>parametric bootstrap 95% interval; {bootstrap['trials']} trials</span></div>
<div class="metric"><strong>{permutation['diagnostic_p_value']:.5f}</strong><span>post-selection cycle-label coherence diagnostic; not a global false-alarm probability</span></div>
</div>
<p class="warning">{html.escape(permutation['limitation'])}</p></section>
<section><h2>Held-out validation</h2><div class="scroll">{_table_html(held_out, 30)}</div></section>
<section><h2>Independent replication implementation</h2>
<p>The median-pairwise-slope estimator independently re-fits the stored event
centers. A separate Astropy Box Least Squares path works directly from the
normalized light curve in a preselected narrow period neighborhood and receives
no primary event centers, cycle labels, or depths.</p>
<div class="grid">
<div class="metric"><strong>{replication['period_days']:.9f} d</strong><span>pairwise-slope event-center replication</span></div>
<div class="metric"><strong>{bls['period_days']:.9f} d</strong><span>light-curve BLS local-period replication</span></div>
<div class="metric"><strong>{bls['depth_snr']:.1f}</strong><span>BLS depth signal-to-noise ratio</span></div>
<div class="metric"><strong>{'yes' if bls['replication_detection'] else 'no'}</strong><span>predeclared BLS replication gate passed</span></div>
</div>
<p class="warning">{html.escape(bls['limitation'])}</p></section>
<section><h2>Synthetic signal injection</h2>
<p>These are targeted, known-ephemeris tests. They do not measure blind-search completeness.</p>
<div class="scroll">{_table_html(injection_results_summary, 30)}</div>
<details><summary>Trial detail</summary><div class="scroll">{_table_html(injection_results, 30)}</div></details></section>
<section><h2>Conventional explanations considered</h2>
<div class="scroll">{_table_html(explanation_rows, 30)}</div></section>
<section><h2>Unresolved weaknesses</h2><div class="scroll">{_table_html(weakness_rows, 30)}</div></section>
<section><h2>Packet inventory</h2><div class="scroll">{_table_html(artifact_manifest, 100)}</div></section>
<footer>Primary classification: {html.escape(str(summary.get('classification')))}.
All numerical claims should be checked against the referenced packet-local artifact
or <code>evidence.sqlite</code>.</footer>
</main></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def _write_sqlite(
    database_path: Path,
    tables: dict[str, pd.DataFrame],
    json_records: dict[str, list[dict]],
) -> None:
    def sqlite_safe(frame: pd.DataFrame) -> pd.DataFrame:
        safe = frame.copy()
        for column in safe.columns:
            if safe[column].dtype == object:
                safe[column] = safe[column].map(
                    lambda value: (
                        json.dumps(_clean_json(value), sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                )
        return safe

    if database_path.exists():
        database_path.unlink()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE packet_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO packet_metadata(key, value) VALUES (?, ?)",
            [("schema_version", SCHEMA_VERSION)],
        )
        for name, frame in tables.items():
            sqlite_safe(frame).to_sql(
                name, connection, if_exists="replace", index=False
            )
        for name, records in json_records.items():
            sqlite_safe(pd.DataFrame(records)).to_sql(
                name, connection, if_exists="replace", index=False
            )


def _artifact_manifest(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.csv":
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(output_dir)),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame(rows)


def generate_evidence_packet(
    tic_id: int,
    *,
    project_root: Path,
    output_dir: Path | None = None,
    max_sectors: int = 8,
    bootstrap_trials: int = DEFAULT_BOOTSTRAP_TRIALS,
    seed: int = DEFAULT_SEED,
    offline: bool = False,
) -> Path:
    """Generate the complete local packet and return its directory."""
    tic_id = int(tic_id)
    project_root = Path(project_root).resolve()
    final_output_dir = (
        Path(output_dir).resolve()
        if output_dir
        else project_root / "results" / "evidence_packets" / f"TIC_{tic_id}"
    )
    output_dir = final_output_dir.parent / f".{final_output_dir.name}.building"
    data_dir = output_dir / "data"
    plot_dir = output_dir / "plots"
    # Build in a clean sibling directory. The last valid packet is replaced
    # only after every analysis, evidence check, plot, database, and report
    # succeeds.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    optional_controls = _copy_optional_controls(project_root, tic_id, data_dir)

    lightcurve, primary_events, repeat, primary_summary = _load_primary_inputs(
        project_root,
        tic_id,
        max_sectors=max_sectors,
        offline=offline,
    )
    if repeat is None:
        raise RuntimeError(
            f"OMEGA did not find a repeat solution for TIC {tic_id}; "
            "a packet was not promoted as complete."
        )
    lightcurve.to_parquet(data_dir / "normalized_lightcurve.parquet", index=False)
    primary_events.to_csv(data_dir / "primary_events.csv", index=False)
    _write_json(data_dir / "primary_repeat.json", repeat)
    _write_json(data_dir / "primary_summary.json", primary_summary)

    products, disposition, qc = _audit_sources(project_root, tic_id, lightcurve)
    disposition.to_csv(data_dir / "cadence_disposition.csv", index=False)
    disposition[disposition["disposition"] == "removed"].to_csv(
        data_dir / "removed_points.csv", index=False
    )
    qc.to_csv(data_dir / "quality_control.csv", index=False)

    cleaning_decisions = pd.DataFrame(
        [
            {
                "decision_id": "CLEAN-001",
                "stage": "source product selection",
                "decision": "one priority-ranked light-curve product per sector",
                "rationale": "avoid mixing reductions within a sector",
                "implementation": "omega_v2.lightcurves.select_products",
            },
            {
                "decision_id": "CLEAN-002",
                "stage": "cadence filtering",
                "decision": "remove nonfinite time/flux and nonzero retained quality flags",
                "rationale": "exclude unusable or explicitly flagged measurements",
                "implementation": "omega_v2.lightcurves.lightcurve_to_frame",
            },
            {
                "decision_id": "CLEAN-003",
                "stage": "segmentation",
                "decision": "split gaps larger than max(6 cadences, 0.20 d)",
                "rationale": "prevent normalization across observing gaps",
                "implementation": "omega_v2.lightcurves._split_indices",
            },
            {
                "decision_id": "CLEAN-004",
                "stage": "normalization",
                "decision": "divide each continuous segment by its median",
                "rationale": "preserve short eclipse-like signals without a flexible detrending model",
                "implementation": "omega_v2.lightcurves.lightcurve_to_frame",
            },
            {
                "decision_id": "CLEAN-005",
                "stage": "short segment exclusion",
                "decision": "remove continuous segments with fewer than 30 retained points",
                "rationale": "local noise/event estimates are unstable for shorter segments",
                "implementation": "omega_v2.lightcurves.lightcurve_to_frame",
            },
            {
                "decision_id": "AUDIT-001",
                "stage": "cache/source reconciliation",
                "decision": (
                    "flag any finite zero-quality source row that should survive "
                    "cleaning but is absent from the normalized cache"
                ),
                "rationale": (
                    "a nonzero count indicates that the cached light curve and "
                    "audited source FITS product are not exactly reconcilable"
                ),
                "implementation": "evidence_packet.generator._audit_sources",
            },
        ]
    )
    cleaning_decisions.to_csv(data_dir / "cleaning_decisions.csv", index=False)

    fitted_events, measurement_source = _measurement_events(
        project_root, tic_id, primary_events, repeat
    )
    fitted_events.to_csv(data_dir / "fitted_events.csv", index=False)
    fit = weighted_ephemeris(
        fitted_events["cycle"].to_numpy(),
        fitted_events["center"].to_numpy(),
        fitted_events["center_error_days"].to_numpy(),
    )
    depth_values = (
        fitted_events["depth_percent"].to_numpy(float)
        if "depth_percent" in fitted_events
        else primary_events["depth_percent"].to_numpy(float)
    )
    fallback_duration = float(repeat.get("duration_days", 3.0 / 24.0) * 24.0)
    duration_hours = _read_joint_duration(project_root, tic_id, fallback_duration)
    signal_measurements = {
        "tic_id": tic_id,
        "measurement_source": measurement_source,
        "event_count": int(len(fitted_events)),
        "sectors": [int(value) for value in sorted(fitted_events["sector"].unique())],
        "epoch_btjd": fit["epoch_btjd"],
        "epoch_error_days": fit["epoch_error_days"],
        "period_days": fit["period_days"],
        "period_error_seconds": fit["period_error_days"] * 86400.0,
        "timing_rms_minutes": fit["rms_minutes"],
        "median_depth_percent": float(np.median(depth_values)),
        "minimum_depth_percent": float(np.min(depth_values)),
        "maximum_depth_percent": float(np.max(depth_values)),
        "duration_hours": duration_hours,
        "measurement_definition": (
            "Weighted linear ephemeris from fitted event centers; depth is the "
            "median of packet-local fitted-event depths."
        ),
    }
    _write_json(data_dir / "signal_measurements.json", signal_measurements)

    period_search = _period_search_table(
        project_root, tic_id, repeat, fit, fitted_events
    )
    period_search.to_csv(data_dir / "period_search.csv", index=False)

    bootstrap_table, bootstrap_summary = bootstrap_ephemeris(
        fitted_events["cycle"].to_numpy(),
        fitted_events["center"].to_numpy(),
        fitted_events["center_error_days"].to_numpy(),
        trials=bootstrap_trials,
        seed=seed,
    )
    bootstrap_table.to_csv(data_dir / "period_bootstrap.csv", index=False)
    permutation_summary = exact_cycle_permutation_test(
        fitted_events["cycle"].to_numpy(),
        fitted_events["center"].to_numpy(),
        fitted_events["center_error_days"].to_numpy(),
    )
    significance = {
        "bootstrap": bootstrap_summary,
        "permutation": permutation_summary,
    }
    _write_json(data_dir / "significance_tests.json", significance)

    held_out = leave_one_event_out(fitted_events)
    held_out.to_csv(data_dir / "held_out_validation.csv", index=False)

    replication_raw = robust_replication_ephemeris(
        fitted_events["cycle"].to_numpy(), fitted_events["center"].to_numpy()
    )
    replication = {
        key: value
        for key, value in replication_raw.items()
        if key != "residuals_days"
    }
    replication.update(
        {
            "primary_period_days": fit["period_days"],
            "period_difference_seconds": abs(
                replication_raw["period_days"] - fit["period_days"]
            )
            * 86400.0,
            "harmonic_agreement": harmonic_period_agreement(
                fit["period_days"], replication_raw["period_days"], 0.001
            ),
            "independence_scope": (
                "Independent regression implementation using the same fitted "
                "astronomical event centers; not independent observational data."
            ),
        }
    )
    replication_search, lightcurve_replication = independent_lightcurve_replication(
        lightcurve,
        candidate_period_days=fit["period_days"],
        candidate_duration_hours=duration_hours,
    )
    replication_search.to_csv(
        data_dir / "replication_period_search.csv", index=False
    )
    replication["lightcurve_bls"] = lightcurve_replication
    _write_json(data_dir / "replication.json", replication)

    injections = targeted_injection_tests(
        lightcurve,
        mask_epoch=fit["epoch_btjd"],
        mask_period=fit["period_days"],
        mask_duration_days=duration_hours / 24.0,
        seed=seed,
    )
    injections.to_csv(data_dir / "injection_tests.csv", index=False)
    injections_summary = injection_summary(injections)
    injections_summary.to_csv(data_dir / "injection_summary.csv", index=False)

    comparisons = _instrument_comparisons(products, qc, fitted_events)
    comparisons.to_csv(data_dir / "instrument_comparisons.csv", index=False)

    provenance = {
        "tic_id": tic_id,
        "public_data_policy": "public astronomical data only",
        "primary_archive": "MAST",
        "primary_collection": "TESS QLP high-level science products",
        "archive_landing_page": "https://archive.stsci.edu/hlsp/qlp",
        "local_normalized_source": primary_summary["source_lightcurve"],
        "acquisition": primary_summary["acquisition"],
        "source_products": [asdict(product) for product in products],
        "source_audit_complete_for_sectors": [
            int(row.sector)
            for row in qc.itertuples()
            if bool(row.source_audit_complete)
        ],
        "note": (
            "Source-product paths are relative to the project root. Hashes "
            "identify the exact local FITS bytes. Archive metadata and catalog "
            "snapshots may change upstream."
        ),
    }
    _write_json(data_dir / "provenance.json", provenance)

    explanations = _conventional_explanations(signal_measurements, optional_controls)
    weaknesses = _weaknesses(products, len(fitted_events), optional_controls)
    _write_json(data_dir / "conventional_explanations.json", explanations)
    _write_json(data_dir / "unresolved_weaknesses.json", weaknesses)

    claims = _claims(
        tic_id, signal_measurements, bootstrap_summary, held_out, replication
    )
    _write_json(data_dir / "claims.json", claims)
    claim_evidence = _claim_evidence_table(output_dir, claims)
    claim_evidence.to_csv(data_dir / "claim_evidence.csv", index=False)

    reproducibility = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": (
            f"python analyze_target.py --tic {tic_id} "
            f"--max-sectors {max_sectors} --bootstrap-trials {bootstrap_trials} "
            f"--seed {seed}" + (" --offline" if offline else "")
        ),
        "project_root": ".",
        "working_directory": "project repository root",
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "random_seed": int(seed),
        "bootstrap_trials": int(bootstrap_trials),
        "environment": {
            "paid_api_used": False,
            "cloud_hosting_required": False,
            "external_database_required": False,
        },
        "code_files": {},
    }
    code_paths = [
        project_root / "analyze_target.py",
        Path(__file__),
        Path(__file__).with_name("analysis.py"),
        project_root / "src" / "omega_v2" / "detection.py",
        project_root / "src" / "omega_v2" / "lightcurves.py",
    ]
    for code_path in code_paths:
        if code_path.exists():
            reproducibility["code_files"][str(code_path.relative_to(project_root))] = {
                "sha256": _sha256(code_path)
            }
    _write_json(data_dir / "reproducibility.json", reproducibility)

    _plot_packet(
        lightcurve,
        fitted_events,
        signal_measurements,
        bootstrap_table,
        injections,
        plot_dir,
    )

    tables = {
        "quality_control": qc,
        "cadence_disposition": disposition,
        "cleaning_decisions": cleaning_decisions,
        "primary_events": primary_events,
        "fitted_events": fitted_events,
        "period_search": period_search,
        "held_out_validation": held_out,
        "injection_tests": injections,
        "injection_summary": injections_summary,
        "instrument_comparisons": comparisons,
        "replication_period_search": replication_search,
        "claim_evidence": claim_evidence,
    }
    _write_sqlite(
        output_dir / "evidence.sqlite",
        tables,
        {
            "claims": claims,
            "signal_measurements": [signal_measurements],
            "significance_tests": [significance],
            "replication": [replication],
            "primary_summary": [primary_summary],
            "primary_repeat": [repeat],
            "provenance": [provenance],
            "reproducibility": [reproducibility],
            "conventional_explanations": explanations,
            "unresolved_weaknesses": weaknesses,
            "source_products": [asdict(product) for product in products],
        },
    )

    artifact_manifest = _artifact_manifest(output_dir)
    artifact_manifest.to_csv(output_dir / "artifact_manifest.csv", index=False)
    _render_html(
        output_dir,
        tic_id,
        primary_summary,
        claims,
        signal_measurements,
        qc,
        cleaning_decisions,
        period_search,
        comparisons,
        significance,
        injections,
        injections_summary,
        held_out,
        replication,
        claim_evidence,
        explanations,
        weaknesses,
        artifact_manifest,
    )
    # Refresh the inventory now that HTML exists. The manifest deliberately
    # excludes itself, avoiding a recursive hash.
    artifact_manifest = _artifact_manifest(output_dir)
    artifact_manifest.to_csv(output_dir / "artifact_manifest.csv", index=False)

    backup_dir = final_output_dir.parent / f".{final_output_dir.name}.previous"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if final_output_dir.exists():
        final_output_dir.rename(backup_dir)
    try:
        output_dir.rename(final_output_dir)
    except Exception:
        if backup_dir.exists() and not final_output_dir.exists():
            backup_dir.rename(final_output_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    return final_output_dir
