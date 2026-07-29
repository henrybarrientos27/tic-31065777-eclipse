"""End-to-end public-data follow-up for OMEGA v2 survivors."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .catalog_vetting import deep_catalog_vet, save_catalog_vet
from .independent_surveys import analyze_ztf_candidate
from .modeling import joint_raw_model
from .pipeline import OmegaPaths
from .pixel_vetting import raw_pixel_vet
from .robustness import robust_raw_model_checks


SURVIVOR_MORPHOLOGIES = {
    "planet_scale_or_faint_companion",
    "detached_or_uncertain_periodic",
    "periodic_with_extra_events",
}


def select_followup_targets(
    leaderboard: pd.DataFrame,
    top_n: int = 3,
    minimum_score: float = 30.0,
) -> pd.DataFrame:
    """Choose repeat signals that are not already obvious ordinary binaries."""
    if len(leaderboard) == 0:
        return leaderboard
    selected = leaderboard.copy()
    required = {
        "classification",
        "morphology_hint",
        "omega_score",
        "villanova_known",
    }
    if not required.issubset(selected.columns):
        return selected.iloc[0:0]
    selected = selected[
        (selected["classification"] == "repeat_candidate")
        & selected["morphology_hint"].isin(SURVIVOR_MORPHOLOGIES)
        & (pd.to_numeric(selected["omega_score"], errors="coerce") >= minimum_score)
        & (~selected["villanova_known"].fillna(False).astype(bool))
    ]
    return selected.sort_values("omega_score", ascending=False).head(top_n)


def _load_repeat_and_sectors(tic_id: int, paths: OmegaPaths):
    repeat_path = paths.repeats / f"TIC_{tic_id}_repeat.json"
    product_path = paths.products / f"TIC_{tic_id}_products.csv"
    if not repeat_path.exists():
        raise RuntimeError(f"Missing repeat model: {repeat_path}")
    repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
    if not repeat:
        raise RuntimeError(f"TIC {tic_id} has no linked repeat model")
    products = pd.read_csv(product_path) if product_path.exists() else pd.DataFrame()
    if len(products) and {"download_ok", "sector"}.issubset(products.columns):
        sectors = (
            products.loc[products["download_ok"].astype(bool), "sector"]
            .astype(int)
            .tolist()
        )
        sectors = [sector for sector in sectors if 1 <= sector <= 200]
    else:
        sectors = sorted(
            set(int(row["sector"]) for row in repeat.get("matched_events", []))
        )
    return repeat, sectors


def follow_up_survivor(
    tic_id: int,
    paths: OmegaPaths,
    max_pixel_sectors: int = 3,
    random_trials: int = 20_000,
    resume: bool = True,
) -> dict:
    """Run every computer-only validation stage on one scan survivor."""
    tic_id = int(tic_id)
    vet_root = paths.result_root / "deep_vetting"
    vet_root.mkdir(parents=True, exist_ok=True)
    final_path = vet_root / f"TIC_{tic_id}_autopilot_summary.json"
    if resume and final_path.exists():
        cached = json.loads(final_path.read_text(encoding="utf-8"))
        # Network/DNS failures are operationally retryable and must never be
        # mistaken for scientific rejection. All completed scientific outcomes
        # remain resumable checkpoints.
        if cached.get("final_status") != "deep_followup_failed_or_timed_out":
            return cached

    repeat, sectors = _load_repeat_and_sectors(tic_id, paths)
    catalog = deep_catalog_vet(tic_id)
    catalog_path = vet_root / f"TIC_{tic_id}_catalog_vet.json"
    save_catalog_vet(catalog, catalog_path)
    result = {
        "tic_id": tic_id,
        "catalog_known_or_classified": bool(catalog.get("known_or_classified")),
        "catalog_status": "known" if catalog.get("known_or_classified") else "not_found",
        "catalog_file": str(catalog_path),
        "pixel_status": "not_run",
        "model_status": "not_run",
        "ztf_status": "not_run",
        "robustness_status": "not_run",
    }

    # Catalog hits are retained in the campaign record but do not consume raw
    # pixel downloads and modeling time.
    if catalog.get("known_or_classified"):
        result["final_status"] = "rejected_already_cataloged"
        final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    pixel_table = raw_pixel_vet(
        tic_id,
        ra=float(catalog["ra"]),
        dec=float(catalog["dec"]),
        repeat=repeat,
        sectors=sectors,
        cache_dir=paths.data_root / "pixel_cache",
        plot_dir=paths.plot_root / "pixel_vetting",
        max_sectors=max_pixel_sectors,
    )
    pixel_path = vet_root / f"TIC_{tic_id}_pixel_vet.csv"
    pixel_table.to_csv(pixel_path, index=False)
    passes = int(
        pixel_table.get("pixel_localization_pass", pd.Series(dtype=bool)).sum()
    )
    tested = int(len(pixel_table))
    result.update(
        {
            "pixel_sectors_tested": tested,
            "pixel_sectors_passed": passes,
            "pixel_file": str(pixel_path),
            "pixel_status": (
                "pass"
                if tested >= 2 and passes == tested
                else "partial_pass"
                if passes >= 2
                else "failed_or_inconclusive"
            ),
        }
    )
    if tested < 2 or passes < 2:
        result["final_status"] = "rejected_or_inconclusive_pixel_localization"
        final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    try:
        model = joint_raw_model(
            tic_id,
            ra=float(catalog["ra"]),
            dec=float(catalog["dec"]),
            repeat=repeat,
            pixel_table=pixel_table,
            stellar_radius_rsun=catalog.get("stellar_radius_rsun"),
            result_dir=vet_root,
            plot_dir=paths.plot_root / "modeling",
        )
    except RuntimeError as error:
        # This is a completed scientific outcome, not an infrastructure
        # failure: sparse/fragmented raw events cannot support a joint clock.
        if "Fewer than three raw eclipse events" not in str(error):
            raise
        result.update(
            {
                "model_status": "insufficient_raw_events",
                "final_status": "inconclusive_insufficient_raw_events",
                "model_error": str(error),
            }
        )
        final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    model_path = vet_root / f"TIC_{tic_id}_joint_raw_model.json"
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    result.update(
        {
            "model_status": "pass" if model.get("fit_success") else "failed",
            "model_file": str(model_path),
            "period_days": model.get("period_days"),
            "duration_hours": model.get("duration_hours"),
            "median_depth_percent": model.get("median_depth_percent"),
            "minimum_companion_radius_rjup": model.get(
                "companion_radius_minimum_rjup"
            ),
            "half_period_alias_likely": model.get("half_period_alias_likely"),
            "physical_period_candidate_days": model.get(
                "physical_period_candidate_days"
            ),
            "physical_interpretation": model.get("physical_interpretation"),
            "duration_at_model_upper_bound": model.get(
                "duration_at_model_upper_bound"
            ),
        }
    )
    if not model.get("fit_success"):
        result["final_status"] = "joint_raw_model_failed"
        final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    robustness = robust_raw_model_checks(
        tic_id,
        ra=float(catalog["ra"]),
        dec=float(catalog["dec"]),
        repeat=repeat,
        pixel_table=pixel_table,
        stellar_radius_rsun=catalog.get("stellar_radius_rsun"),
        result_dir=vet_root,
        plot_dir=paths.plot_root / "modeling",
    )
    result["robustness_status"] = (
        "pass" if robustness.get("robust_validation_passes") else "failed"
    )
    result["robustness_file"] = robustness.get("result_file")

    ztf = analyze_ztf_candidate(
        tic_id,
        ra=float(catalog["ra"]),
        dec=float(catalog["dec"]),
        epoch=float(model["epoch_btjd"]),
        period=float(model["period_days"]),
        duration_hours=float(model["duration_hours"]),
        ingress_ratio=float(model["ingress_ratio"]),
        tess_depth_fraction=float(model["median_depth_percent"]) / 100.0,
        raw_path=vet_root / f"TIC_{tic_id}_ztf_raw.csv",
        result_dir=vet_root,
        plot_dir=paths.plot_root / "independent_surveys",
        random_trials=random_trials,
    )
    result["ztf_status"] = ztf.get("classification")
    result["ztf_result_file"] = ztf.get("result_file")
    result["ztf_night_balanced_p"] = (
        (ztf.get("night_balanced_random_epoch_test") or {}).get(
            "empirical_one_sided_p"
        )
    )

    if model.get("duration_at_model_upper_bound"):
        final_status = "periodic_variable_needs_broader_physical_model"
    elif model.get("half_period_alias_likely"):
        final_status = "independently_supported_eclipsing_binary_candidate"
    elif not robustness.get("robust_validation_passes"):
        final_status = "candidate_failed_analysis_choice_robustness"
    elif ztf.get("classification") == "independent_ztf_phase_support":
        final_status = "high_priority_independently_supported_candidate"
    elif ztf.get("classification") == "weak_to_moderate_ztf_phase_support":
        final_status = "high_priority_candidate_with_ztf_hint"
    else:
        final_status = "clean_public_data_candidate_needs_more_confirmation"
    result["final_status"] = final_status
    final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
