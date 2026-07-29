"""OMEGA-X orbital-clock falsification using public TESS and ZTF data.

This module does not label timing variation as new physics.  It independently
measures raw TESS eclipse centers, adds night-balanced ZTF epoch blocks, and
compares linear and quadratic ephemerides with a trial-corrected test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import chi2

from .modeling import _load_raw_events, _trapezoid


def _fit_tess_event(data: dict, epoch: float, period: float, duration: float, ingress: float):
    cycle = int(data["cycle"])
    predicted = float(epoch + cycle * period)
    times = np.asarray(data["time"], dtype=float)
    flux = np.asarray(data["flux"], dtype=float)
    noise = max(float(data["noise"]), 1e-6)
    shift_limit = min(0.08, max(0.02, 0.35 * duration))

    near = np.abs(times - predicted) <= 0.65 * duration
    depth_guess = float(1.0 - np.nanmedian(flux[near])) if np.any(near) else 0.02
    depth_guess = float(np.clip(depth_guess, 0.002, 0.12))

    def residuals(parameters):
        shift, depth, baseline, slope = parameters
        center = predicted + shift
        shape = _trapezoid(times, center, duration, ingress)
        model = baseline + slope * (times - center) - depth * shape
        return (flux - model) / noise

    fit = least_squares(
        residuals,
        x0=[0.0, depth_guess, 1.0, 0.0],
        bounds=(
            [-shift_limit, 0.0005, 0.94, -0.20],
            [shift_limit, 0.20, 1.06, 0.20],
        ),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=12000,
    )
    vector = residuals(fit.x)
    dof = max(1, len(vector) - len(fit.x))
    reduced_chi_square = float(np.sum(vector**2) / dof)
    center_error = math.nan
    try:
        covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * reduced_chi_square
        diagonal = float(covariance[0, 0])
        center_error = math.sqrt(diagonal) if diagonal >= 0 else math.nan
    except Exception:
        pass

    cadence_days = float(data["cadence_minutes"]) / 1440.0
    # Formal white-noise timing errors are optimistic for TESS FFIs.  A
    # half-cadence floor is retained before any physical interpretation.
    timing_error = max(
        center_error if np.isfinite(center_error) else 0.0,
        0.5 * cadence_days,
        30.0 / 86400.0,
    )
    return {
        "source": "TESS",
        "sector": None,
        "time_block": "",
        "cycle": cycle,
        "predicted_center_btjd": predicted,
        "measured_center_btjd": float(predicted + fit.x[0]),
        "timing_error_days": float(timing_error),
        "oc_minutes_initial": float(fit.x[0] * 1440.0),
        "depth_percent": float(100.0 * fit.x[1]),
        "points": int(len(times)),
        "cadence_minutes": float(data["cadence_minutes"]),
        "fit_reduced_chi_square": reduced_chi_square,
        "fit_success": bool(fit.success),
    }


def _ztf_nightly_table(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    required = {
        "night",
        "btjd_barycentric",
        "corrected_mag",
        "empirical_noise",
    }
    if not required.issubset(data.columns):
        return pd.DataFrame()
    return (
        data.groupby("night", as_index=False)
        .agg(
            time=("btjd_barycentric", "median"),
            magnitude=("corrected_mag", "median"),
            noise=("empirical_noise", "median"),
            exposures=("corrected_mag", "size"),
        )
        .sort_values("time")
        .reset_index(drop=True)
    )


def _fit_ztf_block(
    block: pd.DataFrame,
    epoch: float,
    period: float,
    duration: float,
    ingress: float,
) -> dict | None:
    if len(block) < 25:
        return None
    times = block["time"].to_numpy(dtype=float)
    y = block["magnitude"].to_numpy(dtype=float)
    sigma = np.clip(block["noise"].to_numpy(dtype=float), 0.005, None)
    shifts = np.linspace(-0.45 * duration, 0.45 * duration, 361)
    profile = []
    amplitudes = []

    for shift in shifts:
        phase = ((times - epoch - shift + 0.5 * period) % period) - 0.5 * period
        shape = _trapezoid(phase, 0.0, duration, ingress)
        if np.sum(shape > 0) < 3:
            profile.append(np.nan)
            amplitudes.append(np.nan)
            continue
        design = np.column_stack([np.ones(len(y)), shape])
        weighted_design = design / sigma[:, None]
        weighted_y = y / sigma
        solution, _, _, _ = np.linalg.lstsq(
            weighted_design, weighted_y, rcond=None
        )
        model = design @ solution
        profile.append(float(np.sum(((y - model) / sigma) ** 2)))
        amplitudes.append(float(solution[1]))

    profile = np.asarray(profile, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    valid = np.isfinite(profile) & np.isfinite(amplitudes) & (amplitudes > 0)
    if not np.any(valid):
        return None
    valid_indices = np.flatnonzero(valid)
    best_index = int(valid_indices[np.argmin(profile[valid])])
    if best_index in {0, len(shifts) - 1}:
        return None
    best_shift = float(shifts[best_index])
    best_chi = float(profile[best_index])
    interval = valid & (profile <= best_chi + 1.0)
    if np.any(interval):
        low = float(np.min(shifts[interval]))
        high = float(np.max(shifts[interval]))
        error = max(0.5 * (high - low), float(shifts[1] - shifts[0]))
    else:
        error = float(shifts[1] - shifts[0])
    error = max(error, 3.0 / 1440.0)

    median_time = float(np.median(times))
    cycle = int(np.rint((median_time - epoch) / period))
    predicted = float(epoch + cycle * period)
    phase_at_best = (
        ((times - epoch - best_shift + 0.5 * period) % period) - 0.5 * period
    )
    support = int(np.sum(_trapezoid(phase_at_best, 0.0, duration, ingress) > 0))
    return {
        "source": "ZTF",
        "sector": None,
        "time_block": str(block["time_block"].iloc[0]),
        "cycle": cycle,
        "predicted_center_btjd": predicted,
        "measured_center_btjd": float(predicted + best_shift),
        "timing_error_days": float(error),
        "oc_minutes_initial": float(best_shift * 1440.0),
        "depth_percent": math.nan,
        "points": int(len(block)),
        "cadence_minutes": math.nan,
        "fit_reduced_chi_square": float(best_chi / max(1, len(block) - 2)),
        "fit_success": True,
        "supporting_nights": support,
        "fitted_amplitude_mag": float(amplitudes[best_index]),
    }


def _ephemeris_fit(
    observations: pd.DataFrame,
    initial_epoch: float,
    initial_period: float,
    quadratic: bool,
    source_offsets: bool = False,
) -> dict:
    cycles = observations["cycle"].to_numpy(dtype=float)
    centers = observations["measured_center_btjd"].to_numpy(dtype=float)
    errors = observations["timing_error_days"].to_numpy(dtype=float)
    center_cycle = float(np.median(cycles))
    x = cycles - center_cycle
    initial = initial_epoch + cycles * initial_period
    y = centers - initial
    columns = [np.ones(len(x)), x]
    if quadratic:
        columns.append(x**2)
    source_names = sorted(observations["source"].astype(str).unique())
    reference_source = source_names[0]
    offset_sources = source_names[1:] if source_offsets else []
    for source in offset_sources:
        columns.append(
            (observations["source"].astype(str).to_numpy() == source).astype(float)
        )
    design = np.column_stack(columns)
    weighted_design = design / errors[:, None]
    weighted_y = y / errors
    coefficients, _, _, _ = np.linalg.lstsq(
        weighted_design, weighted_y, rcond=None
    )
    correction = design @ coefficients
    residuals = y - correction
    chi_square = float(np.sum((residuals / errors) ** 2))
    parameter_count = len(coefficients)
    dof = max(1, len(y) - parameter_count)
    bic = float(chi_square + parameter_count * np.log(len(y)))
    covariance = np.linalg.pinv(weighted_design.T @ weighted_design)
    covariance *= max(1.0, chi_square / dof)
    coefficient_errors = np.sqrt(
        np.where(np.diag(covariance) >= 0, np.diag(covariance), np.nan)
    )

    correction_at_cycle_zero = (
        coefficients[0]
        - coefficients[1] * center_cycle
        + (coefficients[2] * center_cycle**2 if quadratic else 0.0)
    )
    period_correction = coefficients[1]
    if quadratic:
        period_correction -= 2.0 * coefficients[2] * center_cycle
    result = {
        "quadratic": bool(quadratic),
        "source_offsets_fitted": bool(source_offsets),
        "reference_source": reference_source,
        "epoch_btjd": float(initial_epoch + correction_at_cycle_zero),
        "period_days": float(initial_period + period_correction),
        "chi_square": chi_square,
        "degrees_of_freedom": int(dof),
        "reduced_chi_square": float(chi_square / dof),
        "bic": bic,
        "max_absolute_oc_minutes": float(np.max(np.abs(residuals)) * 1440.0),
        "max_normalized_residual": float(np.max(np.abs(residuals / errors))),
        "residual_days": residuals,
        "coefficient_errors": coefficient_errors,
    }
    if quadratic:
        c2 = float(coefficients[2])
        c2_error = float(coefficient_errors[2])
        pdot = 2.0 * c2 / max(initial_period, 1e-9)
        pdot_error = 2.0 * c2_error / max(initial_period, 1e-9)
        result.update(
            {
                "quadratic_coefficient_days_per_cycle2": c2,
                "quadratic_coefficient_error": c2_error,
                "period_derivative_days_per_day": float(pdot),
                "period_derivative_error_days_per_day": float(pdot_error),
                "period_derivative_ms_per_year": float(
                    pdot * 86400.0 * 1000.0 * 365.25
                ),
                "period_derivative_error_ms_per_year": float(
                    pdot_error * 86400.0 * 1000.0 * 365.25
                ),
            }
        )
    if offset_sources:
        offset_start = 3 if quadratic else 2
        result["source_time_offsets_minutes"] = {
            source: float(coefficients[offset_start + index] * 1440.0)
            for index, source in enumerate(offset_sources)
        }
    return result


def run_clock_falsification(
    tic_id: int,
    ra: float,
    dec: float,
    repeat: dict,
    pixel_table: pd.DataFrame,
    joint_model: dict,
    ztf_phase_path: Path | None,
    result_dir: Path,
    plot_dir: Path,
    trials_scanned: int = 1,
) -> dict:
    """Measure and test an orbital clock without assuming an anomaly exists."""
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    detected_epoch = float(joint_model["epoch_btjd"])
    detected_period = float(joint_model["period_days"])
    half_period_alias = bool(joint_model.get("half_period_alias_likely", False))
    primary_parity = 0
    if half_period_alias:
        even_depth = float(joint_model.get("even_cycle_median_depth_percent", 0.0))
        odd_depth = float(joint_model.get("odd_cycle_median_depth_percent", 0.0))
        primary_parity = 0 if even_depth >= odd_depth else 1
        epoch = detected_epoch + primary_parity * detected_period
        period = 2.0 * detected_period
    else:
        epoch = detected_epoch
        period = detected_period
    duration = float(joint_model["duration_hours"]) / 24.0
    ingress = float(joint_model["ingress_ratio"])

    rows = []
    for _, pixel_row in pixel_table.iterrows():
        if not bool(pixel_row.get("pixel_localization_pass", False)):
            continue
        sector = int(pixel_row["sector"])
        datasets = _load_raw_events(
            Path(pixel_row["cutout_file"]),
            ra=ra,
            dec=dec,
            epoch=detected_epoch,
            period=detected_period,
            initial_duration=duration,
            aperture_radius=1.5,
        )
        for data in datasets:
            if half_period_alias:
                detected_cycle = int(data["cycle"])
                if detected_cycle % 2 != primary_parity:
                    continue
                data = dict(data)
                data["cycle"] = (detected_cycle - primary_parity) // 2
            measurement = _fit_tess_event(data, epoch, period, duration, ingress)
            measurement["sector"] = sector
            rows.append(measurement)

    tess_count = len(rows)
    ztf_blocks = 0
    if ztf_phase_path is not None and ztf_phase_path.exists():
        ztf = pd.read_csv(ztf_phase_path)
        nights = _ztf_nightly_table(ztf)
        if len(nights):
            minimum = float(nights["time"].min())
            nights["time_block"] = (
                np.floor((nights["time"] - minimum) / 365.25).astype(int)
            )
            for _, block in nights.groupby("time_block"):
                measurement = _fit_ztf_block(
                    block,
                    epoch=epoch,
                    period=period,
                    duration=duration,
                    ingress=ingress,
                )
                if measurement is not None:
                    rows.append(measurement)
                    ztf_blocks += 1

    observations = pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)
    if len(observations) < 6:
        raise RuntimeError("Fewer than six independent clock measurements")

    linear = _ephemeris_fit(observations, epoch, period, quadratic=False)
    quadratic = _ephemeris_fit(observations, epoch, period, quadratic=True)
    delta_chi_square = max(0.0, linear["chi_square"] - quadratic["chi_square"])
    delta_bic = float(linear["bic"] - quadratic["bic"])
    local_p = float(chi2.sf(delta_chi_square, 1))
    global_p = float(min(1.0, local_p * max(1, int(trials_scanned))))

    # A constant timing offset between surveys can masquerade as a long-term
    # orbital drift.  Refit with one nuisance clock offset per source before
    # treating cross-survey curvature as physical.
    offset_linear = _ephemeris_fit(
        observations, epoch, period, quadratic=False, source_offsets=True
    )
    offset_quadratic = _ephemeris_fit(
        observations, epoch, period, quadratic=True, source_offsets=True
    )
    offset_delta_chi_square = max(
        0.0, offset_linear["chi_square"] - offset_quadratic["chi_square"]
    )
    offset_delta_bic = float(offset_linear["bic"] - offset_quadratic["bic"])
    offset_local_p = float(chi2.sf(offset_delta_chi_square, 1))
    offset_global_p = float(
        min(1.0, offset_local_p * max(1, int(trials_scanned)))
    )

    # A discovery-grade clock anomaly cannot depend on one timing block.
    leave_one_out_rows = []
    if len(observations) >= 7:
        for index, omitted in observations.iterrows():
            subset = observations.drop(index=index)
            subset_linear = _ephemeris_fit(
                subset, epoch, period, quadratic=False, source_offsets=True
            )
            subset_quadratic = _ephemeris_fit(
                subset, epoch, period, quadratic=True, source_offsets=True
            )
            subset_delta_chi_square = max(
                0.0,
                subset_linear["chi_square"] - subset_quadratic["chi_square"],
            )
            subset_delta_bic = float(
                subset_linear["bic"] - subset_quadratic["bic"]
            )
            subset_local_p = float(chi2.sf(subset_delta_chi_square, 1))
            leave_one_out_rows.append(
                {
                    "omitted_index": int(index),
                    "omitted_source": str(omitted["source"]),
                    "omitted_sector": (
                        int(omitted["sector"])
                        if pd.notna(omitted["sector"])
                        else None
                    ),
                    "omitted_time_block": (
                        str(omitted["time_block"])
                        if pd.notna(omitted["time_block"])
                        else ""
                    ),
                    "omitted_cycle": int(omitted["cycle"]),
                    "quadratic_delta_chi_square": subset_delta_chi_square,
                    "quadratic_delta_bic": subset_delta_bic,
                    "quadratic_local_p": subset_local_p,
                    "quadratic_trial_corrected_p": float(
                        min(1.0, subset_local_p * max(1, int(trials_scanned)))
                    ),
                }
            )
    leave_one_out = pd.DataFrame(leave_one_out_rows)
    if len(leave_one_out):
        worst_leave_one_out_p = float(
            leave_one_out["quadratic_trial_corrected_p"].max()
        )
        minimum_leave_one_out_delta_bic = float(
            leave_one_out["quadratic_delta_bic"].min()
        )
        most_influential = leave_one_out.loc[
            leave_one_out["quadratic_trial_corrected_p"].idxmax()
        ].to_dict()
    else:
        worst_leave_one_out_p = 1.0
        minimum_leave_one_out_delta_bic = -math.inf
        most_influential = {}

    source_diagnostics = {}
    for source, subset in observations.groupby("source"):
        if len(subset) < 5:
            continue
        source_linear = _ephemeris_fit(
            subset, epoch, period, quadratic=False
        )
        source_quadratic = _ephemeris_fit(
            subset, epoch, period, quadratic=True
        )
        source_delta_chi_square = max(
            0.0, source_linear["chi_square"] - source_quadratic["chi_square"]
        )
        source_diagnostics[str(source)] = {
            "measurements": int(len(subset)),
            "quadratic_delta_chi_square": source_delta_chi_square,
            "quadratic_delta_bic": float(
                source_linear["bic"] - source_quadratic["bic"]
            ),
            "quadratic_local_p": float(chi2.sf(source_delta_chi_square, 1)),
            "linear_reduced_chi_square": float(
                source_linear["reduced_chi_square"]
            ),
        }
    source_count = int(observations["source"].nunique())
    tess_sectors = int(
        observations.loc[observations["source"] == "TESS", "sector"].nunique()
    )
    baseline_days = float(
        observations["measured_center_btjd"].max()
        - observations["measured_center_btjd"].min()
    )

    naive_clock_anomaly = global_p < 5.7e-7 and delta_bic >= 10
    offset_robust_anomaly = (
        offset_global_p < 5.7e-7
        and offset_delta_bic >= 10
        and worst_leave_one_out_p < 5.7e-7
        and minimum_leave_one_out_delta_bic >= 10
    )

    if (
        offset_robust_anomaly
        and source_count >= 2
        and tess_sectors >= 3
        and baseline_days >= 365
    ):
        classification = "clock_drift_anomaly_requires_ordinary_physics_models"
    elif naive_clock_anomaly and not offset_robust_anomaly:
        classification = "rejected_cross_survey_offset_or_single_block_false_positive"
    elif global_p < 0.01 and delta_bic >= 6:
        classification = "possible_clock_drift_not_discovery_grade"
    elif linear["reduced_chi_square"] > 5 and linear["max_normalized_residual"] > 5:
        classification = "excess_timing_scatter_requires_systematics_check"
    else:
        classification = "clock_consistent_with_linear_ephemeris"

    observations["linear_oc_minutes"] = np.asarray(linear["residual_days"]) * 1440.0
    observations["quadratic_oc_minutes"] = (
        np.asarray(quadratic["residual_days"]) * 1440.0
    )
    observation_path = result_dir / f"TIC_{tic_id}_omega_x_clock_measurements.csv"
    observations.to_csv(observation_path, index=False)
    leave_one_out_path = result_dir / f"TIC_{tic_id}_omega_x_clock_leave_one_out.csv"
    leave_one_out.to_csv(leave_one_out_path, index=False)

    result = {
        "tic_id": int(tic_id),
        "classification": classification,
        "revolutionary_claim_ready": False,
        "why_not_revolutionary_yet": [
            "Timing drift must first be modeled as third-body light-time effects, apsidal motion, mass transfer, magnetic cycles, and tidal evolution.",
            "A new-physics claim requires independent reproduction and a global significance beyond five sigma.",
        ],
        "measurements": int(len(observations)),
        "detected_period_days": detected_period,
        "clock_period_days": period,
        "half_period_alias_corrected": half_period_alias,
        "primary_eclipse_parity": int(primary_parity),
        "tess_event_timings": int(tess_count),
        "ztf_time_blocks": int(ztf_blocks),
        "independent_sources": source_count,
        "tess_sectors": tess_sectors,
        "time_baseline_days": baseline_days,
        "trials_scanned": int(trials_scanned),
        "linear_ephemeris": {
            key: value
            for key, value in linear.items()
            if key not in {"residual_days", "coefficient_errors"}
        },
        "quadratic_ephemeris": {
            key: value
            for key, value in quadratic.items()
            if key not in {"residual_days", "coefficient_errors"}
        },
        "quadratic_delta_chi_square": delta_chi_square,
        "quadratic_delta_bic": delta_bic,
        "quadratic_local_p": local_p,
        "quadratic_trial_corrected_p": global_p,
        "source_offset_clock_test": {
            "linear_bic": float(offset_linear["bic"]),
            "quadratic_bic": float(offset_quadratic["bic"]),
            "quadratic_delta_chi_square": offset_delta_chi_square,
            "quadratic_delta_bic": offset_delta_bic,
            "quadratic_local_p": offset_local_p,
            "quadratic_trial_corrected_p": offset_global_p,
            "fitted_source_offsets_minutes": offset_linear.get(
                "source_time_offsets_minutes", {}
            ),
        },
        "leave_one_out_robustness": {
            "worst_trial_corrected_p": worst_leave_one_out_p,
            "minimum_delta_bic": minimum_leave_one_out_delta_bic,
            "most_influential_measurement": most_influential,
            "table_file": str(leave_one_out_path),
        },
        "source_specific_clock_tests": source_diagnostics,
        "ordinary_physics_tests_required": [
            "third-body light-travel-time effect",
            "apsidal motion or eccentric primary/secondary timing",
            "mass transfer and tidal orbital evolution",
            "starspot-induced timing shifts",
            "survey time-system and cadence systematics",
        ],
        "measurement_file": str(observation_path),
    }

    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for source, marker, color in [("TESS", "o", "tab:blue"), ("ZTF", "s", "tab:orange")]:
        subset = observations[observations["source"] == source]
        if len(subset):
            axis.errorbar(
                subset["cycle"],
                subset["linear_oc_minutes"],
                yerr=subset["timing_error_days"] * 1440.0,
                fmt=marker,
                color=color,
                label=source,
                alpha=0.8,
            )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Orbital cycle")
    axis.set_ylabel("Observed minus linear ephemeris (minutes)")
    axis.set_title(
        f"TIC {tic_id} O-C clock test | ΔBIC={delta_bic:.2f} | "
        f"trial-corrected p={global_p:.3g}\n{classification}"
    )
    axis.legend()
    plot_path = plot_dir / f"TIC_{tic_id}_omega_x_clock_test.png"
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    result["plot_file"] = str(plot_path)

    result_path = result_dir / f"TIC_{tic_id}_omega_x_clock_test.json"
    result["result_file"] = str(result_path)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
