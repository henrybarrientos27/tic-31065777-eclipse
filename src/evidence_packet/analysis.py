"""Small, deterministic analysis helpers used by the evidence packet.

The primary event detector remains :mod:`omega_v2.detection`.  The functions
here provide auditable uncertainty, validation, injection, and replication
calculations without mutating the original observations.
"""

from __future__ import annotations

from itertools import permutations
import math

from astropy.timeseries import BoxLeastSquares
import numpy as np
import pandas as pd


def robust_sigma(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return float("nan")
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)))


def weighted_ephemeris(
    cycles: np.ndarray,
    centers: np.ndarray,
    errors: np.ndarray | None = None,
) -> dict:
    """Fit ``center = epoch + period * cycle`` with explicit linear algebra."""
    cycles = np.asarray(cycles, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if errors is None:
        errors = np.ones_like(centers)
    errors = np.asarray(errors, dtype=float)
    valid = (
        np.isfinite(cycles)
        & np.isfinite(centers)
        & np.isfinite(errors)
        & (errors > 0)
    )
    cycles, centers, errors = cycles[valid], centers[valid], errors[valid]
    if len(cycles) < 2 or len(np.unique(cycles)) < 2:
        raise ValueError("At least two distinct event cycles are required")
    design = np.column_stack([np.ones(len(cycles)), cycles])
    weights = 1.0 / np.square(errors)
    normal = design.T @ (weights[:, None] * design)
    covariance = np.linalg.pinv(normal)
    parameters = covariance @ (design.T @ (weights * centers))
    model = design @ parameters
    residuals = centers - model
    dof = max(1, len(centers) - 2)
    chi_square = float(np.sum(np.square(residuals / errors)))
    scaled_covariance = covariance * max(1.0, chi_square / dof)
    return {
        "epoch_btjd": float(parameters[0]),
        "period_days": float(parameters[1]),
        "epoch_error_days": float(math.sqrt(max(0.0, scaled_covariance[0, 0]))),
        "period_error_days": float(math.sqrt(max(0.0, scaled_covariance[1, 1]))),
        "residuals_days": residuals,
        "rms_minutes": float(1440.0 * np.sqrt(np.mean(np.square(residuals)))),
        "chi_square": chi_square,
        "degrees_of_freedom": int(dof),
    }


def robust_replication_ephemeris(cycles: np.ndarray, centers: np.ndarray) -> dict:
    """Independent median-of-pairwise-slopes ephemeris implementation.

    This intentionally does not call the weighted least-squares implementation
    or any OMEGA period-linking function.
    """
    cycles = np.asarray(cycles, dtype=float)
    centers = np.asarray(centers, dtype=float)
    valid = np.isfinite(cycles) & np.isfinite(centers)
    cycles, centers = cycles[valid], centers[valid]
    slopes: list[float] = []
    for left in range(len(cycles)):
        for right in range(left + 1, len(cycles)):
            delta_cycle = cycles[right] - cycles[left]
            if delta_cycle != 0:
                slopes.append(float((centers[right] - centers[left]) / delta_cycle))
    if not slopes:
        raise ValueError("Replication needs two distinct event cycles")
    period = float(np.median(slopes))
    epoch = float(np.median(centers - period * cycles))
    residuals = centers - (epoch + period * cycles)
    return {
        "implementation": "median_pairwise_slopes_and_median_intercept",
        "epoch_btjd": epoch,
        "period_days": period,
        "pairwise_slopes": len(slopes),
        "rms_minutes": float(1440.0 * np.sqrt(np.mean(np.square(residuals)))),
        "maximum_absolute_residual_minutes": float(
            1440.0 * np.max(np.abs(residuals))
        ),
        "residuals_days": residuals,
    }


def independent_lightcurve_replication(
    lightcurve: pd.DataFrame,
    *,
    candidate_period_days: float,
    candidate_duration_hours: float,
    period_fraction_half_width: float = 0.0025,
    period_samples: int = 2_001,
) -> tuple[pd.DataFrame, dict]:
    """Replicate the signal with Astropy BLS directly on the light curve.

    The implementation receives a narrow period neighborhood selected by the
    primary analysis, but it does not receive primary event centers, cycles, or
    depths.  It is therefore a code-path and measurement replication rather
    than a blind-search discovery test or independent observation.
    """
    required = {"time", "flux", "sector"}
    missing = required.difference(lightcurve.columns)
    if missing:
        raise KeyError(f"Replication light curve is missing columns: {sorted(missing)}")

    finite = np.isfinite(lightcurve["time"]) & np.isfinite(lightcurve["flux"])
    frame = lightcurve.loc[finite].copy()
    if len(frame) < 30:
        raise ValueError("Replication requires at least 30 finite light-curve points")

    times = frame["time"].to_numpy(dtype=float)
    flux = frame["flux"].to_numpy(dtype=float)
    baseline = float(np.median(flux))
    scatter = robust_sigma(flux - baseline)
    if not np.isfinite(scatter) or scatter <= 0:
        scatter = float(np.std(flux))
    if not np.isfinite(scatter) or scatter <= 0:
        raise ValueError("Replication could not estimate a positive flux uncertainty")

    period = float(candidate_period_days)
    half_width = max(0.0005, float(period_fraction_half_width)) * period
    periods = np.linspace(
        max(0.1, period - half_width),
        period + half_width,
        max(101, int(period_samples)),
    )
    nominal_duration = max(
        float(candidate_duration_hours) / 24.0,
        2.0 * float(np.median(np.diff(np.unique(times)))),
    )
    durations = np.unique(
        np.clip(
            nominal_duration * np.asarray([0.75, 1.0, 1.25]),
            1.0 / 1440.0,
            min(0.25 * period, 2.0),
        )
    )

    model = BoxLeastSquares(times, flux, dy=np.full(len(flux), scatter))
    result = model.power(periods, durations, objective="snr")
    search = pd.DataFrame(
        {
            "period_days": np.asarray(result.period, dtype=float),
            "duration_hours": 24.0 * np.asarray(result.duration, dtype=float),
            "transit_time_btjd": np.asarray(result.transit_time, dtype=float),
            "depth_fraction": np.asarray(result.depth, dtype=float),
            "depth_error_fraction": np.asarray(result.depth_err, dtype=float),
            "power": np.asarray(result.power, dtype=float),
        }
    )
    search["depth_percent"] = 100.0 * search["depth_fraction"]
    search["depth_snr"] = (
        search["depth_fraction"] / search["depth_error_fraction"]
    )
    best_index = int(np.nanargmax(search["power"].to_numpy(dtype=float)))
    best = search.iloc[best_index]

    replicated_period = float(best["period_days"])
    replicated_epoch = float(best["transit_time_btjd"])
    replicated_duration = float(best["duration_hours"]) / 24.0
    predicted = (
        np.abs(phase_distance(times, replicated_epoch, replicated_period))
        <= 0.5 * replicated_duration
    )
    predicted_cycles = np.rint(
        (times[predicted] - replicated_epoch) / replicated_period
    ).astype(int)
    observed_cycles = int(len(np.unique(predicted_cycles)))
    observed_sectors = int(frame.loc[predicted, "sector"].nunique())
    depth_snr = float(best["depth_snr"])
    depth_fraction = float(best["depth_fraction"])
    detected = bool(
        np.isfinite(depth_snr)
        and depth_snr >= 8.0
        and depth_fraction >= 0.0025
        and observed_cycles >= 3
        and observed_sectors >= 2
    )
    summary = {
        "implementation": "astropy.timeseries.BoxLeastSquares_local_period_scan",
        "search_scope": (
            "Narrow period neighborhood supplied by the primary analysis; "
            "no primary event centers, cycles, or depths supplied"
        ),
        "period_neighborhood_days": [float(periods[0]), float(periods[-1])],
        "period_samples": int(len(periods)),
        "duration_grid_hours": [float(24.0 * value) for value in durations],
        "period_days": replicated_period,
        "transit_time_btjd": replicated_epoch,
        "duration_hours": float(best["duration_hours"]),
        "depth_percent": float(best["depth_percent"]),
        "depth_error_percent": float(100.0 * best["depth_error_fraction"]),
        "depth_snr": depth_snr,
        "observed_cycles": observed_cycles,
        "observed_sectors": observed_sectors,
        "replication_detection": detected,
        "limitation": (
            "This checks implementation and measurement agreement on the same "
            "normalized TESS observations. It is not a blind search and is not "
            "independent observational confirmation."
        ),
    }
    return search, summary


def bootstrap_ephemeris(
    cycles: np.ndarray,
    centers: np.ndarray,
    errors: np.ndarray,
    *,
    trials: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Parametric measurement-error bootstrap with a fixed random seed."""
    cycles = np.asarray(cycles, dtype=float)
    centers = np.asarray(centers, dtype=float)
    errors = np.asarray(errors, dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(int(trials)):
        simulated = centers + rng.normal(0.0, errors)
        fit = weighted_ephemeris(cycles, simulated, errors)
        rows.append({"trial": trial, "period_days": fit["period_days"]})
    table = pd.DataFrame(rows)
    low, median, high = np.percentile(table["period_days"], [2.5, 50.0, 97.5])
    summary = {
        "method": "parametric measurement-error bootstrap",
        "trials": int(trials),
        "seed": int(seed),
        "period_median_days": float(median),
        "period_95_percent_low_days": float(low),
        "period_95_percent_high_days": float(high),
        "period_standard_deviation_seconds": float(
            table["period_days"].std(ddof=1) * 86400.0
        ),
        "limitation": (
            "This propagates fitted event-center errors. It is not a blind-search "
            "false-alarm probability and does not model every correlated systematic."
        ),
    }
    return table, summary


def exact_cycle_permutation_test(
    cycles: np.ndarray,
    centers: np.ndarray,
    errors: np.ndarray,
) -> dict:
    """Exact small-sample cycle-label diagnostic.

    This is intentionally reported as a post-selection coherence diagnostic,
    not as a discovery false-alarm probability.
    """
    cycles = np.asarray(cycles, dtype=float)
    centers = np.asarray(centers, dtype=float)
    errors = np.asarray(errors, dtype=float)
    observed = weighted_ephemeris(cycles, centers, errors)
    observed_rms = float(observed["rms_minutes"])
    null_rms = []
    for order in permutations(range(len(cycles))):
        order_array = np.asarray(order)
        permuted = centers[order_array]
        permuted_errors = errors[order_array]
        fit = weighted_ephemeris(cycles, permuted, permuted_errors)
        null_rms.append(float(fit["rms_minutes"]))
    null_rms_array = np.asarray(null_rms)
    count = int(np.sum(null_rms_array <= observed_rms + 1e-12))
    return {
        "method": "exact permutation of event centers across fixed cycle labels",
        "permutations": int(len(null_rms_array)),
        "observed_rms_minutes": observed_rms,
        "permutations_as_or_more_coherent": count,
        "diagnostic_p_value": float(count / len(null_rms_array)),
        "null_rms_median_minutes": float(np.median(null_rms_array)),
        "claim_scope": "post-selection coherence diagnostic only",
        "limitation": (
            "Cycles and events were selected after looking at the data, so this "
            "number must not be interpreted as a global discovery false-alarm rate."
        ),
    }


def leave_one_event_out(
    events: pd.DataFrame,
    *,
    cycle_column: str = "cycle",
    center_column: str = "center",
    error_column: str = "center_error_days",
) -> pd.DataFrame:
    rows = []
    for index, held_out in events.iterrows():
        training = events.drop(index=index)
        fit = weighted_ephemeris(
            training[cycle_column].to_numpy(),
            training[center_column].to_numpy(),
            training[error_column].to_numpy(),
        )
        prediction = (
            fit["epoch_btjd"] + fit["period_days"] * float(held_out[cycle_column])
        )
        rows.append(
            {
                "held_out_sector": int(held_out["sector"]),
                "held_out_cycle": int(held_out[cycle_column]),
                "measured_center_btjd": float(held_out[center_column]),
                "predicted_center_btjd": float(prediction),
                "prediction_error_minutes": float(
                    1440.0 * (float(held_out[center_column]) - prediction)
                ),
                "training_events": int(len(training)),
                "training_period_days": float(fit["period_days"]),
            }
        )
    return pd.DataFrame(rows)


def phase_distance(times: np.ndarray, epoch: float, period: float) -> np.ndarray:
    return ((np.asarray(times) - epoch + 0.5 * period) % period) - 0.5 * period


def targeted_injection_tests(
    lightcurve: pd.DataFrame,
    *,
    mask_epoch: float,
    mask_period: float,
    mask_duration_days: float,
    depths: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02, 0.05),
    trials_per_depth: int = 4,
    injection_period_days: float = 7.123,
    injection_duration_hours: float = 3.0,
    seed: int = 31065777,
) -> pd.DataFrame:
    """Inject boxes and measure them at their known ephemeris.

    Original event windows are excluded.  This is a targeted sensitivity test,
    deliberately distinct from blind period recovery.
    """
    base = lightcurve[np.isfinite(lightcurve["time"]) & np.isfinite(lightcurve["flux"])].copy()
    original_phase = np.abs(
        phase_distance(base["time"].to_numpy(), mask_epoch, mask_period)
    )
    base = base.loc[original_phase > max(0.15, 1.5 * mask_duration_days)].copy()
    times = base["time"].to_numpy(dtype=float)
    original_flux = base["flux"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    half_duration = 0.5 * injection_duration_hours / 24.0
    start = float(np.min(times))
    for depth in depths:
        for trial in range(int(trials_per_depth)):
            epoch = start + 0.8 + trial * 0.37 + float(rng.uniform(0.0, 0.11))
            distance = np.abs(phase_distance(times, epoch, injection_period_days))
            injected = distance <= half_duration
            reference = distance >= max(0.35, 3.0 * half_duration)
            modified = original_flux.copy()
            modified[injected] *= 1.0 - depth
            baseline = float(np.median(modified[reference]))
            measured_depth = baseline - float(np.median(modified[injected]))
            scatter = robust_sigma(modified[reference] - baseline)
            standard_error = scatter * math.sqrt(
                1.0 / max(1, int(np.sum(injected)))
                + 1.0 / max(1, int(np.sum(reference)))
            )
            snr = measured_depth / standard_error if standard_error > 0 else float("nan")
            recovered = bool(
                np.sum(injected) >= 3
                and np.isfinite(snr)
                and snr >= 7.0
                and measured_depth >= 0.5 * depth
            )
            rows.append(
                {
                    "trial": int(trial),
                    "injected_depth_fraction": float(depth),
                    "injected_depth_percent": float(100.0 * depth),
                    "injected_period_days": float(injection_period_days),
                    "injected_duration_hours": float(injection_duration_hours),
                    "injected_epoch_btjd": float(epoch),
                    "injected_points": int(np.sum(injected)),
                    "reference_points": int(np.sum(reference)),
                    "measured_depth_percent": float(100.0 * measured_depth),
                    "matched_filter_snr": float(snr),
                    "recovered": recovered,
                    "test_scope": "targeted_known-ephemeris_recovery",
                }
            )
    return pd.DataFrame(rows)


def injection_summary(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.groupby("injected_depth_percent", as_index=False)
        .agg(
            trials=("recovered", "size"),
            recovered=("recovered", "sum"),
            recovery_rate=("recovered", "mean"),
            median_measured_depth_percent=("measured_depth_percent", "median"),
            median_snr=("matched_filter_snr", "median"),
        )
        .sort_values("injected_depth_percent")
    )


def harmonic_period_agreement(
    primary_period: float | None,
    replication_period: float | None,
    tolerance_fraction: float = 0.02,
) -> bool:
    if not primary_period or not replication_period:
        return False
    ratio = float(primary_period) / float(replication_period)
    return any(
        abs(ratio - expected) <= tolerance_fraction * expected
        for expected in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
    )
