"""Conservative eclipse-event detection and cross-sector period linking."""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .lightcurves import robust_sigma


SEARCH_DURATIONS_HOURS = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]


def _measure_event(
    time_values: np.ndarray,
    residual: np.ndarray,
    center: float,
    duration_days: float,
    edge_distance: float,
) -> dict | None:
    half = 0.5 * duration_days
    event = np.abs(time_values - center) <= half
    reference = (
        (np.abs(time_values - center) >= max(0.20, 1.5 * duration_days))
        & (np.abs(time_values - center) <= max(0.80, 4.0 * duration_days))
    )
    if np.sum(event) < 2 or np.sum(reference) < 10:
        return None

    baseline = float(np.nanmedian(residual[reference]))
    event_level = float(np.nanmedian(residual[event]))
    depth = baseline - event_level
    noise = robust_sigma(residual[reference])
    if not np.isfinite(noise) or noise <= 0:
        noise = float(np.nanstd(residual[reference]))
    if not np.isfinite(depth) or not np.isfinite(noise) or noise <= 0:
        return None

    weights = np.clip(baseline - residual[event], 0, None)
    if np.sum(weights) > 0:
        measured_center = float(np.sum(time_values[event] * weights) / np.sum(weights))
    else:
        measured_center = float(center)

    return {
        "time": measured_center,
        "depth_fraction": float(depth),
        "depth_percent": float(100.0 * depth),
        "noise_fraction": float(noise),
        "event_score": float(depth / noise),
        "duration_hours": float(24.0 * duration_days),
        "event_points": int(np.sum(event)),
        "reference_points": int(np.sum(reference)),
        "edge_distance_days": float(edge_distance),
    }


def detect_dip_events(lightcurve: pd.DataFrame) -> pd.DataFrame:
    """Detect sustained negative events while retaining conservative scores."""
    if len(lightcurve) == 0:
        return pd.DataFrame()

    candidate_rows: list[dict] = []
    grouped = lightcurve.groupby(["sector", "segment_id"], sort=True)

    for (sector, segment_id), segment in grouped:
        segment = segment.sort_values("time")
        times = segment["time"].to_numpy(dtype=float)
        residual = segment["residual_flux"].to_numpy(dtype=float)
        cadence = float(np.nanmedian(np.diff(times)))
        segment_noise = robust_sigma(residual)
        if len(times) < 30 or not np.isfinite(cadence) or cadence <= 0:
            continue
        if not np.isfinite(segment_noise) or segment_noise <= 0:
            continue

        for duration_hours in SEARCH_DURATIONS_HOURS:
            duration_days = duration_hours / 24.0
            window = max(2, int(round(duration_days / cadence)))
            rolling = (
                pd.Series(residual)
                .rolling(window=window, center=True, min_periods=max(2, window // 2))
                .median()
                .to_numpy()
            )
            detection_series = -np.nan_to_num(rolling, nan=0.0)
            minimum_separation = max(1, int(round(0.75 * duration_days / cadence)))
            peak_indices, _ = find_peaks(
                detection_series,
                prominence=max(0.0015, 2.0 * segment_noise),
                distance=minimum_separation,
            )

            for peak_index in peak_indices:
                center = float(times[peak_index])
                edge_distance = float(
                    min(center - times[0], times[-1] - center)
                )
                required_edge = max(0.15, 0.75 * duration_days)
                if edge_distance < required_edge:
                    continue
                measurement = _measure_event(
                    times,
                    residual,
                    center=center,
                    duration_days=duration_days,
                    edge_distance=edge_distance,
                )
                if measurement is None:
                    continue
                if measurement["depth_fraction"] < 0.0025:
                    continue
                if measurement["event_score"] < 2.5:
                    continue
                candidate_rows.append(
                    {
                        "tic_id": int(segment["tic_id"].iloc[0]),
                        "sector": int(sector),
                        "segment_id": int(segment_id),
                        "author": str(segment["author"].iloc[0]),
                        **measurement,
                    }
                )

    if not candidate_rows:
        return pd.DataFrame()

    candidates = pd.DataFrame(candidate_rows).sort_values(
        "event_score", ascending=False
    )

    # A single physical dip is detected at several smoothing durations.  Merge
    # candidates within 0.18 day and retain the strongest measurement.
    accepted: list[pd.Series] = []
    for _, row in candidates.iterrows():
        duplicate = any(
            int(existing["sector"]) == int(row["sector"])
            and abs(float(existing["time"]) - float(row["time"])) <= 0.18
            for existing in accepted
        )
        if not duplicate:
            accepted.append(row)

    events = pd.DataFrame(accepted)
    if len(events) == 0:
        return events

    # Keep enough events for short-period systems without allowing pathological
    # variables to create an enormous period-combination search.
    events = (
        events.sort_values(["sector", "event_score"], ascending=[True, False])
        .groupby("sector", group_keys=False)
        .head(20)
        .sort_values("time")
        .reset_index(drop=True)
    )
    return events


def _unique_cycle_matches(events: pd.DataFrame, epoch: float, period: float, tolerance: float):
    times = events["time"].to_numpy(dtype=float)
    cycles = np.rint((times - epoch) / period).astype(int)
    predicted = epoch + cycles * period
    residual = times - predicted
    matching = np.abs(residual) <= tolerance
    if not np.any(matching):
        return pd.DataFrame()
    selected = events.loc[matching].copy()
    selected["cycle"] = cycles[matching]
    selected["timing_residual_days"] = residual[matching]
    selected = (
        selected.sort_values("event_score", ascending=False)
        .drop_duplicates(subset=["cycle"], keep="first")
        .sort_values("time")
    )
    return selected


def _validate_ephemeris(
    lightcurve: pd.DataFrame,
    epoch: float,
    period: float,
    duration_days: float,
) -> dict:
    """Penalize aliases that predict eclipses in well-covered flat data."""
    rows: list[dict] = []
    half = max(0.04, 0.6 * duration_days)

    for (sector, segment_id), segment in lightcurve.groupby(
        ["sector", "segment_id"], sort=False
    ):
        times = segment["time"].to_numpy(dtype=float)
        residual = segment["residual_flux"].to_numpy(dtype=float)
        if len(times) < 30:
            continue
        first_cycle = int(math.floor((times[0] - epoch) / period)) - 1
        last_cycle = int(math.ceil((times[-1] - epoch) / period)) + 1
        for cycle in range(first_cycle, last_cycle + 1):
            center = epoch + cycle * period
            event = np.abs(times - center) <= half
            reference = (
                (np.abs(times - center) >= max(0.20, 1.8 * half))
                & (np.abs(times - center) <= max(0.80, 6.0 * half))
            )
            if np.sum(event) < 2 or np.sum(reference) < 10:
                continue
            depth = float(
                np.nanmedian(residual[reference]) - np.nanmedian(residual[event])
            )
            noise = robust_sigma(residual[reference])
            score = depth / noise if np.isfinite(noise) and noise > 0 else np.nan
            rows.append(
                {
                    "sector": int(sector),
                    "segment_id": int(segment_id),
                    "cycle": int(cycle),
                    "predicted_time": float(center),
                    "depth_fraction": depth,
                    "score": score,
                    "positive": bool(depth >= 0.0025 and np.isfinite(score) and score >= 2.0),
                }
            )

    if not rows:
        return {
            "covered_windows": 0,
            "positive_windows": 0,
            "negative_windows": 0,
            "coverage_fraction": 0.0,
            "median_depth_percent": np.nan,
            "windows": [],
        }

    windows = pd.DataFrame(rows)
    positive = int(windows["positive"].sum())
    covered = int(len(windows))
    return {
        "covered_windows": covered,
        "positive_windows": positive,
        "negative_windows": covered - positive,
        "coverage_fraction": positive / covered,
        "median_depth_percent": float(
            100.0 * np.nanmedian(windows.loc[windows["positive"], "depth_fraction"])
        )
        if positive
        else np.nan,
        "windows": rows,
    }


def link_repeating_events(
    events: pd.DataFrame,
    lightcurve: pd.DataFrame,
    min_period: float = 0.5,
    max_period: float = 300.0,
) -> dict | None:
    """Link events across sectors and reject period aliases using flat windows."""
    if len(events) < 2:
        return None

    strongest = events.sort_values("event_score", ascending=False).head(24).copy()
    strongest = strongest.sort_values("time").reset_index(drop=True)
    candidate_periods: dict[tuple[int, int], tuple[float, float]] = {}

    for first_index, second_index in combinations(range(len(strongest)), 2):
        first = strongest.iloc[first_index]
        second = strongest.iloc[second_index]
        delta = float(second["time"] - first["time"])
        if delta <= 0:
            continue
        minimum_cycles = max(1, int(math.ceil(delta / max_period)))
        maximum_cycles = min(160, int(math.floor(delta / min_period)))
        for cycles in range(minimum_cycles, maximum_cycles + 1):
            period = delta / cycles
            if not (min_period <= period <= max_period):
                continue
            # Period and phase are both retained in the key.  A 1e-4-day
            # period bin is finer than the initial event-center precision.
            phase = float(first["time"] % period)
            key = (int(round(period * 10000)), int(round(phase * 1000)))
            candidate_periods.setdefault(key, (period, float(first["time"])))

    preliminary: list[dict] = []
    median_duration_days = float(np.nanmedian(strongest["duration_hours"]) / 24.0)

    for period, epoch in candidate_periods.values():
        tolerance = min(0.25, max(0.06, 0.8 * median_duration_days))
        matched = _unique_cycle_matches(strongest, epoch, period, tolerance)
        if len(matched) < 2:
            continue
        unique_sectors = int(matched["sector"].nunique())
        if unique_sectors < 2:
            continue
        timing_rms = float(
            np.sqrt(np.nanmean(matched["timing_residual_days"] ** 2))
        )
        depth_median = float(np.nanmedian(matched["depth_percent"]))
        depth_spread = float(np.nanstd(matched["depth_percent"]))
        preliminary_score = (
            9.0 * len(matched)
            + 7.0 * unique_sectors
            + min(25.0, float(np.nanmedian(matched["event_score"])))
            - 5.0 * timing_rms / tolerance
            - 3.0 * depth_spread / max(depth_median, 0.1)
        )
        preliminary.append(
            {
                "period_days": float(period),
                "epoch": float(epoch),
                "matched_event_count": int(len(matched)),
                "matched_sector_count": unique_sectors,
                "timing_rms_minutes": 24.0 * 60.0 * timing_rms,
                "median_event_depth_percent": depth_median,
                "depth_spread_percent": depth_spread,
                "preliminary_score": float(preliminary_score),
                "matched_events": matched.to_dict(orient="records"),
                "duration_days": median_duration_days,
            }
        )

    if not preliminary:
        return None

    # Validate only the strongest event-link solutions against the full time
    # series; this keeps large scans fast.
    preliminary = sorted(
        preliminary, key=lambda row: row["preliminary_score"], reverse=True
    )[:80]

    validated: list[dict] = []
    for candidate in preliminary:
        coverage = _validate_ephemeris(
            lightcurve,
            epoch=candidate["epoch"],
            period=candidate["period_days"],
            duration_days=candidate["duration_days"],
        )
        final_score = (
            candidate["preliminary_score"]
            + 12.0 * coverage["positive_windows"]
            - 14.0 * coverage["negative_windows"]
        )
        validated.append(
            {
                **candidate,
                **{key: value for key, value in coverage.items() if key != "windows"},
                "predicted_windows": coverage["windows"],
                "period_score": float(final_score),
            }
        )

    validated.sort(
        key=lambda row: (
            row["positive_windows"],
            -row["negative_windows"],
            row["matched_event_count"],
            row["period_score"],
        ),
        reverse=True,
    )
    return validated[0]


def characterize_periodic_shape(
    lightcurve: pd.DataFrame,
    events: pd.DataFrame,
    repeat: dict | None,
) -> dict:
    """Measure first-pass features that separate ordinary EBs from oddities."""
    empty = {
        "primary_depth_percent": np.nan,
        "secondary_depth_percent": np.nan,
        "secondary_to_primary_ratio": np.nan,
        "out_of_eclipse_scatter_percent": np.nan,
        "phase_curve_amplitude_percent": np.nan,
        "unlinked_event_count": int(len(events)),
        "morphology_hint": "unlinked_or_nonperiodic",
    }
    if repeat is None or len(lightcurve) == 0:
        return empty

    period = float(repeat["period_days"])
    epoch = float(repeat["epoch"])
    duration = max(0.04, float(repeat.get("duration_days", 0.12)))
    phase = (
        ((lightcurve["time"].to_numpy(dtype=float) - epoch + 0.5 * period) % period)
        - 0.5 * period
    )
    residual = lightcurve["residual_flux"].to_numpy(dtype=float)

    primary_distance = np.abs(phase)
    secondary_distance = np.abs(np.abs(phase) - 0.5 * period)
    primary = primary_distance <= 0.65 * duration
    secondary = secondary_distance <= 0.65 * duration
    reference = (
        (primary_distance >= max(0.25, 2.5 * duration))
        & (secondary_distance >= max(0.25, 2.5 * duration))
    )

    reference_level = float(np.nanmedian(residual[reference])) if np.any(reference) else 0.0
    primary_depth = (
        reference_level - float(np.nanmedian(residual[primary]))
        if np.sum(primary) >= 2
        else np.nan
    )
    secondary_depth = (
        reference_level - float(np.nanmedian(residual[secondary]))
        if np.sum(secondary) >= 2
        else np.nan
    )
    secondary_ratio = (
        max(0.0, secondary_depth) / primary_depth
        if np.isfinite(primary_depth) and primary_depth > 0 and np.isfinite(secondary_depth)
        else np.nan
    )
    out_scatter = robust_sigma(residual[reference]) if np.any(reference) else np.nan

    # Phase-bin medians suppress cadence noise and expose ellipsoidal/spot-like
    # orbital modulation outside the eclipses.
    bin_edges = np.linspace(-0.5 * period, 0.5 * period, 101)
    indices = np.digitize(phase, bin_edges) - 1
    bin_medians = []
    for index in range(100):
        mask = (indices == index) & reference
        if np.sum(mask) >= 3:
            bin_medians.append(float(np.nanmedian(residual[mask])))
    phase_amplitude = (
        float(np.nanpercentile(bin_medians, 95) - np.nanpercentile(bin_medians, 5))
        if len(bin_medians) >= 10
        else np.nan
    )

    unlinked = 0
    if len(events):
        event_phase = (
            ((events["time"].to_numpy(dtype=float) - epoch + 0.5 * period) % period)
            - 0.5 * period
        )
        primary_like = np.abs(event_phase) <= max(0.18, 1.5 * duration)
        secondary_like = (
            np.abs(np.abs(event_phase) - 0.5 * period) <= max(0.18, 1.5 * duration)
        )
        unlinked = int(np.sum(~primary_like & ~secondary_like))

    primary_percent = 100.0 * primary_depth if np.isfinite(primary_depth) else np.nan
    secondary_percent = 100.0 * secondary_depth if np.isfinite(secondary_depth) else np.nan
    amplitude_percent = 100.0 * phase_amplitude if np.isfinite(phase_amplitude) else np.nan

    if (
        np.isfinite(primary_percent)
        and 0.25 <= primary_percent <= 5.0
        and (not np.isfinite(secondary_ratio) or secondary_ratio < 0.15)
        and (not np.isfinite(amplitude_percent) or amplitude_percent < 1.5)
    ):
        morphology = "planet_scale_or_faint_companion"
    elif (
        (np.isfinite(primary_percent) and primary_percent >= 10.0)
        or (np.isfinite(secondary_ratio) and secondary_ratio >= 0.20)
        or (np.isfinite(amplitude_percent) and amplitude_percent >= 2.0)
    ):
        morphology = "likely_ordinary_eclipsing_binary"
    elif unlinked >= 2:
        morphology = "periodic_with_extra_events"
    else:
        morphology = "detached_or_uncertain_periodic"

    return {
        "primary_depth_percent": primary_percent,
        "secondary_depth_percent": secondary_percent,
        "secondary_to_primary_ratio": secondary_ratio,
        "out_of_eclipse_scatter_percent": 100.0 * out_scatter
        if np.isfinite(out_scatter)
        else np.nan,
        "phase_curve_amplitude_percent": amplitude_percent,
        "unlinked_event_count": unlinked,
        "morphology_hint": morphology,
    }


def summarize_target(
    tic_id: int,
    lightcurve: pd.DataFrame,
    events: pd.DataFrame,
    repeat: dict | None,
    villanova_known: bool = False,
) -> dict:
    sectors = int(lightcurve["sector"].nunique()) if len(lightcurve) else 0
    points = int(len(lightcurve))
    strongest_score = float(events["event_score"].max()) if len(events) else 0.0
    strongest_depth = float(events["depth_percent"].max()) if len(events) else 0.0
    event_count = int(len(events))

    repeat_events = int(repeat["positive_windows"]) if repeat else 0
    repeat_sectors = int(repeat["matched_sector_count"]) if repeat else 0
    period = float(repeat["period_days"]) if repeat else np.nan
    negative_windows = int(repeat["negative_windows"]) if repeat else 0
    timing_rms = float(repeat["timing_rms_minutes"]) if repeat else np.nan

    if repeat and repeat_events >= 3 and repeat_sectors >= 2 and negative_windows <= 1:
        classification = "repeat_candidate"
    elif strongest_score >= 8.0 and strongest_depth >= 1.0:
        classification = "strong_single_or_unlinked"
    elif event_count:
        classification = "weak_event_candidate"
    else:
        classification = "no_significant_events"

    shape = characterize_periodic_shape(lightcurve, events, repeat)

    signal_component = min(30.0, 3.0 * strongest_score)
    repeat_component = 12.0 * repeat_events + 5.0 * repeat_sectors
    baseline_days = (
        float(lightcurve["time"].max() - lightcurve["time"].min())
        if len(lightcurve)
        else 0.0
    )
    baseline_bonus = min(12.0, 2.5 * np.log10(max(1.0, baseline_days)))
    novelty_bonus = 8.0 if not villanova_known else -100.0
    alias_penalty = 12.0 * negative_windows
    detection_score = (
        signal_component
        + repeat_component
        + baseline_bonus
        + novelty_bonus
        - alias_penalty
    )

    # Exotic-interest is deliberately distinct from detection confidence.
    # Deep, short-period binaries can be extremely secure detections while
    # being scientifically ordinary.  This score prioritizes clean, moderate
    # depths, long periods, missing secondaries, extra events, and timing oddity.
    exotic_score = min(25.0, max(0.0, detection_score) / 10.0)
    if villanova_known:
        exotic_score -= 100.0

    primary_depth = shape["primary_depth_percent"]
    secondary_ratio = shape["secondary_to_primary_ratio"]
    phase_amplitude = shape["phase_curve_amplitude_percent"]
    extra_events = int(shape["unlinked_event_count"])

    if repeat and classification == "repeat_candidate":
        exotic_score += min(20.0, 7.0 * np.log10(max(1.0, period)))
        if np.isfinite(primary_depth):
            if 0.25 <= primary_depth <= 5.0:
                exotic_score += 20.0
            elif 5.0 < primary_depth <= 10.0:
                exotic_score += 7.0
            elif primary_depth > 10.0:
                exotic_score -= 15.0
        if np.isfinite(secondary_ratio):
            if secondary_ratio < 0.10:
                exotic_score += 10.0
            elif secondary_ratio >= 0.25:
                exotic_score -= 12.0
        if np.isfinite(phase_amplitude):
            if phase_amplitude < 1.0:
                exotic_score += 8.0
            elif phase_amplitude >= 3.0:
                exotic_score -= 10.0
        exotic_score += min(20.0, 4.0 * extra_events)
        duration_minutes = max(1.0, float(repeat.get("duration_days", 0.12)) * 1440.0)
        if np.isfinite(timing_rms) and 0.08 * duration_minutes <= timing_rms <= duration_minutes:
            exotic_score += 8.0
    elif classification == "strong_single_or_unlinked":
        exotic_score += 24.0

    omega_score = float(exotic_score)

    return {
        "tic_id": int(tic_id),
        "status": "ok",
        "classification": classification,
        "omega_score": omega_score,
        "detection_confidence_score": float(detection_score),
        "exotic_interest_score": omega_score,
        "points": points,
        "sectors": sectors,
        "time_baseline_days": baseline_days,
        "event_count": event_count,
        "strongest_event_score": strongest_score,
        "strongest_depth_percent": strongest_depth,
        "period_days": period,
        "repeat_positive_windows": repeat_events,
        "repeat_sector_count": repeat_sectors,
        "period_negative_windows": negative_windows,
        "timing_rms_minutes": timing_rms,
        "villanova_known": bool(villanova_known),
        **shape,
    }
