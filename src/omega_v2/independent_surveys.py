"""Independent public-survey checks for deeply vetted OMEGA candidates.

The first implementation uses ZTF public photometry from IRSA.  It deliberately
reports both a point-level matched-filter result and a night-balanced result.
The latter prevents a rapid burst of exposures on one night from masquerading
as many independent eclipse confirmations.
"""

from __future__ import annotations

import json
import math
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u

from .lightcurves import robust_sigma


ZTF_LIGHTCURVE_URL = (
    "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
)

# ZTF observes from Palomar.  Fixed coordinates avoid depending on Astropy's
# remotely updated site registry and let us convert MJD(UTC) to BJD(TDB).
PALOMAR = EarthLocation.from_geodetic(
    lon=-116.8630 * u.deg,
    lat=33.3563 * u.deg,
    height=1706.0 * u.m,
)


def download_ztf_lightcurve(
    ra: float,
    dec: float,
    output_path: Path,
    radius_arcsec: float = 3.0,
    force: bool = False,
) -> pd.DataFrame:
    """Download a positional ZTF light curve, or reuse its local cache."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return pd.read_csv(output_path)

    response = requests.get(
        ZTF_LIGHTCURVE_URL,
        params={
            "POS": f"CIRCLE {ra} {dec} {radius_arcsec / 3600.0}",
            "FORMAT": "CSV",
            "BAD_CATFLAGS_MASK": "32768",
        },
        timeout=180,
    )
    response.raise_for_status()
    text = response.text.strip()
    if not text or text.startswith("<"):
        frame = pd.DataFrame()
    else:
        frame = pd.read_csv(StringIO(text))
    frame.to_csv(output_path, index=False)
    return frame


def _trapezoid(phase_days: np.ndarray, duration: float, ingress_ratio: float):
    half = 0.5 * duration
    ingress = ingress_ratio * duration
    flat_half = max(0.0, half - ingress)
    distance = np.abs(phase_days)
    shape = np.zeros_like(distance, dtype=float)
    shape[distance <= flat_half] = 1.0
    ramp = (distance > flat_half) & (distance < half)
    if ingress > 0:
        shape[ramp] = (half - distance[ramp]) / ingress
    return np.clip(shape, 0.0, 1.0)


def _phase_days(times: np.ndarray, epoch: float, period: float, shift: float = 0.0):
    return ((times - epoch - shift + 0.5 * period) % period) - 0.5 * period


def _bjd_btjd(frame: pd.DataFrame, ra: float, dec: float) -> np.ndarray:
    """Convert ZTF MJD(UTC) to the TESS-compatible BJD(TDB)-2457000 scale."""
    mjd = pd.to_numeric(frame["mjd"], errors="coerce").to_numpy(dtype=float)
    times = Time(mjd, format="mjd", scale="utc", location=PALOMAR)
    target = SkyCoord(ra * u.deg, dec * u.deg)
    correction = times.light_travel_time(target, kind="barycentric")
    return np.asarray((times.tdb + correction).jd - 2457000.0, dtype=float)


def _prepare_ztf(frame: pd.DataFrame, ra: float, dec: float) -> pd.DataFrame:
    required = {"oid", "mjd", "mag", "magerr", "filtercode"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"ZTF result lacks columns: {sorted(missing)}")

    data = frame.copy()
    for column in ["mjd", "mag", "magerr"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    good = (
        np.isfinite(data["mjd"])
        & np.isfinite(data["mag"])
        & np.isfinite(data["magerr"])
        & (data["magerr"] > 0)
        & (data["magerr"] < 0.10)
    )
    if "catflags" in data.columns:
        good &= pd.to_numeric(data["catflags"], errors="coerce").fillna(1) == 0
    data = data.loc[good].copy()
    if len(data) == 0:
        return data

    data["group"] = data["oid"].astype(str) + "_" + data["filtercode"].astype(str)
    data["btjd_barycentric"] = _bjd_btjd(data, ra=ra, dec=dec)
    data["corrected_mag"] = data["mag"] - data.groupby("group")["mag"].transform(
        "median"
    )

    # Remove only extreme group-specific outliers.  A real 3%-deep eclipse is
    # retained because it is far below this six-MAD boundary.
    keep = np.ones(len(data), dtype=bool)
    for _, indices in data.groupby("group").groups.items():
        index = np.asarray(list(indices))
        values = data.loc[index, "corrected_mag"].to_numpy(dtype=float)
        scatter = robust_sigma(values)
        if np.isfinite(scatter) and scatter > 0:
            keep[data.index.get_indexer(index)] = np.abs(values - np.median(values)) <= 6 * scatter
    data = data.iloc[np.flatnonzero(keep)].copy()

    # Per-group empirical scatter is more realistic than the frequently small
    # pipeline magnitude errors for long-baseline inference.
    group_noise = {}
    for group, subset in data.groupby("group"):
        noise = robust_sigma(subset["corrected_mag"].to_numpy(dtype=float))
        median_error = float(np.nanmedian(subset["magerr"]))
        if not np.isfinite(noise) or noise <= 0:
            noise = median_error
        group_noise[group] = max(float(noise), median_error, 0.005)
    data["empirical_noise"] = data["group"].map(group_noise).astype(float)
    data["night"] = np.floor(data["mjd"]).astype(int)
    return data.sort_values("btjd_barycentric").reset_index(drop=True)


def _template_fit(
    y: np.ndarray,
    shape: np.ndarray,
    groups: np.ndarray,
    sigma: np.ndarray,
) -> dict | None:
    active = shape > 0
    if np.sum(active) < 3 or len(np.unique(groups[active])) < 1:
        return None
    labels, group_index = np.unique(groups, return_inverse=True)
    design = np.zeros((len(y), len(labels) + 1), dtype=float)
    design[np.arange(len(y)), group_index] = 1.0
    design[:, -1] = shape
    weights = 1.0 / np.clip(sigma, 1e-6, None)
    weighted_design = design * weights[:, None]
    weighted_y = y * weights
    solution, _, _, _ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
    model = design @ solution
    residual = y - model
    dof = max(1, len(y) - len(solution))
    reduced_chi_square = float(np.sum((residual / sigma) ** 2) / dof)
    try:
        covariance = np.linalg.inv(weighted_design.T @ weighted_design)
        covariance *= reduced_chi_square
        amplitude_error = float(np.sqrt(covariance[-1, -1]))
    except np.linalg.LinAlgError:
        amplitude_error = math.nan
    amplitude = float(solution[-1])
    return {
        "amplitude_mag": amplitude,
        "amplitude_error_mag": amplitude_error,
        "formal_sigma": (
            amplitude / amplitude_error
            if np.isfinite(amplitude_error) and amplitude_error > 0
            else math.nan
        ),
        "supporting_points": int(np.sum(active)),
        "reduced_chi_square": reduced_chi_square,
    }


def _night_table(data: pd.DataFrame, shape: np.ndarray) -> pd.DataFrame:
    work = data.copy()
    work["template_shape"] = shape
    return (
        work.groupby("night", as_index=False)
        .agg(
            corrected_mag=("corrected_mag", "median"),
            template_shape=("template_shape", "mean"),
            empirical_noise=("empirical_noise", "median"),
            points=("corrected_mag", "size"),
        )
        .assign(group="all_nights")
    )


def _random_epoch_test(
    data: pd.DataFrame,
    epoch: float,
    period: float,
    duration: float,
    ingress_ratio: float,
    observed_amplitude: float,
    trials: int,
    seed: int,
    night_balanced: bool,
) -> dict | None:
    rng = np.random.default_rng(seed)
    times = data["btjd_barycentric"].to_numpy(dtype=float)
    amplitudes = []
    usable_shifts = rng.uniform(0.0, period, trials * 2)
    for shift in usable_shifts:
        # The real phase and its immediate neighborhood are excluded.
        distance_from_real = min(shift, period - shift)
        if distance_from_real < max(2.0 * duration, 0.25):
            continue
        phase = _phase_days(times, epoch=epoch, period=period, shift=shift)
        shape = _trapezoid(phase, duration=duration, ingress_ratio=ingress_ratio)
        if night_balanced:
            test_data = _night_table(data, shape)
            fit = _template_fit(
                test_data["corrected_mag"].to_numpy(dtype=float),
                test_data["template_shape"].to_numpy(dtype=float),
                test_data["group"].to_numpy(),
                test_data["empirical_noise"].to_numpy(dtype=float),
            )
        else:
            fit = _template_fit(
                data["corrected_mag"].to_numpy(dtype=float),
                shape,
                data["group"].to_numpy(),
                data["empirical_noise"].to_numpy(dtype=float),
            )
        if fit is not None and np.isfinite(fit["amplitude_mag"]):
            amplitudes.append(float(fit["amplitude_mag"]))
        if len(amplitudes) >= trials:
            break
    if not amplitudes:
        return None
    random_amplitudes = np.asarray(amplitudes, dtype=float)
    p_value = float(
        (1 + np.sum(random_amplitudes >= observed_amplitude))
        / (len(random_amplitudes) + 1)
    )
    return {
        "random_trials": int(len(random_amplitudes)),
        "random_median_amplitude_mag": float(np.median(random_amplitudes)),
        "random_scatter_mag": float(robust_sigma(random_amplitudes)),
        "observed_percentile": float(
            100.0 * np.mean(random_amplitudes < observed_amplitude)
        ),
        "empirical_one_sided_p": p_value,
    }


def analyze_ztf_candidate(
    tic_id: int,
    ra: float,
    dec: float,
    epoch: float,
    period: float,
    duration_hours: float,
    ingress_ratio: float,
    tess_depth_fraction: float | None,
    raw_path: Path,
    result_dir: Path,
    plot_dir: Path,
    random_trials: int = 20_000,
    force_download: bool = False,
) -> dict:
    """Test a fixed TESS ephemeris against independent public ZTF data."""
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    raw = download_ztf_lightcurve(
        ra=ra,
        dec=dec,
        output_path=raw_path,
        force=force_download,
    )
    if len(raw) == 0:
        return {
            "tic_id": int(tic_id),
            "survey": "ZTF",
            "classification": "no_public_coverage",
            "raw_points": 0,
        }

    data = _prepare_ztf(raw, ra=ra, dec=dec)
    duration = duration_hours / 24.0
    times = data["btjd_barycentric"].to_numpy(dtype=float)
    phase = _phase_days(times, epoch=epoch, period=period)
    shape = _trapezoid(phase, duration=duration, ingress_ratio=ingress_ratio)
    cycle = np.rint((times - epoch) / period).astype(int)
    data["phase_days"] = phase
    data["phase_hours"] = 24.0 * phase
    data["cycle"] = cycle
    data["template_shape"] = shape

    point_fit = _template_fit(
        data["corrected_mag"].to_numpy(dtype=float),
        shape,
        data["group"].to_numpy(),
        data["empirical_noise"].to_numpy(dtype=float),
    )
    if point_fit is None:
        classification = "insufficient_public_coverage"
        result = {
            "tic_id": int(tic_id),
            "survey": "ZTF",
            "classification": classification,
            "raw_points": int(len(raw)),
            "clean_points": int(len(data)),
        }
        return result

    nights = _night_table(data, shape)
    night_fit = _template_fit(
        nights["corrected_mag"].to_numpy(dtype=float),
        nights["template_shape"].to_numpy(dtype=float),
        nights["group"].to_numpy(),
        nights["empirical_noise"].to_numpy(dtype=float),
    )
    if night_fit is None:
        night_fit = {
            "amplitude_mag": math.nan,
            "amplitude_error_mag": math.nan,
            "formal_sigma": math.nan,
            "supporting_points": 0,
            "reduced_chi_square": math.nan,
        }

    point_random = _random_epoch_test(
        data,
        epoch=epoch,
        period=period,
        duration=duration,
        ingress_ratio=ingress_ratio,
        observed_amplitude=point_fit["amplitude_mag"],
        trials=random_trials,
        seed=int(tic_id),
        night_balanced=False,
    )
    night_random = _random_epoch_test(
        data,
        epoch=epoch,
        period=period,
        duration=duration,
        ingress_ratio=ingress_ratio,
        observed_amplitude=night_fit["amplitude_mag"],
        trials=random_trials,
        seed=int(tic_id) + 1,
        night_balanced=True,
    )

    active = shape > 0
    active_cycles = sorted(set(int(value) for value in cycle[active]))
    active_nights = sorted(set(int(value) for value in data.loc[active, "night"]))
    night_p = (
        night_random["empirical_one_sided_p"]
        if night_random is not None
        else math.nan
    )
    night_amplitude = float(night_fit["amplitude_mag"])
    if (
        np.isfinite(night_p)
        and night_p <= 0.01
        and night_amplitude > 0
        and len(active_cycles) >= 3
    ):
        classification = "independent_ztf_phase_support"
    elif (
        np.isfinite(night_p)
        and night_p <= 0.05
        and night_amplitude > 0
        and len(active_cycles) >= 2
    ):
        classification = "weak_to_moderate_ztf_phase_support"
    elif night_amplitude > 0:
        classification = "ztf_positive_but_inconclusive"
    else:
        classification = "ztf_does_not_support_dimming"

    expected_amplitude = (
        -2.5 * math.log10(1.0 - tess_depth_fraction)
        if tess_depth_fraction is not None
        and np.isfinite(tess_depth_fraction)
        and 0 < tess_depth_fraction < 1
        else math.nan
    )
    result = {
        "tic_id": int(tic_id),
        "survey": "ZTF",
        "classification": classification,
        "timing_scale": "BJD_TDB_minus_2457000",
        "raw_points": int(len(raw)),
        "clean_points": int(len(data)),
        "ztf_object_ids": int(data["oid"].nunique()),
        "filters": sorted(str(value) for value in data["filtercode"].unique()),
        "predicted_epoch_btjd": float(epoch),
        "predicted_period_days": float(period),
        "predicted_duration_hours": float(duration_hours),
        "predicted_ingress_ratio": float(ingress_ratio),
        "tess_expected_amplitude_mag": float(expected_amplitude),
        "strict_window_points": int(np.sum(active)),
        "strict_window_nights": int(len(active_nights)),
        "strict_window_cycles": int(len(active_cycles)),
        "cycles_sampled": active_cycles,
        "point_level_fit": point_fit,
        "night_balanced_fit": night_fit,
        "point_level_random_epoch_test": point_random,
        "night_balanced_random_epoch_test": night_random,
        "interpretation_rule": (
            "The night-balanced random-epoch p-value controls the classification; "
            "point-level significance is diagnostic only."
        ),
    }

    table_path = result_dir / f"TIC_{tic_id}_ztf_phase_data.csv"
    result_path = result_dir / f"TIC_{tic_id}_ztf_independent_test.json"
    data.to_csv(table_path, index=False)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    zoom = np.abs(phase) <= max(0.45, 3.0 * duration)
    phase_grid = np.linspace(-max(0.45, 3.0 * duration), max(0.45, 3.0 * duration), 800)
    fitted_curve = night_amplitude * _trapezoid(
        phase_grid, duration=duration, ingress_ratio=ingress_ratio
    )
    expected_curve = expected_amplitude * _trapezoid(
        phase_grid, duration=duration, ingress_ratio=ingress_ratio
    )

    figure, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    axes[0].scatter(
        phase[zoom] * 24.0,
        data.loc[zoom, "corrected_mag"],
        s=22,
        alpha=0.65,
        label="ZTF exposures",
    )
    axes[0].plot(phase_grid * 24.0, fitted_curve, linewidth=2.0, label="Night-balanced ZTF fit")
    if np.isfinite(expected_amplitude):
        axes[0].plot(
            phase_grid * 24.0,
            expected_curve,
            linestyle="--",
            linewidth=1.6,
            label="TESS depth prediction",
        )
    axes[0].axvspan(-0.5 * duration_hours, 0.5 * duration_hours, alpha=0.12)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Hours from TESS-predicted eclipse")
    axes[0].set_ylabel("Group-corrected ZTF magnitude")
    axes[0].legend()
    axes[0].set_title(f"TIC {tic_id}: independent ZTF ephemeris test")

    phase_fraction = phase / period
    bins = np.linspace(-0.5, 0.5, 81)
    centers = 0.5 * (bins[:-1] + bins[1:])
    medians = []
    for left, right in zip(bins[:-1], bins[1:]):
        values = data.loc[
            (phase_fraction >= left) & (phase_fraction < right), "corrected_mag"
        ]
        medians.append(float(np.median(values)) if len(values) else math.nan)
    axes[1].scatter(phase_fraction, data["corrected_mag"], s=8, alpha=0.25)
    axes[1].plot(centers, medians, marker="o", linewidth=1.0)
    axes[1].axvline(0.0, linestyle="--", linewidth=1.0)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Orbital phase")
    axes[1].set_ylabel("Group-corrected ZTF magnitude")
    axes[1].set_title(
        f"Night-balanced empirical p = {night_p:.4g} | {classification}"
    )
    plot_path = plot_dir / f"TIC_{tic_id}_ztf_independent_test.png"
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)

    result["phase_data_file"] = str(table_path)
    result["result_file"] = str(result_path)
    result["plot_file"] = str(plot_path)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
