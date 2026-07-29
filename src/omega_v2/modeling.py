"""Joint raw-pixel eclipse modeling for deeply vetted OMEGA survivors."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from scipy.optimize import least_squares

from .lightcurves import robust_sigma


def _as_array(values, dtype=float):
    return np.asarray(np.ma.filled(values, np.nan), dtype=dtype)


def _trapezoid(time_values, center, duration, ingress_ratio):
    half = 0.5 * duration
    ingress = ingress_ratio * duration
    flat_half = max(0.0, half - ingress)
    distance = np.abs(time_values - center)
    shape = np.zeros_like(time_values, dtype=float)
    shape[distance <= flat_half] = 1.0
    ramp = (distance > flat_half) & (distance < half)
    if ingress > 0:
        shape[ramp] = (half - distance[ramp]) / ingress
    return np.clip(shape, 0.0, 1.0)


def _load_raw_events(
    path,
    ra,
    dec,
    epoch,
    period,
    initial_duration,
    aperture_radius=1.5,
):
    with fits.open(path) as hdul:
        table = hdul[1].data
        times = _as_array(table["TIME"])
        cube = _as_array(table["FLUX"])
        quality = (
            np.nan_to_num(_as_array(table["QUALITY"]), nan=1).astype(np.int64)
            if "QUALITY" in table.columns.names
            else np.zeros(len(times), dtype=np.int64)
        )
        wcs = WCS(hdul[2].header).celestial
        target_x, target_y = wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))

    clean = np.isfinite(times) & np.any(np.isfinite(cube), axis=(1, 2)) & (quality == 0)
    times = times[clean]
    cube = cube[clean]
    ny, nx = cube.shape[1:]
    border = np.zeros((ny, nx), dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    background = np.nanmedian(cube[:, border], axis=1)
    cube = cube - background[:, None, None]
    yy, xx = np.indices((ny, nx))
    aperture = (
        (xx - target_x) ** 2 + (yy - target_y) ** 2
        <= float(aperture_radius) ** 2
    )
    raw_flux = np.nansum(cube[:, aperture], axis=1)

    first_cycle = int(math.floor((times.min() - epoch) / period)) - 1
    last_cycle = int(math.ceil((times.max() - epoch) / period)) + 1
    datasets = []
    for cycle in range(first_cycle, last_cycle + 1):
        expected = epoch + cycle * period
        local = np.abs(times - expected) <= 0.38
        if np.sum(local) < 25:
            continue
        local_time = times[local]
        local_flux = raw_flux[local]
        # A sector-edge sliver can contain enough points numerically while
        # lacking one side of the local baseline.  Such partial events drove
        # unconstrained depths in the first pilot fit and must not be modeled.
        if local_time.min() > expected - 0.25 or local_time.max() < expected + 0.25:
            continue
        reference = (
            (np.abs(local_time - expected) >= max(0.12, 2.0 * initial_duration))
            & (np.abs(local_time - expected) <= 0.36)
        )
        if np.sum(reference) < 12:
            continue
        baseline = float(np.nanmedian(local_flux[reference]))
        if not np.isfinite(baseline) or baseline == 0:
            continue
        normalized = local_flux / baseline
        noise = robust_sigma(normalized[reference])
        if not np.isfinite(noise) or noise <= 0:
            noise = float(np.nanstd(normalized[reference]))
        cadence = float(np.nanmedian(np.diff(local_time)))
        datasets.append(
            {
                "cycle": int(cycle),
                "expected": float(expected),
                "time": local_time,
                "flux": normalized,
                "noise": float(noise),
                "cadence_minutes": 1440.0 * cadence,
                "source_file": str(path),
            }
        )
    return datasets


def joint_raw_model(
    tic_id: int,
    ra: float,
    dec: float,
    repeat: dict,
    pixel_table: pd.DataFrame,
    stellar_radius_rsun: float | None,
    result_dir: Path,
    plot_dir: Path,
    aperture_radius: float = 1.5,
    excluded_sectors: set[int] | None = None,
    output_tag: str = "",
) -> dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    initial_epoch = float(repeat["epoch"])
    initial_period = float(repeat["period_days"])
    initial_duration = max(0.025, float(repeat.get("duration_days", 1.0 / 24.0)))

    datasets = []
    excluded_sectors = set(excluded_sectors or set())
    for _, row in pixel_table.iterrows():
        if not bool(row.get("pixel_localization_pass", False)):
            continue
        if int(row["sector"]) in excluded_sectors:
            continue
        datasets.extend(
            _load_raw_events(
                Path(row["cutout_file"]),
                ra=ra,
                dec=dec,
                epoch=initial_epoch,
                period=initial_period,
                initial_duration=initial_duration,
                aperture_radius=aperture_radius,
            )
        )
    if len(datasets) < 3:
        raise RuntimeError("Fewer than three raw eclipse events were available")

    event_count = len(datasets)
    depth_guesses = []
    for data in datasets:
        near = np.abs(data["time"] - data["expected"]) <= initial_duration
        if np.sum(near) >= 2:
            depth_guess = float(1.0 - np.nanmedian(data["flux"][near]))
        else:
            depth_guess = 0.025
        if not np.isfinite(depth_guess):
            depth_guess = 0.025
        depth_guesses.append(float(np.clip(depth_guess, 0.003, 0.12)))

    # epoch, period, duration, ingress ratio, then per-event depths, baselines,
    # and local slopes.
    initial = np.concatenate(
        [
            [initial_epoch, initial_period, initial_duration, 0.25],
            depth_guesses,
            np.ones(event_count),
            np.zeros(event_count),
        ]
    )
    lower = np.concatenate(
        [
            [initial_epoch - 0.08, initial_period - 0.02, 0.012, 0.02],
            np.full(event_count, 0.001),
            np.full(event_count, 0.94),
            np.full(event_count, -0.20),
        ]
    )
    upper = np.concatenate(
        [
            [initial_epoch + 0.08, initial_period + 0.02, 0.20, 0.49],
            np.full(event_count, 0.15),
            np.full(event_count, 1.06),
            np.full(event_count, 0.20),
        ]
    )

    def unpack(parameters):
        epoch, period, duration, ingress_ratio = parameters[:4]
        index = 4
        depths = parameters[index : index + event_count]
        index += event_count
        baselines = parameters[index : index + event_count]
        index += event_count
        slopes = parameters[index : index + event_count]
        return epoch, period, duration, ingress_ratio, depths, baselines, slopes

    def residuals(parameters):
        epoch, period, duration, ingress_ratio, depths, baselines, slopes = unpack(
            parameters
        )
        blocks = []
        for index, data in enumerate(datasets):
            center = epoch + data["cycle"] * period
            shape = _trapezoid(data["time"], center, duration, ingress_ratio)
            model = (
                baselines[index]
                + slopes[index] * (data["time"] - center)
                - depths[index] * shape
            )
            blocks.append((data["flux"] - model) / data["noise"])
        return np.concatenate(blocks)

    fit = least_squares(
        residuals,
        x0=initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=40000,
    )
    epoch, period, duration, ingress_ratio, depths, baselines, slopes = unpack(fit.x)
    vector = residuals(fit.x)
    dof = max(1, len(vector) - len(fit.x))
    reduced_chi_square = float(np.sum(vector**2) / dof)

    errors = np.full(len(fit.x), np.nan)
    try:
        covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * reduced_chi_square
        diagonal = np.diag(covariance)
        errors = np.sqrt(np.where(diagonal >= 0, diagonal, np.nan))
    except Exception:
        pass

    rows = []
    phase_values = []
    flux_values = []
    model_values = []
    sector_labels = []
    for index, data in enumerate(datasets):
        center = epoch + data["cycle"] * period
        shape = _trapezoid(data["time"], center, duration, ingress_ratio)
        model = baselines[index] + slopes[index] * (data["time"] - center) - depths[index] * shape
        sector = int(Path(data["source_file"]).name.split("_sector_")[1].split("_")[0])
        rows.append(
            {
                "sector": sector,
                "cycle": int(data["cycle"]),
                "center_btjd": float(center),
                "depth_percent": float(100.0 * depths[index]),
                "depth_error_percent": float(100.0 * errors[4 + index])
                if np.isfinite(errors[4 + index])
                else np.nan,
                "cadence_minutes": data["cadence_minutes"],
                "points": len(data["time"]),
            }
        )
        phase_values.extend((data["time"] - center).tolist())
        flux_values.extend(data["flux"].tolist())
        model_values.extend(model.tolist())
        sector_labels.extend([sector] * len(data["time"]))

    event_table = pd.DataFrame(rows).sort_values("cycle")
    suffix = f"_{output_tag}" if output_tag else ""
    event_path = result_dir / f"TIC_{tic_id}_joint_raw_events{suffix}.csv"
    event_table.to_csv(event_path, index=False)

    even = event_table[event_table["cycle"] % 2 == 0]["depth_percent"]
    odd = event_table[event_table["cycle"] % 2 != 0]["depth_percent"]
    even_depth = float(np.nanmedian(even)) if len(even) else np.nan
    odd_depth = float(np.nanmedian(odd)) if len(odd) else np.nan
    odd_even_difference = abs(even_depth - odd_depth) if len(even) and len(odd) else np.nan
    median_depth = float(np.nanmedian(event_table["depth_percent"]))
    depth_spread = float(np.nanstd(event_table["depth_percent"]))
    odd_even_fraction = (
        float(odd_even_difference / median_depth)
        if np.isfinite(odd_even_difference) and median_depth > 0
        else np.nan
    )
    half_period_alias_likely = bool(
        len(even) >= 3
        and len(odd) >= 3
        and np.isfinite(odd_even_fraction)
        and odd_even_fraction >= 0.25
    )
    radius_ratio = float(np.sqrt(max(0.0, median_depth / 100.0)))
    companion_rjup = (
        float(stellar_radius_rsun * radius_ratio * 9.735)
        if stellar_radius_rsun is not None
        else np.nan
    )

    phase_values = np.asarray(phase_values)
    flux_values = np.asarray(flux_values)
    model_values = np.asarray(model_values)
    sector_labels = np.asarray(sector_labels)
    plot_path = plot_dir / f"TIC_{tic_id}_joint_raw_model{suffix}.png"
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    for sector in sorted(set(sector_labels)):
        mask = sector_labels == sector
        axes[0].scatter(
            24.0 * phase_values[mask],
            flux_values[mask],
            s=5,
            alpha=0.45,
            label=f"Sector {sector}",
        )
    order = np.argsort(phase_values)
    axes[0].plot(24.0 * phase_values[order], model_values[order], color="black", linewidth=1)
    axes[0].set_xlim(-6, 6)
    axes[0].set_xlabel("Hours from fitted eclipse center")
    axes[0].set_ylabel("Raw normalized aperture flux")
    axes[0].set_title(
        f"TIC {tic_id}: joint raw-pixel model, P={period:.8f} d, "
        f"depth={median_depth:.2f}%"
    )
    axes[0].legend(ncol=3, fontsize=8)

    axes[1].errorbar(
        event_table["cycle"],
        event_table["depth_percent"],
        yerr=event_table["depth_error_percent"],
        fmt="o",
    )
    axes[1].axhline(median_depth, linestyle="--", color="black")
    axes[1].set_xlabel("Orbital cycle")
    axes[1].set_ylabel("Fitted raw eclipse depth (%)")
    axes[1].set_title("Per-event depth stability")
    fig.savefig(plot_path, dpi=190)
    plt.close(fig)

    result = {
        "tic_id": int(tic_id),
        "fit_success": bool(fit.success),
        "fit_message": str(fit.message),
        "epoch_btjd": float(epoch),
        "epoch_formal_error_days": float(errors[0]),
        "period_days": float(period),
        "period_formal_error_seconds": float(errors[1] * 86400.0),
        "duration_hours": float(duration * 24.0),
        "duration_at_model_upper_bound": bool(duration >= 0.198),
        "duration_formal_error_hours": float(errors[2] * 24.0),
        "ingress_ratio": float(ingress_ratio),
        "ingress_minutes": float(duration * ingress_ratio * 1440.0),
        "event_count": int(event_count),
        "median_depth_percent": median_depth,
        "depth_spread_percent": depth_spread,
        "even_cycle_median_depth_percent": even_depth,
        "odd_cycle_median_depth_percent": odd_depth,
        "odd_even_depth_difference_percent": odd_even_difference,
        "odd_even_depth_fraction": odd_even_fraction,
        "half_period_alias_likely": half_period_alias_likely,
        "physical_period_candidate_days": float(2.0 * period)
        if half_period_alias_likely
        else float(period),
        "physical_interpretation": (
            "alternating_primary_secondary_eclipses_likely_binary"
            if half_period_alias_likely
            else "single_repeating_eclipse_family"
        ),
        "radius_ratio_minimum": radius_ratio,
        "companion_radius_minimum_rjup": companion_rjup,
        "reduced_chi_square": reduced_chi_square,
        "aperture_radius_pixels": float(aperture_radius),
        "excluded_sectors": sorted(int(value) for value in excluded_sectors),
        "event_file": str(event_path),
        "plot_file": str(plot_path),
    }
    return result
