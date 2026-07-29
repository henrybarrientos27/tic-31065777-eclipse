"""Raw TESSCut pixel localization for top OMEGA v2 candidates."""

from __future__ import annotations

import json
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
from astroquery.mast import Tesscut


def _as_array(values, dtype=float):
    return np.asarray(np.ma.filled(values, np.nan), dtype=dtype)


def _aperture(shape, x, y, radius=1.5):
    yy, xx = np.indices(shape)
    return (xx - x) ** 2 + (yy - y) ** 2 <= radius**2


def _obtain_cutout(
    tic_id: int,
    ra: float,
    dec: float,
    sector: int,
    cache_dir: Path,
    size: int = 15,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"TIC_{int(tic_id)}_sector_{int(sector)}_tesscut.fits"
    if path.exists():
        return path
    cutouts = Tesscut.get_cutouts(
        coordinates=SkyCoord(ra * u.deg, dec * u.deg),
        size=size,
        sector=int(sector),
    )
    if len(cutouts) == 0:
        raise RuntimeError(f"No TESSCut data for TIC {tic_id}, Sector {sector}")
    cutouts[0].writeto(path, overwrite=True)
    return path


def _analyze_sector(
    path: Path,
    ra: float,
    dec: float,
    sector: int,
    epoch: float,
    period: float,
    duration_days: float,
    plot_path: Path,
) -> dict:
    with fits.open(path) as hdul:
        table = hdul[1].data
        time_values = _as_array(table["TIME"])
        cube = _as_array(table["FLUX"])
        quality = (
            np.nan_to_num(_as_array(table["QUALITY"]), nan=1).astype(np.int64)
            if "QUALITY" in table.columns.names
            else np.zeros(len(time_values), dtype=np.int64)
        )
        wcs = WCS(hdul[2].header).celestial
        target_x, target_y = wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))

    clean = (
        np.isfinite(time_values)
        & np.any(np.isfinite(cube), axis=(1, 2))
        & (quality == 0)
    )
    time_values = time_values[clean]
    cube = cube[clean]
    if len(time_values) < 30:
        raise RuntimeError(f"Sector {sector} has too few clean cutout cadences")

    ny, nx = cube.shape[1:]
    border = np.zeros((ny, nx), dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    background = np.nanmedian(cube[:, border], axis=1)
    corrected = cube - background[:, None, None]

    first_cycle = int(math.floor((time_values.min() - epoch) / period)) - 1
    last_cycle = int(math.ceil((time_values.max() - epoch) / period)) + 1
    centers = [epoch + cycle * period for cycle in range(first_cycle, last_cycle + 1)]
    centers = [center for center in centers if time_values.min() <= center <= time_values.max()]

    half = max(0.04, 0.65 * duration_days)
    distance = np.full(len(time_values), np.inf)
    for center in centers:
        distance = np.minimum(distance, np.abs(time_values - center))
    event = distance <= half
    reference = (distance >= max(0.20, 2.0 * half)) & (
        distance <= max(0.80, 6.0 * half)
    )
    if np.sum(event) < 3 or np.sum(reference) < 12:
        raise RuntimeError(
            f"Sector {sector} lacks enough raw-pixel event/reference cadences"
        )

    event_image = np.nanmedian(corrected[event], axis=0)
    reference_image = np.nanmedian(corrected[reference], axis=0)
    missing = reference_image - event_image
    floor = float(np.nanmedian(missing))
    weights = np.clip(missing - floor, 0, None)
    if np.nansum(weights) <= 0:
        raise RuntimeError(f"Sector {sector} produced no positive missing-light image")
    yy, xx = np.indices(weights.shape)
    centroid_x = float(np.nansum(xx * weights) / np.nansum(weights))
    centroid_y = float(np.nansum(yy * weights) / np.nansum(weights))
    offset = float(np.hypot(centroid_x - target_x, centroid_y - target_y))
    brightest_y, brightest_x = np.unravel_index(np.nanargmax(missing), missing.shape)
    brightest_offset = float(np.hypot(brightest_x - target_x, brightest_y - target_y))

    target_aperture = _aperture((ny, nx), target_x, target_y, radius=1.5)
    aperture_flux = np.nansum(corrected[:, target_aperture], axis=1)
    baseline = float(np.nanmedian(aperture_flux[reference]))
    event_level = float(np.nanmedian(aperture_flux[event]))
    raw_depth = (baseline - event_level) / baseline if baseline != 0 else np.nan
    target_missing = float(np.nansum(np.clip(missing[target_aperture], 0, None)))
    total_missing = float(np.nansum(np.clip(missing, 0, None)))
    target_fraction = target_missing / total_missing if total_missing > 0 else np.nan

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    axes[0].scatter(time_values, aperture_flux / baseline, s=3)
    for center in centers:
        axes[0].axvline(center, color="tab:red", alpha=0.5)
    axes[0].set_xlabel("TESS time (BTJD)")
    axes[0].set_ylabel("Raw target-aperture flux / baseline")
    axes[0].set_title(f"Sector {sector}: raw aperture, depth {100*raw_depth:.2f}%")

    image = axes[1].imshow(reference_image, origin="lower", cmap="viridis")
    axes[1].scatter([target_x], [target_y], marker="+", s=180, linewidth=3)
    axes[1].set_title("Reference image and TIC position")
    fig.colorbar(image, ax=axes[1], fraction=0.046)

    image = axes[2].imshow(missing, origin="lower", cmap="magma")
    axes[2].scatter([target_x], [target_y], marker="+", s=180, linewidth=3, label="TIC")
    axes[2].scatter(
        [centroid_x], [centroid_y], marker="x", s=130, linewidth=2.5, label="Missing light"
    )
    axes[2].legend(fontsize=8)
    axes[2].set_title(f"Missing-light centroid offset {offset:.2f} px")
    fig.colorbar(image, ax=axes[2], fraction=0.046)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    return {
        "sector": int(sector),
        "raw_depth_percent": float(100.0 * raw_depth),
        "predicted_events_covered": int(len(centers)),
        "event_cadences": int(np.sum(event)),
        "reference_cadences": int(np.sum(reference)),
        "target_x": float(target_x),
        "target_y": float(target_y),
        "missing_centroid_x": centroid_x,
        "missing_centroid_y": centroid_y,
        "missing_centroid_offset_pixels": offset,
        "brightest_missing_pixel_offset": brightest_offset,
        "target_aperture_missing_light_fraction": target_fraction,
        "pixel_localization_pass": bool(offset <= 1.25 and brightest_offset <= 1.5),
        "cutout_file": str(path),
        "plot_file": str(plot_path),
    }


def raw_pixel_vet(
    tic_id: int,
    ra: float,
    dec: float,
    repeat: dict,
    sectors: list[int],
    cache_dir: Path,
    plot_dir: Path,
    max_sectors: int = 3,
) -> pd.DataFrame:
    selected = sorted(set(int(sector) for sector in sectors))
    if len(selected) > max_sectors:
        positions = np.linspace(0, len(selected) - 1, max_sectors)
        selected = [selected[int(round(position))] for position in positions]

    rows = []
    for sector in selected:
        try:
            path = _obtain_cutout(tic_id, ra, dec, sector, cache_dir)
            rows.append(
                _analyze_sector(
                    path,
                    ra=ra,
                    dec=dec,
                    sector=sector,
                    epoch=float(repeat["epoch"]),
                    period=float(repeat["period_days"]),
                    duration_days=float(repeat.get("duration_days", 0.12)),
                    plot_path=plot_dir / f"TIC_{tic_id}_sector_{sector}_pixel_vet.png",
                )
            )
        except Exception as error:
            rows.append(
                {
                    "sector": int(sector),
                    "pixel_localization_pass": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return pd.DataFrame(rows)

