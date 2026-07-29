"""TESS light-curve discovery, caching, and conservative normalization."""

from __future__ import annotations

from pathlib import Path
import warnings

import lightkurve as lk
import numpy as np
import pandas as pd


AUTHOR_PRIORITY = [
    "QLP",
    "SPOC",
    "TESS-SPOC",
    "TGLC",
    "TARS",
    "TASOC",
    "GSFC-ELEANOR-LITE",
]


def robust_sigma(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return np.nan
    median = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - median)))


def _as_array(values, dtype=float):
    if hasattr(values, "value"):
        values = values.value
    return np.asarray(np.ma.filled(values, np.nan), dtype=dtype)


def _sector_number(search_result, index: int) -> int:
    row = search_result.table[index]
    for column in ["sequence_number", "sector"]:
        if column in search_result.table.colnames:
            value = row[column]
            try:
                sector = int(value)
                # MAST occasionally exposes non-sector sequence identifiers as
                # TESS light-curve products (for example 1751).  They are not
                # valid TESS observing sectors and must not enter ephemerides.
                return sector if 1 <= sector <= 200 else -1
            except Exception:
                pass
    mission = str(row["mission"])
    digits = "".join(character for character in mission if character.isdigit())
    sector = int(digits) if digits else -1
    return sector if 1 <= sector <= 200 else -1


def select_products(search_result, max_sectors: int = 8) -> list[dict]:
    """Select one consistent, high-priority light curve per observed sector."""
    by_sector: dict[int, list[int]] = {}
    for index in range(len(search_result)):
        sector = _sector_number(search_result, index)
        if sector < 0:
            continue
        by_sector.setdefault(sector, []).append(index)

    selected: list[dict] = []
    for sector in sorted(by_sector):
        indices = by_sector[sector]
        author_to_indices: dict[str, list[int]] = {}
        for index in indices:
            author = str(search_result.table[index]["author"]).strip().upper()
            author_to_indices.setdefault(author, []).append(index)
        chosen = None
        for wanted in AUTHOR_PRIORITY:
            if wanted.upper() in author_to_indices:
                chosen = author_to_indices[wanted.upper()][0]
                break
        if chosen is None:
            chosen = indices[0]
        selected.append(
            {
                "sector": sector,
                "index": chosen,
                "author": str(search_result.table[chosen]["author"]).strip(),
            }
        )

    if len(selected) <= max_sectors:
        return selected

    # Keep sectors spread across the full time baseline, not merely the first
    # observations.  Long-baseline repetition is scientifically valuable.
    positions = np.linspace(0, len(selected) - 1, max_sectors)
    keep = sorted(set(int(round(position)) for position in positions))
    return [selected[index] for index in keep]


def _split_indices(time_values: np.ndarray) -> list[np.ndarray]:
    if len(time_values) < 2:
        return []
    differences = np.diff(time_values)
    positive = differences[np.isfinite(differences) & (differences > 0)]
    if len(positive) == 0:
        return []
    cadence = float(np.nanmedian(positive))
    threshold = max(6.0 * cadence, 0.20)
    locations = np.where(differences > threshold)[0] + 1
    return np.split(np.arange(len(time_values)), locations)


def lightcurve_to_frame(lc, tic_id: int, sector: int, author: str) -> pd.DataFrame:
    """Convert one downloaded product into normalized continuous segments."""
    time_values = _as_array(lc.time)
    flux_values = _as_array(lc.flux)

    quality_column = getattr(lc, "quality", None)
    if quality_column is None:
        quality = np.zeros(len(time_values), dtype=np.int64)
    else:
        quality = _as_array(quality_column)
        quality = np.nan_to_num(quality, nan=1).astype(np.int64)

    length = min(len(time_values), len(flux_values), len(quality))
    time_values = time_values[:length]
    flux_values = flux_values[:length]
    quality = quality[:length]

    finite = np.isfinite(time_values) & np.isfinite(flux_values)
    # Products downloaded with the default bitmask are usually already clean,
    # but preserve only zero-quality cadences where a quality column exists.
    if np.any(quality != 0):
        finite &= quality == 0

    time_values = time_values[finite]
    flux_values = flux_values[finite]
    order = np.argsort(time_values)
    time_values = time_values[order]
    flux_values = flux_values[order]

    frames: list[pd.DataFrame] = []
    for segment_id, indices in enumerate(_split_indices(time_values)):
        if len(indices) < 30:
            continue
        segment_time = time_values[indices]
        segment_flux = flux_values[indices]
        median_flux = float(np.nanmedian(segment_flux))
        if not np.isfinite(median_flux) or median_flux == 0:
            continue
        normalized = segment_flux / median_flux
        residual = normalized - 1.0
        noise = robust_sigma(residual)
        cadence = float(np.nanmedian(np.diff(segment_time)))
        edge_distance = np.minimum(
            segment_time - segment_time[0],
            segment_time[-1] - segment_time,
        )
        frames.append(
            pd.DataFrame(
                {
                    "tic_id": int(tic_id),
                    "sector": int(sector),
                    "author": author,
                    "segment_id": int(segment_id),
                    "time": segment_time,
                    "flux": normalized,
                    "residual_flux": residual,
                    "noise": noise,
                    "cadence_days": cadence,
                    "edge_distance_days": edge_distance,
                }
            )
        )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def download_target_lightcurves(
    tic_id: int,
    download_dir: Path,
    max_sectors: int = 8,
) -> tuple[pd.DataFrame, list[dict]]:
    """Search, select, download, and normalize public TESS light curves."""
    download_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search_result = lk.search_lightcurve(f"TIC {int(tic_id)}", mission="TESS")

    if len(search_result) == 0:
        return pd.DataFrame(), []

    products = select_products(search_result, max_sectors=max_sectors)
    frames: list[pd.DataFrame] = []
    product_rows: list[dict] = []

    for product in products:
        index = product["index"]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lc = search_result[index].download(
                    download_dir=str(download_dir),
                    quality_bitmask="default",
                )
            if lc is None:
                raise RuntimeError("download returned no light curve")
            frame = lightcurve_to_frame(
                lc,
                tic_id=tic_id,
                sector=product["sector"],
                author=product["author"],
            )
            if len(frame):
                frames.append(frame)
            product_rows.append(
                {
                    **product,
                    "download_ok": bool(len(frame)),
                    "points": int(len(frame)),
                    "error": "",
                }
            )
        except Exception as error:
            product_rows.append(
                {
                    **product,
                    "download_ok": False,
                    "points": 0,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    if not frames:
        return pd.DataFrame(), product_rows
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("time").reset_index(drop=True), product_rows
