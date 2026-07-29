"""Target selection and lightweight live-catalog checks for OMEGA v2."""

from __future__ import annotations

from io import StringIO
import math
import time

import numpy as np
import pandas as pd
import requests


VIZIER_TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
UNVETTED_TABLE = '"J/ApJS/279/50/table2"'
VILLANOVA_SEARCH_URL = "https://tessebs.villanova.edu/search_results"


def _tap_csv(query: str, timeout: int = 90) -> pd.DataFrame:
    response = requests.get(
        VIZIER_TAP_URL,
        params={
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "QUERY": query,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    text = response.text.strip()
    if not text or text.startswith("<?xml"):
        raise RuntimeError(f"VizieR TAP did not return CSV: {text[:500]}")
    return pd.read_csv(StringIO(text))


def fetch_sky_distributed_unvetted_targets(
    count: int,
    tmag_min: float = 9.0,
    tmag_max: float = 13.5,
    seed: int = 20260713,
    ra_bins: int | None = None,
) -> pd.DataFrame:
    """Fetch a reproducible, sky-distributed sample from the unvetted list.

    Consecutive rows often lie in the same sky patch.  Sampling separately in
    right-ascension bins reduces shared-sector artifacts and gives the pilot a
    broader mix of TESS fields.
    """
    if count <= 0:
        raise ValueError("count must be positive")

    bins = ra_bins or min(12, count)
    bins = max(1, int(bins))
    per_bin_needed = int(math.ceil(count / bins))
    fetch_per_bin = max(20, 10 * per_bin_needed)
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []

    for bin_index in range(bins):
        ra_low = 360.0 * bin_index / bins
        ra_high = 360.0 * (bin_index + 1) / bins
        query = f"""
        SELECT TOP {fetch_per_bin}
            TIC, RAJ2000, DEJ2000, Tmag, recno
        FROM {UNVETTED_TABLE}
        WHERE Tmag BETWEEN {float(tmag_min)} AND {float(tmag_max)}
          AND RAJ2000 >= {ra_low}
          AND RAJ2000 < {ra_high}
        """
        frame = _tap_csv(query)
        if len(frame) == 0:
            continue
        take = min(per_bin_needed, len(frame))
        selected = frame.iloc[
            rng.choice(len(frame), size=take, replace=False)
        ].copy()
        selected["ra_bin"] = bin_index
        frames.append(selected)
        time.sleep(0.05)

    if not frames:
        return pd.DataFrame(columns=["tic_id", "ra", "dec", "tmag", "recno"])

    targets = pd.concat(frames, ignore_index=True)
    targets = targets.drop_duplicates(subset=["TIC"])

    # If bin rounding returned too few rows, fill from a broad deterministic
    # query and sample only IDs not already selected.
    if len(targets) < count:
        remaining = count - len(targets)
        query = f"""
        SELECT TOP {max(100, 20 * remaining)}
            TIC, RAJ2000, DEJ2000, Tmag, recno
        FROM {UNVETTED_TABLE}
        WHERE Tmag BETWEEN {float(tmag_min)} AND {float(tmag_max)}
        """
        broad = _tap_csv(query)
        broad = broad[~broad["TIC"].isin(targets["TIC"])]
        if len(broad):
            take = min(remaining, len(broad))
            broad = broad.iloc[
                rng.choice(len(broad), size=take, replace=False)
            ].copy()
            broad["ra_bin"] = -1
            targets = pd.concat([targets, broad], ignore_index=True)

    targets = targets.head(count).rename(
        columns={
            "TIC": "tic_id",
            "RAJ2000": "ra",
            "DEJ2000": "dec",
            "Tmag": "tmag",
        }
    )
    targets["tic_id"] = targets["tic_id"].astype("int64")
    targets["source_catalog"] = "J/ApJS/279/50/table2"
    targets["source_status"] = "unvetted_unvalidated"
    return targets.reset_index(drop=True)


def villanova_catalog_status(tic_id: int, timeout: int = 30) -> dict:
    """Check whether a TIC is already in the live TESS EB catalog."""
    response = requests.get(
        VILLANOVA_SEARCH_URL,
        params={"tic": str(int(tic_id))},
        timeout=timeout,
    )
    response.raise_for_status()
    normalized = " ".join(response.text.split())
    known = "0 objects satisfy the search criteria" not in normalized
    return {
        "villanova_known": bool(known),
        "villanova_url": response.url,
    }

