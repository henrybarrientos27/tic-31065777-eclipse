"""Expensive catalog checks reserved for high-ranking OMEGA survivors."""

from __future__ import annotations

from io import StringIO
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import threading
import time

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
import numpy as np
import requests

from .catalogs import villanova_catalog_status


NASA_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


@contextmanager
def _network_timeout(seconds: float = 60.0):
    """Bound third-party catalog calls on macOS/Linux main-thread runs."""
    if (
        seconds <= 0
        or not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    timeout_started = time.monotonic()

    def handle_timeout(signum, frame):
        raise TimeoutError(f"Catalog request exceeded {seconds:.0f} seconds")

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        previous_delay, previous_interval = previous_timer
        if previous_delay > 0:
            remaining = max(
                1e-6, previous_delay - (time.monotonic() - timeout_started)
            )
            signal.setitimer(
                signal.ITIMER_REAL, remaining, previous_interval
            )


def _nasa_toi(tic_id: int) -> list[dict]:
    query = f"""
    SELECT toi, toipfx, tid, tfopwg_disp, pl_orbper, pl_trandurh,
           pl_trandep, pl_rade, ra, dec, st_tmag, rowupdate
    FROM toi WHERE tid = {int(tic_id)}
    """
    response = requests.get(
        NASA_TAP,
        params={"query": " ".join(query.split()), "format": "json"},
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def deep_catalog_vet(tic_id: int) -> dict:
    result: dict = {"tic_id": int(tic_id), "errors": []}

    with _network_timeout():
        tic = Catalogs.query_criteria(catalog="TIC", ID=int(tic_id))
    if len(tic) == 0:
        raise RuntimeError(f"TIC {tic_id} was not found in the TESS Input Catalog")
    row = tic[0]
    ra = float(row["ra"])
    dec = float(row["dec"])
    gaia_id = str(row["GAIA"]).strip()
    coordinate = SkyCoord(ra * u.deg, dec * u.deg)

    def safe_float(column):
        try:
            value = float(row[column])
            return value if np.isfinite(value) else None
        except Exception:
            return None

    result.update(
        {
            "ra": ra,
            "dec": dec,
            "tmag": safe_float("Tmag"),
            "teff_k": safe_float("Teff"),
            "stellar_radius_rsun": safe_float("rad"),
            "stellar_mass_msun": safe_float("mass"),
            "distance_pc": safe_float("d"),
            "gaia_source_id": gaia_id,
        }
    )

    try:
        with _network_timeout():
            result.update(villanova_catalog_status(tic_id))
    except Exception as error:
        result["villanova_known"] = None
        result["errors"].append(f"Villanova: {type(error).__name__}: {error}")

    try:
        with _network_timeout():
            simbad_client = Simbad()
            try:
                simbad_client.add_votable_fields("otype", "otypes")
            except Exception:
                pass
            simbad = simbad_client.query_region(coordinate, radius=5 * u.arcsec)
        result["simbad_match"] = bool(simbad is not None and len(simbad))
        simbad_frame = (
            simbad.to_pandas().astype(str)
            if simbad is not None and len(simbad)
            else None
        )
        result["simbad_rows"] = (
            simbad_frame.to_dict(orient="records")
            if simbad_frame is not None
            else []
        )
        type_text = " " .join(
            simbad_frame[column].str.upper().str.cat(sep=" ")
            for column in simbad_frame.columns
            if column.lower() in {"otype", "otypes"}
        ) if simbad_frame is not None else ""
        variable_markers = ["V*", "EB*", "ECLBIN", "VARIABLE", "PULSV", "ROTV"]
        result["simbad_variable_match"] = any(
            marker in type_text for marker in variable_markers
        )
    except Exception as error:
        result["simbad_match"] = None
        result["simbad_variable_match"] = None
        result["simbad_rows"] = []
        result["errors"].append(f"SIMBAD: {type(error).__name__}: {error}")

    try:
        with _network_timeout():
            from pyasassn.client import SkyPatrolClient

            client = SkyPatrolClient()
            vsx = client.cone_search(
                ra, dec, 5, units="arcsec", catalog="aavsovsx"
            )
        result["vsx_match"] = bool(len(vsx))
        result["vsx_rows"] = vsx.astype(str).to_dict(orient="records")
    except Exception as error:
        result["vsx_match"] = None
        result["vsx_rows"] = []
        result["errors"].append(f"VSX: {type(error).__name__}: {error}")

    try:
        toi = _nasa_toi(tic_id)
        result["toi_match"] = bool(toi)
        result["toi_rows"] = toi
    except Exception as error:
        result["toi_match"] = None
        result["toi_rows"] = []
        result["errors"].append(f"NASA TOI: {type(error).__name__}: {error}")

    result["gaia_eb_match"] = None
    result["gaia_variable_match"] = None
    if gaia_id and gaia_id not in {"--", "nan", "None"}:
        try:
            with _network_timeout():
                table = Gaia.launch_job_async(
                    f"SELECT * FROM gaiadr3.vari_eclipsing_binary "
                    f"WHERE source_id={int(gaia_id)}"
                ).get_results()
            result["gaia_eb_match"] = bool(len(table))
        except Exception as error:
            result["errors"].append(f"Gaia EB: {type(error).__name__}: {error}")
        try:
            with _network_timeout():
                table = Gaia.launch_job_async(
                    f"SELECT * FROM gaiadr3.vari_classifier_result "
                    f"WHERE source_id={int(gaia_id)}"
                ).get_results()
            result["gaia_variable_match"] = bool(len(table))
        except Exception as error:
            result["errors"].append(f"Gaia variable: {type(error).__name__}: {error}")

    try:
        query = f"""
        SELECT TOP 100 source_id, phot_g_mean_mag,
               DISTANCE(POINT('ICRS',ra,dec),
                        POINT('ICRS',{ra},{dec}))*3600 AS sep_arcsec
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(POINT('ICRS',ra,dec),
                         CIRCLE('ICRS',{ra},{dec},{60/3600}))
        ORDER BY sep_arcsec
        """
        with _network_timeout():
            nearby = Gaia.launch_job_async(query).get_results().to_pandas()
        nearest_g = float(nearby.iloc[0]["phot_g_mean_mag"]) if len(nearby) else np.nan
        neighbors = nearby.iloc[1:].copy() if len(nearby) > 1 else nearby.iloc[0:0]
        result["gaia_sources_21arcsec"] = int(
            np.sum(nearby["sep_arcsec"] <= 21.0)
        )
        result["bright_neighbors_42arcsec"] = int(
            np.sum(
                (neighbors["sep_arcsec"] <= 42.0)
                & (neighbors["phot_g_mean_mag"] <= nearest_g + 3.0)
            )
        )
        result["nearest_neighbor_sep_arcsec"] = (
            float(neighbors.iloc[0]["sep_arcsec"]) if len(neighbors) else None
        )
        result["nearest_neighbor_delta_g"] = (
            float(neighbors.iloc[0]["phot_g_mean_mag"] - nearest_g)
            if len(neighbors)
            else None
        )
    except Exception as error:
        result["errors"].append(f"Gaia neighbors: {type(error).__name__}: {error}")

    try:
        with _network_timeout():
            vizier = Vizier(columns=["*"], row_limit=-1)
            catalog_flags = {}
            for label, catalog in {
                "tess_ten_new": "J/ApJS/279/50/table3",
                "tess_ten_known": "J/ApJS/279/50/table4",
                "green_ellipsoidal_selected": "J/MNRAS/522/29/table3",
            }.items():
                match = vizier.query_region(
                    coordinate, radius=2 * u.arcsec, catalog=catalog
                )
                catalog_flags[label] = bool(len(match) and len(match[0]))
        result.update(catalog_flags)
    except Exception as error:
        result["errors"].append(f"VizieR specialized: {type(error).__name__}: {error}")

    known_fields = [
        "villanova_known",
        "simbad_variable_match",
        "vsx_match",
        "toi_match",
        "gaia_eb_match",
        "gaia_variable_match",
        "tess_ten_new",
        "tess_ten_known",
        "green_ellipsoidal_selected",
    ]
    result["known_or_classified"] = any(result.get(field) is True for field in known_fields)
    return result


def save_catalog_vet(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
