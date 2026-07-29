"""Analysis-choice robustness checks for raw-pixel eclipse models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .modeling import joint_raw_model


def robust_raw_model_checks(
    tic_id: int,
    ra: float,
    dec: float,
    repeat: dict,
    pixel_table: pd.DataFrame,
    stellar_radius_rsun: float | None,
    result_dir: Path,
    plot_dir: Path,
    aperture_radii: tuple[float, ...] = (1.0, 1.5, 2.0),
) -> dict:
    """Refit across apertures and with each raw-pixel sector removed."""
    robust_dir = result_dir / "robustness"
    robust_plot_dir = plot_dir / "robustness"
    robust_dir.mkdir(parents=True, exist_ok=True)
    robust_plot_dir.mkdir(parents=True, exist_ok=True)

    aperture_results = []
    for radius in aperture_radii:
        tag = f"aperture_{str(radius).replace('.', 'p')}px"
        model = joint_raw_model(
            tic_id,
            ra=ra,
            dec=dec,
            repeat=repeat,
            pixel_table=pixel_table,
            stellar_radius_rsun=stellar_radius_rsun,
            result_dir=robust_dir,
            plot_dir=robust_plot_dir,
            aperture_radius=radius,
            output_tag=tag,
        )
        aperture_results.append(model)

    passed_sectors = sorted(
        int(value)
        for value in pixel_table.loc[
            pixel_table["pixel_localization_pass"].astype(bool), "sector"
        ].unique()
    )
    leave_one_out_results = []
    if len(passed_sectors) >= 3:
        for sector in passed_sectors:
            tag = f"without_sector_{sector}"
            model = joint_raw_model(
                tic_id,
                ra=ra,
                dec=dec,
                repeat=repeat,
                pixel_table=pixel_table,
                stellar_radius_rsun=stellar_radius_rsun,
                result_dir=robust_dir,
                plot_dir=robust_plot_dir,
                aperture_radius=1.5,
                excluded_sectors={sector},
                output_tag=tag,
            )
            leave_one_out_results.append(model)

    aperture_table = pd.DataFrame(aperture_results)
    leave_table = pd.DataFrame(leave_one_out_results)
    aperture_path = robust_dir / f"TIC_{tic_id}_aperture_models.csv"
    leave_path = robust_dir / f"TIC_{tic_id}_leave_sector_out_models.csv"
    aperture_table.to_csv(aperture_path, index=False)
    leave_table.to_csv(leave_path, index=False)

    aperture_period_range_seconds = float(
        (aperture_table["period_days"].max() - aperture_table["period_days"].min())
        * 86400.0
    )
    aperture_depth_median = float(
        np.nanmedian(aperture_table["median_depth_percent"])
    )
    aperture_depth_range_fraction = float(
        (
            aperture_table["median_depth_percent"].max()
            - aperture_table["median_depth_percent"].min()
        )
        / max(aperture_depth_median, 1e-6)
    )
    leave_period_range_seconds = (
        float(
            (leave_table["period_days"].max() - leave_table["period_days"].min())
            * 86400.0
        )
        if len(leave_table)
        else np.nan
    )
    aperture_stable = bool(aperture_period_range_seconds <= 120.0)
    depth_stable = bool(aperture_depth_range_fraction <= 0.50)
    leave_one_out_stable = bool(
        len(leave_table) == 0 or leave_period_range_seconds <= 240.0
    )
    all_fits_converged = bool(
        aperture_table["fit_success"].all()
        and (len(leave_table) == 0 or leave_table["fit_success"].all())
    )
    passed = bool(
        aperture_stable and depth_stable and leave_one_out_stable and all_fits_converged
    )

    result = {
        "tic_id": int(tic_id),
        "robust_validation_passes": passed,
        "all_fits_converged": all_fits_converged,
        "aperture_period_stable": aperture_stable,
        "aperture_depth_stable": depth_stable,
        "leave_one_sector_out_stable": leave_one_out_stable,
        "leave_one_sector_out_available": bool(len(leave_table)),
        "aperture_period_range_seconds": aperture_period_range_seconds,
        "aperture_depth_range_fraction": aperture_depth_range_fraction,
        "leave_one_out_period_range_seconds": leave_period_range_seconds,
        "apertures_tested_pixels": [float(value) for value in aperture_radii],
        "sectors_removed_one_at_a_time": passed_sectors,
        "aperture_table": str(aperture_path),
        "leave_one_out_table": str(leave_path),
        "caution": (
            "This tests analysis-choice stability. Formal covariance errors still do "
            "not include every TESS systematic and should not be treated as final."
        ),
    }
    output_path = robust_dir / f"TIC_{tic_id}_robustness_summary.json"
    result["result_file"] = str(output_path)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
