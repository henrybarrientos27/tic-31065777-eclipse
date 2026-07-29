"""Campaign-scale OMEGA-X falsification tests.

Every jointly modeled survivor is tested with the same public-data orbital-clock
analysis. Failures remain in the table so a campaign never silently drops an
inconvenient result.
"""

from __future__ import annotations

import json

import pandas as pd

from .pipeline import OmegaPaths
from .timing_physics import run_clock_falsification


def run_physics_campaign(
    paths: OmegaPaths,
    trials_scanned: int | None = None,
) -> pd.DataFrame:
    vet_root = paths.result_root / "deep_vetting"
    vet_root.mkdir(parents=True, exist_ok=True)
    model_paths = sorted(vet_root.glob("TIC_*_joint_raw_model.json"))

    leaderboard_path = paths.result_root / "omega_v2_leaderboard.csv"
    if trials_scanned is None or trials_scanned < 1:
        if leaderboard_path.exists():
            trials_scanned = max(1, len(pd.read_csv(leaderboard_path)))
        else:
            trials_scanned = max(1, len(model_paths))

    rows = []
    for model_path in model_paths:
        tic_id = int(model_path.name.split("_")[1])
        catalog_path = vet_root / f"TIC_{tic_id}_catalog_vet.json"
        pixel_path = vet_root / f"TIC_{tic_id}_pixel_vet.csv"
        repeat_path = paths.repeats / f"TIC_{tic_id}_repeat.json"
        required = [catalog_path, pixel_path, repeat_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            rows.append(
                {
                    "tic_id": tic_id,
                    "status": "missing_prerequisites",
                    "classification": "not_tested",
                    "revolutionary_claim_ready": False,
                    "error": "; ".join(missing),
                }
            )
            continue

        try:
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            model = json.loads(model_path.read_text(encoding="utf-8"))
            repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
            pixels = pd.read_csv(pixel_path)
            ztf_path = vet_root / f"TIC_{tic_id}_ztf_phase_data.csv"
            result = run_clock_falsification(
                tic_id,
                ra=float(catalog["ra"]),
                dec=float(catalog["dec"]),
                repeat=repeat,
                pixel_table=pixels,
                joint_model=model,
                ztf_phase_path=ztf_path if ztf_path.exists() else None,
                result_dir=vet_root,
                plot_dir=paths.plot_root / "physics_clock",
                trials_scanned=int(trials_scanned),
            )
            rows.append(
                {
                    "tic_id": tic_id,
                    "status": "tested",
                    "classification": result["classification"],
                    "revolutionary_claim_ready": bool(
                        result["revolutionary_claim_ready"]
                    ),
                    "measurements": result["measurements"],
                    "independent_sources": result["independent_sources"],
                    "tess_sectors": result["tess_sectors"],
                    "baseline_days": result["time_baseline_days"],
                    "half_period_alias_corrected": result[
                        "half_period_alias_corrected"
                    ],
                    "naive_trial_corrected_p": result[
                        "quadratic_trial_corrected_p"
                    ],
                    "source_offset_trial_corrected_p": result[
                        "source_offset_clock_test"
                    ]["quadratic_trial_corrected_p"],
                    "worst_leave_one_out_p": result[
                        "leave_one_out_robustness"
                    ]["worst_trial_corrected_p"],
                    "result_file": result["result_file"],
                    "plot_file": result["plot_file"],
                }
            )
        except Exception as error:
            rows.append(
                {
                    "tic_id": tic_id,
                    "status": "test_failed",
                    "classification": "not_tested",
                    "revolutionary_claim_ready": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    table = pd.DataFrame(rows)
    csv_path = paths.result_root / "omega_x_physics_campaign.csv"
    json_path = paths.result_root / "omega_x_physics_campaign.json"
    table.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8"
    )
    return table
