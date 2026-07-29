import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from scipy.optimize import least_squares


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TIC_ID = 31065777
RA = 135.070041925556
DEC = -45.0522181112265

# Validated defaults. When the joint-fit summary exists, the values below are
# replaced with the newly regenerated joint-fit parameters before validation.
INITIAL_EPOCH = 1527.5353330962
INITIAL_PERIOD = 40.5717218077
FIXED_DURATION_DAYS = 2.9198 / 24.0
FIXED_INGRESS_RATIO = 0.3774

EVENTS = [
    {"sector": 8, "cycle": 0},
    {"sector": 9, "cycle": 1},
    {"sector": 35, "cycle": 18},
    {"sector": 36, "cycle": 19},
    {"sector": 89, "cycle": 54},
]

APERTURE_RADII = [1.0, 1.5, 2.0, 2.5]
CONTROL_OFFSETS = [(-4.0, -4.0), (4.0, -4.0), (-4.0, 4.0), (4.0, 4.0)]
TEMPORAL_OFFSETS_DAYS = [-2.0, -1.0, 1.0, 2.0]
BOOTSTRAP_TRIALS = 5000
RANDOM_SEED = 31065777

WINDOW_HALF_WIDTH = 0.45
REFERENCE_INNER = 0.20
REFERENCE_OUTER = 0.42

SUMMARY_PATH = RESULTS_DIR / "TIC_31065777_robust_validation_summary.txt"
EVENT_PATH = RESULTS_DIR / "TIC_31065777_robust_event_fits.csv"
APERTURE_PATH = RESULTS_DIR / "TIC_31065777_aperture_robustness.csv"
LEAVE_ONE_OUT_PATH = RESULTS_DIR / "TIC_31065777_leave_one_out.csv"
BOOTSTRAP_PATH = RESULTS_DIR / "TIC_31065777_period_bootstrap.csv"
CONTROL_PATH = RESULTS_DIR / "TIC_31065777_blank_apertures.csv"
TEMPORAL_PATH = RESULTS_DIR / "TIC_31065777_temporal_controls.csv"
JOINT_SUMMARY_PATH = RESULTS_DIR / "TIC_31065777_joint_raw_pixel_fit_summary.txt"


def load_joint_parameters():
    global INITIAL_EPOCH, INITIAL_PERIOD, FIXED_DURATION_DAYS, FIXED_INGRESS_RATIO
    if not JOINT_SUMMARY_PATH.exists():
        return
    values = {}
    for line in JOINT_SUMMARY_PATH.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    INITIAL_EPOCH = float(values.get("Epoch", INITIAL_EPOCH))
    INITIAL_PERIOD = float(values.get("Period days", INITIAL_PERIOD))
    duration_hours = float(
        values.get("Duration hours", FIXED_DURATION_DAYS * 24.0)
    )
    FIXED_DURATION_DAYS = duration_hours / 24.0
    FIXED_INGRESS_RATIO = float(
        values.get("Ingress ratio", FIXED_INGRESS_RATIO)
    )


load_joint_parameters()


def as_array(values, dtype=float):
    return np.asarray(np.ma.filled(values, np.nan), dtype=dtype)


def robust_sigma(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return np.nan
    median = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - median)))


def circular_aperture(shape, center_x, center_y, radius):
    yy, xx = np.indices(shape)
    return (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2


def trapezoid_shape(time_values, center):
    half_duration = FIXED_DURATION_DAYS / 2.0
    ingress_duration = FIXED_INGRESS_RATIO * FIXED_DURATION_DAYS
    flat_half_width = max(0.0, half_duration - ingress_duration)
    distance = np.abs(time_values - center)
    shape = np.zeros_like(time_values, dtype=float)
    shape[distance <= flat_half_width] = 1.0
    ingress = (distance > flat_half_width) & (distance < half_duration)
    if ingress_duration > 0:
        shape[ingress] = (half_duration - distance[ingress]) / ingress_duration
    return np.clip(shape, 0.0, 1.0)


def fits_path(sector):
    paths = [
        DATA_DIR / f"TIC_{TIC_ID}_sector_{sector}_repeat_check.fits",
        DATA_DIR / f"TIC_{TIC_ID}_sector_{sector}_tesscut.fits",
    ]
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"No saved TESSCut FITS file for Sector {sector}")


def load_sector(sector):
    path = fits_path(sector)
    with fits.open(path) as hdul:
        table = hdul[1].data
        time_values = as_array(table["TIME"])
        pixel_cube = as_array(table["FLUX"])
        if "QUALITY" in table.columns.names:
            quality = np.nan_to_num(as_array(table["QUALITY"]), nan=1).astype(np.int64)
        else:
            quality = np.zeros(len(time_values), dtype=np.int64)
        wcs = WCS(hdul[2].header).celestial
        target_x, target_y = wcs.world_to_pixel(SkyCoord(RA * u.deg, DEC * u.deg))

    usable = np.isfinite(time_values) & np.any(np.isfinite(pixel_cube), axis=(1, 2))
    clean = usable & (quality == 0)
    ny, nx = pixel_cube.shape[1:]
    border = np.zeros((ny, nx), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    background = np.nanmedian(pixel_cube[:, border], axis=1)
    corrected_cube = pixel_cube - background[:, None, None]
    finite_clean_time = time_values[clean]
    cadence = float(np.nanmedian(np.diff(np.sort(finite_clean_time))))
    return {
        "sector": sector,
        "path": str(path),
        "time": time_values,
        "cube": corrected_cube,
        "clean": clean,
        "shape": (ny, nx),
        "target_x": float(target_x),
        "target_y": float(target_y),
        "cadence_days": cadence,
    }


def extract_flux(sector_data, center_x, center_y, radius):
    aperture = circular_aperture(sector_data["shape"], center_x, center_y, radius)
    return np.nansum(sector_data["cube"][:, aperture], axis=1)


def fit_event(sector_data, raw_flux, expected_time):
    time_values = sector_data["time"]
    local = sector_data["clean"] & (np.abs(time_values - expected_time) <= WINDOW_HALF_WIDTH)
    time_local = time_values[local]
    flux_local = raw_flux[local]
    if len(time_local) < 20:
        return None

    reference = (
        (np.abs(time_local - expected_time) >= REFERENCE_INNER)
        & (np.abs(time_local - expected_time) <= REFERENCE_OUTER)
    )
    if np.sum(reference) < 8:
        return None
    raw_baseline = float(np.nanmedian(flux_local[reference]))
    if not np.isfinite(raw_baseline) or raw_baseline <= 0:
        return None
    normalized = flux_local / raw_baseline
    noise = robust_sigma(normalized[reference])
    if not np.isfinite(noise) or noise <= 0:
        noise = float(np.nanstd(normalized[reference]))
    if not np.isfinite(noise) or noise <= 0:
        return None

    near = np.abs(time_local - expected_time) <= 0.08
    depth_guess = float(np.clip(1.0 - np.nanmin(normalized[near]), 0.005, 0.12))

    def residuals(parameters):
        center, depth, baseline, slope = parameters
        shape = trapezoid_shape(time_local, center)
        model = baseline + slope * (time_local - expected_time) - depth * shape
        return (normalized - model) / noise

    fit = least_squares(
        residuals,
        x0=[expected_time, depth_guess, 1.0, 0.0],
        bounds=(
            [expected_time - 0.12, 0.001, 0.90, -0.15],
            [expected_time + 0.12, 0.15, 1.10, 0.15],
        ),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=10000,
    )

    parameter_errors = np.full(4, np.nan)
    reduced_chi_square = np.nan
    try:
        raw_residuals = residuals(fit.x)
        degrees_of_freedom = max(1, len(raw_residuals) - len(fit.x))
        reduced_chi_square = float(np.sum(raw_residuals ** 2) / degrees_of_freedom)
        covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * reduced_chi_square
        parameter_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    except Exception:
        pass

    cadence_floor = sector_data["cadence_days"] / 2.0
    formal_center_error = float(parameter_errors[0])
    if not np.isfinite(formal_center_error) or formal_center_error <= 0:
        formal_center_error = cadence_floor
    conservative_center_error = max(formal_center_error, cadence_floor)

    return {
        "center": float(fit.x[0]),
        "center_error_days": conservative_center_error,
        "formal_center_error_days": formal_center_error,
        "depth_fraction": float(fit.x[1]),
        "depth_percent": float(100 * fit.x[1]),
        "depth_error_percent": float(100 * parameter_errors[1]),
        "baseline": float(fit.x[2]),
        "slope_per_day": float(fit.x[3]),
        "raw_baseline": raw_baseline,
        "noise_percent": float(100 * noise),
        "cadence_minutes": float(sector_data["cadence_days"] * 24 * 60),
        "points": len(time_local),
        "reduced_chi_square": reduced_chi_square,
        "fit_success": bool(fit.success),
    }


def fit_ephemeris(cycles, centers, center_errors):
    cycles = np.asarray(cycles, dtype=float)
    centers = np.asarray(centers, dtype=float)
    center_errors = np.asarray(center_errors, dtype=float)
    weights = 1.0 / np.square(center_errors)
    design = np.column_stack([np.ones(len(cycles)), cycles])
    normal = design.T @ (weights[:, None] * design)
    right = design.T @ (weights * centers)
    solution = np.linalg.solve(normal, right)
    covariance = np.linalg.inv(normal)
    epoch, period = solution
    model = epoch + cycles * period
    residuals = centers - model
    return {
        "epoch": float(epoch),
        "period": float(period),
        "epoch_error": float(np.sqrt(covariance[0, 0])),
        "period_error": float(np.sqrt(covariance[1, 1])),
        "residuals": residuals,
        "max_residual_minutes": float(np.max(np.abs(residuals)) * 24 * 60),
    }


def measure_fixed_window(sector_data, raw_flux, center, target_baseline=None):
    time_values = sector_data["time"]
    event = sector_data["clean"] & (np.abs(time_values - center) <= FIXED_DURATION_DAYS / 2)
    reference = sector_data["clean"] & (
        (np.abs(time_values - center) >= REFERENCE_INNER)
        & (np.abs(time_values - center) <= REFERENCE_OUTER)
    )
    if np.sum(event) < 3 or np.sum(reference) < 8:
        return None
    baseline = float(np.nanmedian(raw_flux[reference]))
    event_flux = float(np.nanmedian(raw_flux[event]))
    scale = baseline if target_baseline is None else target_baseline
    if not np.isfinite(scale) or scale <= 0:
        return None
    return {
        "depth_target_units_percent": float(100 * (baseline - event_flux) / scale),
        "event_points": int(np.sum(event)),
        "reference_points": int(np.sum(reference)),
    }


print("=" * 78)
print("TIC 31065777 CONSERVATIVE ROBUSTNESS VALIDATION")
print("=" * 78)

sector_cache = {}
for event in EVENTS:
    print(f"Loading Sector {event['sector']}...")
    sector_cache[event["sector"]] = load_sector(event["sector"])


print("\n" + "=" * 78)
print("APERTURE-SIZE ROBUSTNESS")
print("=" * 78)

all_event_rows = []
aperture_rows = []
measurements_by_radius = {}

for radius in APERTURE_RADII:
    rows = []
    for event in EVENTS:
        sector_data = sector_cache[event["sector"]]
        expected_time = INITIAL_EPOCH + event["cycle"] * INITIAL_PERIOD
        raw_flux = extract_flux(
            sector_data,
            sector_data["target_x"],
            sector_data["target_y"],
            radius,
        )
        result = fit_event(sector_data, raw_flux, expected_time)
        if result is None:
            continue
        row = {
            "radius_pixels": radius,
            "sector": event["sector"],
            "cycle": event["cycle"],
            "expected_time": expected_time,
            **result,
        }
        rows.append(row)
        all_event_rows.append(row)

    frame = pd.DataFrame(rows)
    measurements_by_radius[radius] = frame
    if len(frame) < 3:
        continue
    fit = fit_ephemeris(frame["cycle"], frame["center"], frame["center_error_days"])
    aperture_rows.append({
        "radius_pixels": radius,
        "events_measured": len(frame),
        "epoch": fit["epoch"],
        "period_days": fit["period"],
        "period_difference_seconds_from_initial": (fit["period"] - INITIAL_PERIOD) * 86400,
        "formal_period_error_seconds": fit["period_error"] * 86400,
        "max_timing_residual_minutes": fit["max_residual_minutes"],
        "median_depth_percent": float(np.nanmedian(frame["depth_percent"])),
        "depth_spread_percent": float(np.nanstd(frame["depth_percent"])),
    })

event_table = pd.DataFrame(all_event_rows)
event_table.to_csv(EVENT_PATH, index=False)
aperture_table = pd.DataFrame(aperture_rows)
aperture_table.to_csv(APERTURE_PATH, index=False)
print(aperture_table.to_string(index=False))


standard = measurements_by_radius[1.5].copy()
if len(standard) != len(EVENTS):
    raise RuntimeError("The standard 1.5-pixel aperture did not measure all five events")
standard_fit = fit_ephemeris(
    standard["cycle"], standard["center"], standard["center_error_days"]
)


print("\n" + "=" * 78)
print("STANDARD FIVE EVENT FITS")
print("=" * 78)
print(
    standard[
        [
            "sector", "cycle", "center", "center_error_days", "cadence_minutes",
            "depth_percent", "depth_error_percent", "noise_percent", "reduced_chi_square",
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 78)
print("LEAVE-ONE-SECTOR-OUT TEST")
print("=" * 78)

leave_rows = []
for excluded_sector in standard["sector"]:
    subset = standard[standard["sector"] != excluded_sector]
    fit = fit_ephemeris(subset["cycle"], subset["center"], subset["center_error_days"])
    excluded = standard[standard["sector"] == excluded_sector].iloc[0]
    predicted = fit["epoch"] + excluded["cycle"] * fit["period"]
    leave_rows.append({
        "excluded_sector": int(excluded_sector),
        "period_days": fit["period"],
        "period_difference_seconds": (fit["period"] - standard_fit["period"]) * 86400,
        "excluded_event_prediction_error_minutes": (excluded["center"] - predicted) * 24 * 60,
    })

leave_table = pd.DataFrame(leave_rows)
leave_table.to_csv(LEAVE_ONE_OUT_PATH, index=False)
print(leave_table.to_string(index=False))


print("\n" + "=" * 78)
print("PARAMETRIC BOOTSTRAP PERIOD TEST")
print("=" * 78)

rng = np.random.default_rng(RANDOM_SEED)
cycles = standard["cycle"].to_numpy(dtype=float)
centers = standard["center"].to_numpy(dtype=float)
center_errors = standard["center_error_days"].to_numpy(dtype=float)
bootstrap_periods = []
for _ in range(BOOTSTRAP_TRIALS):
    synthetic = centers + rng.normal(0.0, center_errors)
    fit = fit_ephemeris(cycles, synthetic, center_errors)
    bootstrap_periods.append(fit["period"])

bootstrap_periods = np.asarray(bootstrap_periods)
pd.DataFrame({"period_days": bootstrap_periods}).to_csv(BOOTSTRAP_PATH, index=False)
period_low, period_median, period_high = np.nanpercentile(bootstrap_periods, [2.5, 50, 97.5])
print(f"Standard period: {standard_fit['period']:.10f} days")
print(f"Bootstrap median: {period_median:.10f} days")
print(f"95% interval: {period_low:.10f} to {period_high:.10f} days")
print(f"Bootstrap 1-sigma scatter: {np.nanstd(bootstrap_periods) * 86400:.3f} seconds")


print("\n" + "=" * 78)
print("BLANK-APERTURE SPATIAL CONTROLS")
print("=" * 78)

control_rows = []
for offset_x, offset_y in CONTROL_OFFSETS:
    values = []
    for event in EVENTS:
        sector_data = sector_cache[event["sector"]]
        x = sector_data["target_x"] + offset_x
        y = sector_data["target_y"] + offset_y
        ny, nx = sector_data["shape"]
        if not (1.5 <= x <= nx - 2.5 and 1.5 <= y <= ny - 2.5):
            continue
        expected_time = standard.loc[standard["sector"] == event["sector"], "center"].iloc[0]
        control_flux = extract_flux(sector_data, x, y, 1.5)
        target_baseline = standard.loc[standard["sector"] == event["sector"], "raw_baseline"].iloc[0]
        result = measure_fixed_window(sector_data, control_flux, expected_time, target_baseline)
        if result is None:
            continue
        values.append(result["depth_target_units_percent"])
        control_rows.append({
            "offset_x": offset_x,
            "offset_y": offset_y,
            "sector": event["sector"],
            **result,
        })
    print(
        f"Offset ({offset_x:+.1f}, {offset_y:+.1f}) px: "
        f"median={np.nanmedian(values):.4f}% of target baseline, "
        f"max_abs={np.nanmax(np.abs(values)):.4f}%"
    )

control_table = pd.DataFrame(control_rows)
control_table.to_csv(CONTROL_PATH, index=False)


print("\n" + "=" * 78)
print("TARGET TEMPORAL CONTROLS")
print("=" * 78)

temporal_rows = []
for event in EVENTS:
    sector_data = sector_cache[event["sector"]]
    raw_flux = extract_flux(
        sector_data, sector_data["target_x"], sector_data["target_y"], 1.5
    )
    event_center = standard.loc[standard["sector"] == event["sector"], "center"].iloc[0]
    for offset in TEMPORAL_OFFSETS_DAYS:
        result = measure_fixed_window(sector_data, raw_flux, event_center + offset)
        if result is None:
            continue
        temporal_rows.append({
            "sector": event["sector"],
            "offset_days": offset,
            **result,
        })

temporal_table = pd.DataFrame(temporal_rows)
temporal_table.to_csv(TEMPORAL_PATH, index=False)
if len(temporal_table):
    print(
        temporal_table.groupby("offset_days")["depth_target_units_percent"]
        .agg(["count", "median", "min", "max"])
        .to_string()
    )


print("\n" + "=" * 78)
print("FINAL ROBUSTNESS ASSESSMENT")
print("=" * 78)

period_range_seconds = float(
    (aperture_table["period_days"].max() - aperture_table["period_days"].min()) * 86400
)
maximum_leave_out_error = float(
    np.nanmax(np.abs(leave_table["excluded_event_prediction_error_minutes"]))
)
target_median_depth = float(np.nanmedian(standard["depth_percent"]))
control_medians = (
    control_table.groupby(["offset_x", "offset_y"])["depth_target_units_percent"].median()
)
largest_control_median = float(np.nanmax(np.abs(control_medians)))
largest_temporal_depth = (
    float(np.nanmax(temporal_table["depth_target_units_percent"]))
    if len(temporal_table)
    else np.nan
)
bootstrap_width_days = float(period_high - period_low)

aperture_stable = period_range_seconds <= 300
leave_one_out_stable = maximum_leave_out_error <= 120
controls_clear = largest_control_median < 0.5 * target_median_depth
temporal_controls_clear = (
    not np.isfinite(largest_temporal_depth)
    or largest_temporal_depth < 0.5 * target_median_depth
)
bootstrap_narrow = bootstrap_width_days <= 0.01
all_pass = (
    aperture_stable
    and leave_one_out_stable
    and controls_clear
    and temporal_controls_clear
    and bootstrap_narrow
)

print(f"Conservative epoch: {standard_fit['epoch']:.10f} BTJD")
print(f"Conservative period: {standard_fit['period']:.10f} days")
print(f"Target median fitted depth: {target_median_depth:.4f}%")
print(f"Period range across apertures: {period_range_seconds:.3f} seconds")
print(f"Largest leave-one-out prediction error: {maximum_leave_out_error:.3f} minutes")
print(f"Bootstrap 95% width: {bootstrap_width_days:.8f} days")
print(f"Largest blank-control median: {largest_control_median:.4f}% of target baseline")
print(f"Largest temporal-control depth: {largest_temporal_depth:.4f}%")
print(f"Aperture period stable: {aperture_stable}")
print(f"Leave-one-out stable: {leave_one_out_stable}")
print(f"Bootstrap interval narrow: {bootstrap_narrow}")
print(f"Blank spatial controls clear: {controls_clear}")
print(f"Target temporal controls clear: {temporal_controls_clear}")
print(f"ROBUST VALIDATION PASSES: {all_pass}")

with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
    file.write("TIC 31065777 conservative robustness validation\n")
    file.write("=" * 62 + "\n")
    file.write(f"Epoch BTJD: {standard_fit['epoch']:.10f}\n")
    file.write(f"Period days: {standard_fit['period']:.10f}\n")
    file.write(f"Bootstrap 95% low days: {period_low:.10f}\n")
    file.write(f"Bootstrap 95% high days: {period_high:.10f}\n")
    file.write(f"Median fitted depth percent: {target_median_depth:.6f}\n")
    file.write(f"Period range across apertures seconds: {period_range_seconds:.6f}\n")
    file.write(f"Maximum leave-one-out error minutes: {maximum_leave_out_error:.6f}\n")
    file.write(f"Largest blank-control median percent: {largest_control_median:.6f}\n")
    file.write(f"Largest temporal-control depth percent: {largest_temporal_depth:.6f}\n")
    file.write(f"Aperture stable: {aperture_stable}\n")
    file.write(f"Leave-one-out stable: {leave_one_out_stable}\n")
    file.write(f"Bootstrap narrow: {bootstrap_narrow}\n")
    file.write(f"Spatial controls clear: {controls_clear}\n")
    file.write(f"Temporal controls clear: {temporal_controls_clear}\n")
    file.write(f"Robust validation passes: {all_pass}\n")

print("\nSaved:")
for path in [SUMMARY_PATH, EVENT_PATH, APERTURE_PATH, LEAVE_ONE_OUT_PATH, BOOTSTRAP_PATH, CONTROL_PATH, TEMPORAL_PATH]:
    print(path)
