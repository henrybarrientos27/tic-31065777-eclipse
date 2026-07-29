import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Tesscut

from scipy.optimize import least_squares


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TIC_ID = 31065777
RA = 135.070041925556
DEC = -45.0522181112265

INITIAL_EPOCH = 1527.5261576497
INITIAL_PERIOD = 40.5718968

EVENTS = [
    {"sector": 8,  "cycle": 0,  "time": 1527.5261576497},
    {"sector": 9,  "cycle": 1,  "time": 1568.0980544275},
    {"sector": 35, "cycle": 18, "time": 2257.8202996493},
    {"sector": 36, "cycle": 19, "time": 2298.3921964271},
    {"sector": 89, "cycle": 54, "time": 3718.4085836485},
]

WINDOW_HALF_WIDTH = 0.55
APERTURE_RADIUS = 1.5
CUTOUT_SIZE = 15

SUMMARY_PATH = (
    RESULTS_DIR
    / "TIC_31065777_joint_raw_pixel_fit_summary.txt"
)

EVENT_TABLE_PATH = (
    RESULTS_DIR
    / "TIC_31065777_joint_raw_pixel_events.csv"
)

LIGHTCURVE_TABLE_PATH = (
    RESULTS_DIR
    / "TIC_31065777_raw_pixel_lightcurves.csv"
)


def as_array(values, dtype=float):
    return np.asarray(
        np.ma.filled(values, np.nan),
        dtype=dtype,
    )


def robust_sigma(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 5:
        return np.nan

    median = np.nanmedian(values)
    mad = np.nanmedian(
        np.abs(values - median)
    )

    return float(1.4826 * mad)


def circular_aperture(shape, center_x, center_y, radius):
    yy, xx = np.indices(shape)

    return (
        (xx - center_x) ** 2
        + (yy - center_y) ** 2
        <= radius ** 2
    )


def locate_or_download_cutout(sector):
    candidates = [
        DATA_DIR
        / f"TIC_{TIC_ID}_sector_{sector}_repeat_check.fits",

        DATA_DIR
        / f"TIC_{TIC_ID}_sector_{sector}_tesscut.fits",
    ]

    for path in candidates:
        if path.exists():
            return path

    print(
        f"Sector {sector}: no saved FITS file found; "
        "downloading through TESSCut..."
    )

    coordinate = SkyCoord(
        RA * u.deg,
        DEC * u.deg,
    )

    cutouts = Tesscut.get_cutouts(
        coordinates=coordinate,
        size=CUTOUT_SIZE,
        sector=sector,
    )

    if len(cutouts) == 0:
        raise RuntimeError(
            f"No TESSCut data returned for Sector {sector}."
        )

    output_path = candidates[0]

    cutouts[0].writeto(
        output_path,
        overwrite=True,
    )

    return output_path


def extract_target_lightcurve(sector, expected_time):
    path = locate_or_download_cutout(sector)

    with fits.open(path) as hdul:
        table = hdul[1].data

        time_values = as_array(
            table["TIME"]
        )

        pixel_cube = as_array(
            table["FLUX"]
        )

        if "QUALITY" in table.columns.names:
            quality = as_array(
                table["QUALITY"]
            )

            quality = np.nan_to_num(
                quality,
                nan=1,
            ).astype(np.int64)
        else:
            quality = np.zeros(
                len(time_values),
                dtype=np.int64,
            )

        wcs = WCS(
            hdul[2].header
        ).celestial

        target_coord = SkyCoord(
            RA * u.deg,
            DEC * u.deg,
        )

        target_x, target_y = (
            wcs.world_to_pixel(
                target_coord
            )
        )

    target_x = float(target_x)
    target_y = float(target_y)

    finite_frame = np.any(
        np.isfinite(pixel_cube),
        axis=(1, 2),
    )

    clean = (
        np.isfinite(time_values)
        & finite_frame
        & (quality == 0)
    )

    local = (
        clean
        & (
            np.abs(
                time_values - expected_time
            )
            <= WINDOW_HALF_WIDTH
        )
    )

    if np.sum(local) < 25:
        raise RuntimeError(
            f"Sector {sector} has too few clean points."
        )

    ny, nx = pixel_cube.shape[1:]

    border = np.zeros(
        (ny, nx),
        dtype=bool,
    )

    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    background = np.nanmedian(
        pixel_cube[:, border],
        axis=1,
    )

    corrected_cube = (
        pixel_cube
        - background[:, None, None]
    )

    aperture = circular_aperture(
        (ny, nx),
        target_x,
        target_y,
        APERTURE_RADIUS,
    )

    aperture_flux = np.nansum(
        corrected_cube[:, aperture],
        axis=1,
    )

    local_time = time_values[local]
    local_flux = aperture_flux[local]

    broad_event = (
        np.abs(
            local_time - expected_time
        )
        <= 0.14
    )

    out_of_event = ~broad_event

    baseline = float(
        np.nanmedian(
            local_flux[out_of_event]
        )
    )

    if not np.isfinite(baseline) or baseline == 0:
        raise RuntimeError(
            f"Sector {sector} baseline is invalid."
        )

    normalized_flux = (
        local_flux / baseline
    )

    noise = robust_sigma(
        normalized_flux[out_of_event]
    )

    if not np.isfinite(noise) or noise <= 0:
        noise = float(
            np.nanstd(
                normalized_flux[out_of_event]
            )
        )

    if not np.isfinite(noise) or noise <= 0:
        raise RuntimeError(
            f"Sector {sector} noise estimate failed."
        )

    order = np.argsort(local_time)

    return {
        "sector": sector,
        "time": local_time[order],
        "flux": normalized_flux[order],
        "noise": noise,
        "target_x": target_x,
        "target_y": target_y,
        "path": str(path),
    }


def trapezoid_shape(time_values, center, duration, ingress_ratio):
    half_duration = duration / 2.0

    ingress_duration = (
        ingress_ratio * duration
    )

    flat_half_width = max(
        0.0,
        half_duration - ingress_duration,
    )

    distance = np.abs(
        time_values - center
    )

    shape = np.zeros_like(
        time_values,
        dtype=float,
    )

    flat = (
        distance <= flat_half_width
    )

    ingress = (
        (distance > flat_half_width)
        & (distance < half_duration)
    )

    shape[flat] = 1.0

    if ingress_duration > 0:
        shape[ingress] = (
            half_duration
            - distance[ingress]
        ) / ingress_duration

    return np.clip(
        shape,
        0.0,
        1.0,
    )


print("=" * 76)
print("TIC 31065777 JOINT RAW-PIXEL ECLIPSE FIT")
print("=" * 76)

datasets = []

for event in EVENTS:
    print(
        f"Loading Sector {event['sector']} "
        f"near time {event['time']:.8f}..."
    )

    dataset = extract_target_lightcurve(
        event["sector"],
        event["time"],
    )

    dataset["cycle"] = event["cycle"]
    dataset["expected_time"] = event["time"]

    datasets.append(dataset)

    print(
        f"  points={len(dataset['time'])} | "
        f"noise={100 * dataset['noise']:.4f}% | "
        f"target pixel=({dataset['target_x']:.3f}, "
        f"{dataset['target_y']:.3f})"
    )


number_of_events = len(datasets)

# Parameter layout:
# 0: epoch
# 1: period
# 2: duration
# 3: ingress ratio
# next N: event depths
# next N: baselines
# next N: local slopes

initial_depths = []

for dataset in datasets:
    near = (
        np.abs(
            dataset["time"]
            - dataset["expected_time"]
        )
        <= 0.08
    )

    depth_guess = float(
        1.0
        - np.nanmedian(
            dataset["flux"][near]
        )
    )

    initial_depths.append(
        float(
            np.clip(
                depth_guess,
                0.005,
                0.08,
            )
        )
    )


initial_parameters = np.concatenate([
    [
        INITIAL_EPOCH,
        INITIAL_PERIOD,
        0.125,
        0.20,
    ],
    np.asarray(initial_depths),
    np.ones(number_of_events),
    np.zeros(number_of_events),
])


lower_bounds = np.concatenate([
    [
        INITIAL_EPOCH - 0.20,
        40.45,
        0.035,
        0.03,
    ],
    np.full(number_of_events, 0.001),
    np.full(number_of_events, 0.94),
    np.full(number_of_events, -0.10),
])


upper_bounds = np.concatenate([
    [
        INITIAL_EPOCH + 0.20,
        40.70,
        0.35,
        0.49,
    ],
    np.full(number_of_events, 0.10),
    np.full(number_of_events, 1.06),
    np.full(number_of_events, 0.10),
])


def unpack(parameters):
    epoch = float(parameters[0])
    period = float(parameters[1])
    duration = float(parameters[2])
    ingress_ratio = float(parameters[3])

    index = 4

    depths = parameters[
        index:index + number_of_events
    ]

    index += number_of_events

    baselines = parameters[
        index:index + number_of_events
    ]

    index += number_of_events

    slopes = parameters[
        index:index + number_of_events
    ]

    return (
        epoch,
        period,
        duration,
        ingress_ratio,
        depths,
        baselines,
        slopes,
    )


def residuals(parameters):
    (
        epoch,
        period,
        duration,
        ingress_ratio,
        depths,
        baselines,
        slopes,
    ) = unpack(parameters)

    residual_blocks = []

    for index, dataset in enumerate(datasets):
        center = (
            epoch
            + dataset["cycle"] * period
        )

        shape = trapezoid_shape(
            dataset["time"],
            center,
            duration,
            ingress_ratio,
        )

        local_time = (
            dataset["time"] - center
        )

        model = (
            baselines[index]
            + slopes[index] * local_time
            - depths[index] * shape
        )

        residual_blocks.append(
            (
                dataset["flux"] - model
            )
            / dataset["noise"]
        )

    return np.concatenate(
        residual_blocks
    )


fit = least_squares(
    residuals,
    x0=initial_parameters,
    bounds=(
        lower_bounds,
        upper_bounds,
    ),
    loss="soft_l1",
    f_scale=1.0,
    x_scale="jac",
    max_nfev=30000,
)


(
    best_epoch,
    best_period,
    best_duration,
    best_ingress_ratio,
    best_depths,
    best_baselines,
    best_slopes,
) = unpack(fit.x)


residual_vector = residuals(
    fit.x
)

number_of_points = len(
    residual_vector
)

number_of_parameters = len(
    fit.x
)

degrees_of_freedom = max(
    1,
    number_of_points
    - number_of_parameters,
)

reduced_chi_square = float(
    np.sum(
        residual_vector ** 2
    )
    / degrees_of_freedom
)


# Approximate parameter uncertainties from the Jacobian.
parameter_errors = np.full(
    len(fit.x),
    np.nan,
)

try:
    jacobian = fit.jac

    covariance = np.linalg.inv(
        jacobian.T @ jacobian
    )

    covariance *= reduced_chi_square

    parameter_errors = np.sqrt(
        np.diag(covariance)
    )

except Exception:
    pass


epoch_error = float(
    parameter_errors[0]
)

period_error = float(
    parameter_errors[1]
)

duration_error = float(
    parameter_errors[2]
)

ingress_ratio_error = float(
    parameter_errors[3]
)


event_rows = []

for index, dataset in enumerate(datasets):
    model_center = (
        best_epoch
        + dataset["cycle"] * best_period
    )

    shape = trapezoid_shape(
        dataset["time"],
        model_center,
        best_duration,
        best_ingress_ratio,
    )

    model_flux = (
        best_baselines[index]
        + best_slopes[index]
        * (
            dataset["time"]
            - model_center
        )
        - best_depths[index]
        * shape
    )

    event_window = (
        shape > 0
    )

    event_residual_rms = float(
        np.sqrt(
            np.nanmean(
                (
                    dataset["flux"]
                    - model_flux
                ) ** 2
            )
        )
    )

    event_rows.append({
        "sector": dataset["sector"],
        "cycle": dataset["cycle"],
        "model_center": model_center,
        "depth_percent": (
            100 * best_depths[index]
        ),
        "depth_error_percent": (
            100
            * parameter_errors[
                4 + index
            ]
            if np.isfinite(
                parameter_errors[
                    4 + index
                ]
            )
            else np.nan
        ),
        "event_points": int(
            np.sum(event_window)
        ),
        "residual_rms_percent": (
            100 * event_residual_rms
        ),
        "local_noise_percent": (
            100 * dataset["noise"]
        ),
        "baseline": (
            best_baselines[index]
        ),
        "slope_per_day": (
            best_slopes[index]
        ),
    })


event_table = pd.DataFrame(
    event_rows
)

event_table.to_csv(
    EVENT_TABLE_PATH,
    index=False,
)

lightcurve_rows = []

for index, dataset in enumerate(datasets):
    center = best_epoch + dataset["cycle"] * best_period
    shape = trapezoid_shape(
        dataset["time"],
        center,
        best_duration,
        best_ingress_ratio,
    )
    model_flux = (
        best_baselines[index]
        + best_slopes[index] * (dataset["time"] - center)
        - best_depths[index] * shape
    )
    for time_value, flux_value, model_value in zip(
        dataset["time"], dataset["flux"], model_flux
    ):
        lightcurve_rows.append({
            "sector": dataset["sector"],
            "cycle": dataset["cycle"],
            "time_btjd": float(time_value),
            "hours_from_center": float(24.0 * (time_value - center)),
            "normalized_flux": float(flux_value),
            "model_flux": float(model_value),
        })

pd.DataFrame(lightcurve_rows).to_csv(
    LIGHTCURVE_TABLE_PATH,
    index=False,
)


mean_depth = float(
    np.mean(best_depths)
)

median_depth = float(
    np.median(best_depths)
)

depth_spread = float(
    np.std(best_depths)
)

radius_ratio = float(
    np.sqrt(
        max(
            median_depth,
            0.0,
        )
    )
)


print("\n" + "=" * 76)
print("JOINT-FIT ORBIT")
print("=" * 76)

print(
    "Fit converged:",
    fit.success,
)

print(
    "Fit message:",
    fit.message,
)

print(
    "Reference eclipse time:",
    f"{best_epoch:.10f}",
)

print(
    "Epoch uncertainty:",
    f"{epoch_error:.8f} days",
)

print(
    "Orbital period:",
    f"{best_period:.10f} days",
)

print(
    "Period uncertainty:",
    f"{period_error:.10f} days",
)

print(
    "Period uncertainty:",
    f"{period_error * 24 * 60:.3f} minutes",
)

print(
    "Total eclipse duration:",
    f"{best_duration * 24:.4f} hours",
)

print(
    "Duration uncertainty:",
    f"{duration_error * 24:.4f} hours",
)

print(
    "Ingress/egress ratio:",
    f"{best_ingress_ratio:.4f}",
)

print(
    "Approximate ingress duration:",
    f"{best_duration * best_ingress_ratio * 24:.4f} hours",
)

print(
    "Reduced chi-square:",
    f"{reduced_chi_square:.4f}",
)


print("\n" + "=" * 76)
print("FIVE RAW-PIXEL ECLIPSES")
print("=" * 76)

print(
    event_table[
        [
            "sector",
            "cycle",
            "model_center",
            "depth_percent",
            "depth_error_percent",
            "event_points",
            "local_noise_percent",
            "residual_rms_percent",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 76)
print("COMBINED ECLIPSE PROPERTIES")
print("=" * 76)

print(
    "Mean raw-pixel depth:",
    f"{100 * mean_depth:.4f}%",
)

print(
    "Median raw-pixel depth:",
    f"{100 * median_depth:.4f}%",
)

print(
    "Depth spread:",
    f"{100 * depth_spread:.4f}%",
)

print(
    "Approximate radius ratio sqrt(depth):",
    f"{radius_ratio:.4f}",
)

print(
    "Number of fitted data points:",
    number_of_points,
)

print(
    "Number of fitted parameters:",
    number_of_parameters,
)


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as file:
    file.write(
        "TIC 31065777 joint raw-pixel eclipse fit\n"
    )

    file.write("=" * 60 + "\n")

    file.write(
        f"Fit converged: {fit.success}\n"
    )

    file.write(
        f"Epoch: {best_epoch:.10f}\n"
    )

    file.write(
        f"Epoch uncertainty days: "
        f"{epoch_error:.10f}\n"
    )

    file.write(
        f"Period days: "
        f"{best_period:.10f}\n"
    )

    file.write(
        f"Period uncertainty days: "
        f"{period_error:.10f}\n"
    )

    file.write(
        f"Duration hours: "
        f"{best_duration * 24:.6f}\n"
    )

    file.write(
        f"Duration uncertainty hours: "
        f"{duration_error * 24:.6f}\n"
    )

    file.write(
        f"Ingress ratio: "
        f"{best_ingress_ratio:.6f}\n"
    )

    file.write(
        f"Mean depth percent: "
        f"{100 * mean_depth:.6f}\n"
    )

    file.write(
        f"Median depth percent: "
        f"{100 * median_depth:.6f}\n"
    )

    file.write(
        f"Depth spread percent: "
        f"{100 * depth_spread:.6f}\n"
    )

    file.write(
        f"Radius ratio estimate: "
        f"{radius_ratio:.6f}\n"
    )

    file.write(
        f"Reduced chi-square: "
        f"{reduced_chi_square:.6f}\n"
    )

    file.write("\nPer-sector results:\n")

    file.write(
        event_table.to_string(
            index=False
        )
    )

    file.write("\n")


print("\n" + "=" * 76)
print("PLAIN-ENGLISH CHECK")
print("=" * 76)

good_depths = np.sum(
    best_depths >= 0.01
)

consistent_depths = (
    depth_spread <= 0.01
)

if fit.success:
    print(
        "The five raw-pixel events were successfully "
        "fit with one shared orbital clock and eclipse shape."
    )
else:
    print(
        "The optimizer did not fully converge, so the "
        "numbers should not yet be trusted."
    )

print(
    "Events deeper than 1%:",
    f"{good_depths} of {number_of_events}",
)

print(
    "Depths reasonably consistent:",
    consistent_depths,
)

if (
    fit.success
    and good_depths == number_of_events
    and consistent_depths
):
    print(
        "Result: the five events strongly support one "
        "recurring eclipsing system."
    )
else:
    print(
        "Result: at least one event or fitted parameter "
        "still needs closer inspection."
    )

print("\nSaved:")
print(SUMMARY_PATH)
print(EVENT_TABLE_PATH)
print(LIGHTCURVE_TABLE_PATH)
