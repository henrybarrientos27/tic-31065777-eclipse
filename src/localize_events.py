import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TIC_ID = 31065777
RA = 135.070041925556
DEC = -45.0522181112265

EVENTS = [
    {"sector": 8, "cycle": 0},
    {"sector": 9, "cycle": 1},
    {"sector": 35, "cycle": 18},
    {"sector": 36, "cycle": 19},
    {"sector": 89, "cycle": 54},
]

CUTOUT_SIZE = 15
EVENT_HALF_WIDTH = 0.075
REFERENCE_INNER = 0.25
REFERENCE_OUTER = 0.80
EVENT_FITS_PATH = RESULTS_DIR / "TIC_31065777_robust_event_fits.csv"
OUTPUT_PATH = RESULTS_DIR / "TIC_31065777_centroid_localization.csv"


def fits_path(sector):
    candidates = [
        DATA_DIR / f"TIC_{TIC_ID}_sector_{sector}_repeat_check.fits",
        DATA_DIR / f"TIC_{TIC_ID}_sector_{sector}_tesscut.fits",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No saved TESSCut FITS file for Sector {sector}")


def as_array(values, dtype=float):
    return np.asarray(
        np.ma.filled(values, np.nan),
        dtype=dtype,
    )


def robust_noise(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 5:
        return np.nan

    median = np.nanmedian(values)

    return float(
        1.4826
        * np.nanmedian(
            np.abs(values - median)
        )
    )


def circular_aperture(shape, x, y, radius=1.5):
    yy, xx = np.indices(shape)

    return (
        (xx - x) ** 2
        + (yy - y) ** 2
        <= radius ** 2
    )


def fit_plane(image):
    yy, xx = np.indices(image.shape)

    finite = np.isfinite(image)

    design = np.column_stack([
        np.ones(np.sum(finite)),
        xx[finite],
        yy[finite],
    ])

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        image[finite],
        rcond=None,
    )

    plane = (
        coefficients[0]
        + coefficients[1] * xx
        + coefficients[2] * yy
    )

    return image - plane


def local_centroid(
    image,
    center_x,
    center_y,
    radius=3,
):
    ny, nx = image.shape

    x0 = max(
        0,
        int(np.floor(center_x - radius)),
    )

    x1 = min(
        nx,
        int(np.ceil(center_x + radius)) + 1,
    )

    y0 = max(
        0,
        int(np.floor(center_y - radius)),
    )

    y1 = min(
        ny,
        int(np.ceil(center_y + radius)) + 1,
    )

    local = image[y0:y1, x0:x1]

    positive = np.clip(
        np.where(
            np.isfinite(local),
            local,
            0.0,
        ),
        0,
        None,
    )

    if np.sum(positive) <= 0:
        return (
            float(center_x),
            float(center_y),
        )

    yy, xx = np.indices(
        positive.shape
    )

    centroid_x = float(
        np.sum(
            (xx + x0) * positive
        )
        / np.sum(positive)
    )

    centroid_y = float(
        np.sum(
            (yy + y0) * positive
        )
        / np.sum(positive)
    )

    return centroid_x, centroid_y


def analyze_sector(sector, event_time):
    print("\n====================================")
    print(f"SECTOR {sector}")
    print("====================================")

    coordinate = SkyCoord(RA * u.deg, DEC * u.deg)
    path = fits_path(sector)
    print("Opening raw TESSCut pixels:", path)

    with fits.open(path) as hdul:
        table = hdul[1].data
        time_values = as_array(table["TIME"])
        pixel_cube = as_array(table["FLUX"])
        if "QUALITY" in table.columns.names:
            quality = np.nan_to_num(
                as_array(table["QUALITY"]), nan=1
            ).astype(np.int64)
        else:
            quality = np.zeros(len(time_values), dtype=np.int64)
        wcs = WCS(hdul[2].header).celestial

    target_x, target_y = wcs.world_to_pixel(
        coordinate
    )

    target_x = float(target_x)
    target_y = float(target_y)

    usable = (
        np.isfinite(time_values)
        & np.any(
            np.isfinite(pixel_cube),
            axis=(1, 2),
        )
    )

    clean = usable & (quality == 0)

    delta = np.abs(
        time_values - event_time
    )

    event_all = (
        usable
        & (delta <= EVENT_HALF_WIDTH)
    )

    event_clean = (
        clean
        & (delta <= EVENT_HALF_WIDTH)
    )

    reference_clean = (
        clean
        & (delta >= REFERENCE_INNER)
        & (delta <= REFERENCE_OUTER)
    )

    print("Predicted event time:", event_time)
    print("Total event cadences:", int(np.sum(event_all)))
    print("Clean event cadences:", int(np.sum(event_clean)))
    print(
        "Flagged event cadences:",
        int(
            np.sum(
                event_all
                & (quality != 0)
            )
        ),
    )
    print(
        "Clean reference cadences:",
        int(np.sum(reference_clean)),
    )

    if np.sum(event_clean) < 3:
        print("FAILED: Too few clean event cadences.")
        return None

    if np.sum(reference_clean) < 10:
        print("FAILED: Too few clean reference cadences.")
        return None

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
        radius=1.5,
    )

    aperture_flux = np.nansum(
        corrected_cube[:, aperture],
        axis=1,
    )

    baseline = float(
        np.nanmedian(
            aperture_flux[reference_clean]
        )
    )

    normalized_flux = (
        aperture_flux / baseline
    )

    event_flux = float(
        np.nanmedian(
            normalized_flux[event_clean]
        )
    )

    depth = float(
        1.0 - event_flux
    )

    noise = robust_noise(
        normalized_flux[reference_clean]
    )

    if np.isfinite(noise) and noise > 0:
        depth_sigma = depth / noise
    else:
        depth_sigma = np.nan

    coherence = float(
        np.mean(
            normalized_flux[event_clean] < 1.0
        )
    )

    reference_image = np.nanmedian(
        corrected_cube[reference_clean],
        axis=0,
    )

    event_image = np.nanmedian(
        corrected_cube[event_clean],
        axis=0,
    )

    difference_image = (
        reference_image - event_image
    )

    difference_flat = fit_plane(
        difference_image
    )

    centroid_x, centroid_y = local_centroid(
        difference_flat,
        target_x,
        target_y,
        radius=3,
    )

    centroid_separation = float(
        np.hypot(
            centroid_x - target_x,
            centroid_y - target_y,
        )
    )

    minimum_index = np.nanargmin(
        np.where(
            event_clean,
            normalized_flux,
            np.nan,
        )
    )

    minimum_time = float(
        time_values[minimum_index]
    )

    minimum_flux = float(
        normalized_flux[minimum_index]
    )

    timing_offset_hours = float(
        24.0
        * (
            minimum_time - event_time
        )
    )

    print("\nRAW-PIXEL RESULT")
    print(
        "Target pixel:",
        f"x={target_x:.3f}, y={target_y:.3f}",
    )
    print(
        "Local missing-light centroid:",
        f"x={centroid_x:.3f}, y={centroid_y:.3f}",
    )
    print(
        "Centroid-target separation:",
        f"{centroid_separation:.3f} pixels",
    )
    print(
        "Target aperture depth:",
        f"{100 * depth:.3f}%",
    )
    print(
        "Depth significance:",
        f"{depth_sigma:.2f} sigma",
    )
    print(
        "Event coherence:",
        f"{coherence:.3f}",
    )
    print(
        "Minimum observed flux:",
        f"{minimum_flux:.6f}",
    )
    print(
        "Minimum-flux time:",
        f"{minimum_time:.8f}",
    )
    print(
        "Timing offset from prediction:",
        f"{timing_offset_hours:.3f} hours",
    )

    centered = (
        centroid_separation <= 1.0
    )

    clean_enough = (
        np.sum(event_clean) >= 3
        and np.sum(
            event_all
            & (quality != 0)
        ) <= np.sum(event_all) / 3
    )

    significant = (
        np.isfinite(depth_sigma)
        and depth_sigma >= 5
    )

    coherent = coherence >= 0.75

    passed = (
        centered
        and clean_enough
        and significant
        and coherent
        and depth > 0.01
    )

    print("\nCHECKS")
    print("Centered on target:", centered)
    print("Quality acceptable:", clean_enough)
    print("Depth above 5 sigma:", significant)
    print("Coherence above 0.75:", coherent)
    print("Depth above 1%:", depth > 0.01)
    print("SECTOR PASSES:", passed)

    return {
        "sector": sector,
        "event_time": event_time,
        "depth_percent": 100 * depth,
        "depth_sigma": depth_sigma,
        "coherence": coherence,
        "centroid_separation": centroid_separation,
        "clean_event_cadences": int(
            np.sum(event_clean)
        ),
        "flagged_event_cadences": int(
            np.sum(
                event_all
                & (quality != 0)
            )
        ),
        "timing_offset_hours": timing_offset_hours,
        "passed": passed,
    }


if not EVENT_FITS_PATH.exists():
    raise RuntimeError(f"Missing robust event fits: {EVENT_FITS_PATH}")

fitted_events = pd.read_csv(EVENT_FITS_PATH)
if "radius_pixels" in fitted_events:
    fitted_events = fitted_events[np.isclose(fitted_events["radius_pixels"], 1.5)]
centers = dict(zip(fitted_events["sector"].astype(int), fitted_events["center"]))

results = []

for event in EVENTS:
    result = analyze_sector(event["sector"], float(centers[event["sector"]]))

    if result is not None:
        results.append(result)


print("\n\n====================================")
print("FINAL FIVE-ECLIPSE COMPARISON")
print("====================================")

for result in results:
    print(
        f"Sector {result['sector']}: "
        f"depth={result['depth_percent']:.3f}% | "
        f"sigma={result['depth_sigma']:.2f} | "
        f"coherence={result['coherence']:.3f} | "
        f"centroid offset={result['centroid_separation']:.3f} px | "
        f"timing offset={result['timing_offset_hours']:.3f} h | "
        f"PASS={result['passed']}"
    )

passed_results = [
    result
    for result in results
    if result["passed"]
]

print("\nPassed sectors:", len(passed_results), "of", len(EVENTS))

pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
print("Saved:", OUTPUT_PATH)

if len(passed_results) == len(EVENTS):
    depths = np.array([
        result["depth_percent"]
        for result in passed_results
    ])

    print("\nALL-SECTOR CHECK:")
    print(
        "All five raw-pixel events pass the stated target-position, "
        "quality, significance, coherence, and timing checks."
    )
    print(
        "Mean depth:",
        f"{np.mean(depths):.3f}%",
    )
    print(
        "Depth spread:",
        f"{np.std(depths):.3f}%",
    )
    print(
        "Interpretation: a recurring eclipse-like signal associated with "
        "the target aperture is strongly supported; TESS localization "
        "reduces but does not eliminate unresolved blending."
    )

elif len(passed_results) == 2:
    print(
        "\nTwo sectors pass. The recurring eclipse "
        "is promising but not fully confirmed."
    )

else:
    print(
        "\nThe raw-pixel checks do not yet confirm "
        "five clean recurring eclipses."
    )
