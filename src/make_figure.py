from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "results" / "evidence_packets" / "TIC_31065777"
OUTPUT = ROOT / "results" / "publication"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    lightcurve = pd.read_csv(ROOT / "results" / "TIC_31065777_raw_pixel_lightcurves.csv")
    events = pd.read_csv(ROOT / "results" / "TIC_31065777_robust_event_fits.csv")
    events = events[np.isclose(events["radius_pixels"], 1.5)].copy()
    blank = pd.read_csv(ROOT / "results" / "TIC_31065777_blank_apertures.csv")
    temporal = pd.read_csv(ROOT / "results" / "TIC_31065777_temporal_controls.csv")
    leave_out = pd.read_csv(ROOT / "results" / "TIC_31065777_leave_one_out.csv")
    centroids = pd.read_csv(ROOT / "results" / "TIC_31065777_centroid_localization.csv")
    with (PACKET / "data" / "signal_measurements.json").open() as handle:
        measurement = json.load(handle)

    period = float(measurement["period_days"])
    epoch = float(measurement["epoch_btjd"])
    median_depth = float(measurement["median_depth_percent"])

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
        }
    )
    blue = "#0072B2"
    orange = "#D55E00"
    green = "#009E73"
    gray = "#5B6573"

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.35), constrained_layout=True)

    # (a) Phase-folded photometry.
    axis = axes[0, 0]
    phase_hours = lightcurve["hours_from_center"].to_numpy(float)
    flux = lightcurve["normalized_flux"].to_numpy(float)
    selected = np.abs(phase_hours) <= 8.0
    axis.scatter(
        phase_hours[selected],
        flux[selected],
        s=5,
        color=blue,
        alpha=0.23,
        linewidths=0,
        rasterized=True,
        label="TESS cadences",
    )
    bins = np.linspace(-8, 8, 65)
    bin_index = np.digitize(phase_hours[selected], bins)
    centers: list[float] = []
    medians: list[float] = []
    for index in range(1, len(bins)):
        in_bin = bin_index == index
        if np.count_nonzero(in_bin) >= 3:
            centers.append(0.5 * (bins[index - 1] + bins[index]))
            medians.append(float(np.median(flux[selected][in_bin])))
    axis.plot(centers, medians, color=orange, lw=1.8, label="bin median")
    axis.set(
        title="Phase-folded TESS photometry",
        xlabel="Hours from eclipse center",
        ylabel="Normalized flux",
        xlim=(-8, 8),
    )
    axis.legend(loc="lower right", frameon=True)
    axis.text(0.02, 0.96, "(a)", transform=axis.transAxes, va="top", fontweight="bold")

    # (b) Depth consistency across sectors and cadences.
    axis = axes[0, 1]
    axis.errorbar(
        events["sector"],
        events["depth_percent"],
        yerr=events["depth_error_percent"],
        fmt="o",
        color=blue,
        ecolor=blue,
        capsize=2.5,
        ms=4.5,
    )
    axis.axhline(
        median_depth,
        color=orange,
        ls="--",
        lw=1.4,
        label=f"median {median_depth:.3f}%",
    )
    axis.set(
        title="Raw-pixel depth by sector",
        xlabel="TESS sector",
        ylabel="Depth (%)",
        ylim=(5.65, 6.95),
    )
    axis.legend(loc="lower right", frameon=True)
    axis.text(0.02, 0.96, "(b)", transform=axis.transAxes, va="top", fontweight="bold")

    # (c) Timing residuals from the conservative linear ephemeris.
    axis = axes[1, 0]
    predicted = epoch + events["cycle"].to_numpy(float) * period
    residual_minutes = 1440.0 * (events["center"].to_numpy(float) - predicted)
    axis.axhline(0, color=gray, lw=1)
    axis.plot(events["cycle"], residual_minutes, "o-", color=green, ms=4.5, lw=1)
    max_leave_out = float(
        np.max(np.abs(leave_out["excluded_event_prediction_error_minutes"]))
    )
    axis.axhspan(
        -max_leave_out,
        max_leave_out,
        color=green,
        alpha=0.10,
        label="max leave-one-out error",
    )
    for cycle, sector, residual in zip(events["cycle"], events["sector"], residual_minutes):
        axis.annotate(
            f"S{int(sector)}",
            (cycle, residual),
            xytext=(0, 6 if residual <= 0 else -11),
            textcoords="offset points",
            ha="center",
            fontsize=6.5,
        )
    axis.set(
        title="Linear-ephemeris O-C residuals",
        xlabel="Cycle number",
        ylabel="Observed minus calculated (min)",
        ylim=(-2.2, 2.2),
    )
    axis.legend(loc="lower right", frameon=True)
    axis.text(0.02, 0.96, "(c)", transform=axis.transAxes, va="top", fontweight="bold")

    # (d) Spatial and temporal control depths versus the target signal.
    axis = axes[1, 1]
    blank_groups = (
        blank.assign(offset=lambda frame: frame.apply(
            lambda row: f"({int(row.offset_x):+d},{int(row.offset_y):+d}) px", axis=1
        ))
        .groupby("offset")["depth_target_units_percent"]
        .apply(lambda values: float(np.max(np.abs(values))))
    )
    temporal_max = float(np.max(np.abs(temporal["depth_target_units_percent"])))
    labels = ["target"] + list(blank_groups.index) + ["time-shift"]
    values = [median_depth] + list(blank_groups.values) + [temporal_max]
    colors = [orange] + [blue] * len(blank_groups) + [green]
    positions = np.arange(len(labels))
    axis.bar(positions, values, color=colors, width=0.72)
    axis.set_xticks(positions, labels, rotation=28, ha="right")
    axis.set(
        title="Target signal versus controls",
        ylabel="Maximum absolute depth (%)",
        ylim=(0, 6.9),
    )
    passed_centroids = centroids[centroids["passed"].astype(bool)]
    centroid_low = float(passed_centroids["centroid_separation"].min())
    centroid_high = float(passed_centroids["centroid_separation"].max())
    centroid_sectors = "/".join(
        f"S{int(value)}" for value in passed_centroids["sector"]
    )
    axis.text(
        0.98,
        0.95,
        f"Missing-light centroids in {centroid_sectors}:\n"
        f"{centroid_low:.2f}-{centroid_high:.2f} TESS pixel from target",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    axis.text(0.02, 0.96, "(d)", transform=axis.transAxes, va="top", fontweight="bold")

    figure.suptitle(
        "TIC 31065777: five TESS eclipse-like events on a 40.5717-day ephemeris",
        fontsize=11,
        fontweight="bold",
    )
    png_path = OUTPUT / "TIC_31065777_RNAAS_FIGURE.png"
    pdf_path = OUTPUT / "TIC_31065777_RNAAS_FIGURE.pdf"
    figure.savefig(png_path, dpi=300, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
