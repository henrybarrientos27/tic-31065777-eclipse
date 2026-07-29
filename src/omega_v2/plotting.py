"""Diagnostic plots for OMEGA v2 candidates."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_candidate(
    tic_id: int,
    lightcurve: pd.DataFrame,
    events: pd.DataFrame,
    summary: dict,
    repeat: dict | None,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)

    for sector, group in lightcurve.groupby("sector"):
        axes[0].scatter(
            group["time"],
            100.0 * group["residual_flux"],
            s=3,
            alpha=0.55,
            label=f"Sector {int(sector)}",
        )

    if len(events):
        axes[0].scatter(
            events["time"],
            -events["depth_percent"],
            marker="v",
            s=65,
            color="tab:red",
            edgecolor="black",
            linewidth=0.4,
            label="Detected sustained dips",
            zorder=6,
        )

    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("TESS time (BTJD)")
    axes[0].set_ylabel("Segment-normalized flux residual (%)")
    axes[0].set_title(
        f"TIC {int(tic_id)} — {summary['classification']} — "
        f"OMEGA score {summary['omega_score']:.1f}"
    )
    axes[0].legend(fontsize=8, ncol=3)

    if repeat and np.isfinite(repeat.get("period_days", np.nan)):
        period = float(repeat["period_days"])
        epoch = float(repeat["epoch"])
        phase = (
            ((lightcurve["time"].to_numpy() - epoch + 0.5 * period) % period)
            - 0.5 * period
        )
        order = np.argsort(phase)
        axes[1].scatter(
            phase[order],
            100.0 * lightcurve["residual_flux"].to_numpy()[order],
            s=4,
            alpha=0.35,
        )
        axes[1].axvline(0, linestyle="--", color="tab:red")
        axes[1].set_xlabel(f"Days from linked-event phase (P={period:.7f} d)")
        axes[1].set_ylabel("Flux residual (%)")
        axes[1].set_title(
            f"Candidate phase fold: {repeat['positive_windows']} positive / "
            f"{repeat['negative_windows']} flat covered windows"
        )
    elif len(events):
        plot_events = events.sort_values("event_score", ascending=False).head(30)
        labels = [f"S{int(value)}" for value in plot_events["sector"]]
        axes[1].bar(
            np.arange(len(plot_events)),
            plot_events["event_score"],
            color="tab:blue",
        )
        axes[1].set_xticks(np.arange(len(plot_events)), labels, rotation=60)
        axes[1].set_xlabel("Detected event")
        axes[1].set_ylabel("Depth / local robust noise")
        axes[1].set_title("Strongest unlinked dip events")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No sustained eclipse-like events passed the first-pass thresholds.",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_axis_off()

    fig.savefig(output_path, dpi=180)
    plt.close(fig)

