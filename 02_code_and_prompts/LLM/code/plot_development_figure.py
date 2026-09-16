"""Render the non-demonstration development performance figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_panel_alignment import require_matplotlib_panel_alignment


SHOT_ORDER = [0, 4, 8, 10, 20, 40, 80, 109]
BLUE = "#3F6FA6"
ACCENT = "#B64342"
NE_FILL = "#EFEFEF"
GRID = "#E6E6E6"
STEM = "FigureS3_Development_performance_by_shot"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    source_path = run_dir / "publication" / "tables" / "TableS3_development_performance_by_shot.csv"
    figures_dir = run_dir / "publication" / "figures"
    if not source_path.is_file():
        raise FileNotFoundError("Development performance source table is missing")
    figures_dir.mkdir(exist_ok=True)
    for suffix in ("svg", "pdf", "tiff", "png", "alignment.json", "alignment.svg"):
        if (figures_dir / f"{STEM}.{suffix}").exists():
            raise FileExistsError(f"Refusing to overwrite existing {STEM}.{suffix}")

    source = pd.read_csv(source_path)
    if source["shot_size"].astype(int).tolist() != SHOT_ORDER or len(source) != 8:
        raise ValueError("Development source must contain all eight shot rows in frozen order")
    if int(source.loc[source["shot_size"] == 109, "n_evaluable_min"].iloc[0]) != 0:
        raise ValueError("109-shot development performance must be non-evaluable")
    if source.loc[source["shot_size"] == 109, ["auroc_mean", "auprc_mean", "brier_mean"]].notna().any().any():
        raise ValueError("109-shot development metrics must remain missing")

    evaluable = source.loc[source["n_evaluable_min"] > 0].copy()
    if len(evaluable) != 7:
        raise ValueError("Expected seven evaluable development shot conditions")
    x = np.arange(len(SHOT_ORDER), dtype=float)
    x_eval = x[: len(evaluable)]
    n_values = source["n_evaluable_min"].astype(int).tolist()
    tick_labels = [f"{shot}\n({n})" for shot, n in zip(SHOT_ORDER, n_values)]

    metrics = [
        ("auroc", "AUROC", (0.15, 0.90)),
        ("auprc", "AUPRC", (0.15, 0.90)),
        ("brier", "Brier score\n(lower is better)", (0.15, 0.56)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.10, 2.65), sharex=True)
    for panel, (ax, (metric, ylabel, ylim)) in enumerate(zip(axes, metrics)):
        center = evaluable[f"{metric}_mean"].to_numpy(dtype=float)
        low = evaluable[f"{metric}_min"].to_numpy(dtype=float)
        high = evaluable[f"{metric}_max"].to_numpy(dtype=float)
        if np.any(low > center) or np.any(center > high):
            raise ValueError(f"Invalid replicate range ordering for {metric}")

        ax.axvspan(6.62, 7.38, color=NE_FILL, zorder=0)
        ax.plot(x_eval, center, color=BLUE, lw=1.0, alpha=0.8, zorder=2)
        for index, (xi, value, lo, hi) in enumerate(zip(x_eval, center, low, high)):
            color = ACCENT if SHOT_ORDER[index] == 20 else BLUE
            ax.errorbar(
                xi,
                value,
                yerr=np.array([[value - lo], [hi - value]]),
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.2,
                capthick=1.0,
                markersize=4.0 if SHOT_ORDER[index] == 20 else 3.4,
                zorder=3,
            )
        ax.text(7.0, ylim[1] - 0.035 * (ylim[1] - ylim[0]), "NE", ha="center", va="top", color="#666666", fontsize=6.5)
        ax.set_xticks(x, tick_labels)
        ax.set_xlim(-0.45, 7.45)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Shots (evaluable n)")
        ax.grid(axis="y", color=GRID, lw=0.55)
        ax.text(
            -0.16,
            1.04,
            chr(ord("a") + panel),
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.88, wspace=0.34)
    alignment = require_matplotlib_panel_alignment(
        fig,
        json_out=figures_dir / f"{STEM}.alignment.json",
        overlay_svg=figures_dir / f"{STEM}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(figures_dir / f"{STEM}.svg", bbox_inches="tight")
    fig.savefig(figures_dir / f"{STEM}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{STEM}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(figures_dir / f"{STEM}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    record = {
        "status": "COMPLETE",
        "stem": STEM,
        "source": str(source_path),
        "rows_before_evaluability_filter": int(len(source)),
        "rows_plotted": int(len(evaluable)),
        "excluded_condition": "109-shot development; n_evaluable=0 because all labels were demonstrations",
        "center": "mean across independent prompt replicates",
        "interval": "minimum to maximum across independent prompt replicates",
        "replicates": {str(int(row.shot_size)): int(row.prompt_replicates) for row in source.itertuples()},
        "evaluable_n": {str(int(row.shot_size)): int(row.n_evaluable_min) for row in source.itertuples()},
        "primary_shot_highlighted": 20,
        "alignment_verdict": alignment["verdict"],
    }
    (figures_dir / f"{STEM}.generation.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
