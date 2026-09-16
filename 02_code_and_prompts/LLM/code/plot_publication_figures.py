"""Render Nature-style single-panel publication figures from frozen evaluation outputs."""

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
BASELINE = "#4D4D4D"
CI_FILL = "#D9D9D9"


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


def render_metric(
    source: pd.DataFrame,
    figures_dir: Path,
    *,
    metric: str,
    ylabel: str,
    stem: str,
) -> dict:
    if source["shot_size"].astype(int).tolist() != SHOT_ORDER:
        raise ValueError("Figure source shot order is invalid")
    x = np.arange(len(SHOT_ORDER), dtype=float)
    center = source[f"{metric}_mean"].to_numpy(dtype=float)
    low = source[f"{metric}_ci_low"].to_numpy(dtype=float)
    high = source[f"{metric}_ci_high"].to_numpy(dtype=float)
    if np.any(low > center) or np.any(center > high):
        raise ValueError(f"Invalid confidence interval ordering for {metric}")
    baseline = float(source[f"baseline_{metric}"].iloc[0])
    baseline_low = float(source[f"baseline_{metric}_ci_low"].iloc[0])
    baseline_high = float(source[f"baseline_{metric}_ci_high"].iloc[0])

    fig, ax = plt.subplots(figsize=(3.50, 2.75))
    ax.axhspan(baseline_low, baseline_high, color=CI_FILL, alpha=0.55, zorder=0)
    ax.axhline(baseline, color=BASELINE, lw=1.0, ls="--", zorder=1)
    ax.plot(x, center, color=BLUE, lw=1.0, alpha=0.8, zorder=2)
    for index, (xi, value, lo, hi) in enumerate(zip(x, center, low, high)):
        color = ACCENT if SHOT_ORDER[index] == 20 else BLUE
        ax.errorbar(
            xi,
            value,
            yerr=np.array([[value - lo], [hi - value]]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
            markersize=4.2 if SHOT_ORDER[index] == 20 else 3.6,
            zorder=3,
        )
    ax.set_xticks(x, [str(value) for value in SHOT_ORDER])
    ax.set_xlabel("Demonstrations (shots)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.45, len(SHOT_ORDER) - 0.55)
    lower = min(float(low.min()), baseline_low)
    upper = max(float(high.max()), baseline_high)
    span = max(upper - lower, 0.05)
    ax.set_ylim(max(0.0, lower - 0.12 * span), min(1.0, upper + 0.18 * span))
    ax.grid(axis="y", color="#E6E6E6", lw=0.55)
    fig.tight_layout(pad=0.9)
    alignment = require_matplotlib_panel_alignment(
        fig,
        json_out=figures_dir / f"{stem}.alignment.json",
        overlay_svg=figures_dir / f"{stem}.alignment.svg",
        strict=True,
    )
    fig.savefig(figures_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "metric": metric,
        "stem": stem,
        "alignment_verdict": alignment["verdict"],
        "n_validation": int(source["n"].iloc[0]),
        "primary_shot": 20,
        "baseline": baseline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    source_path = run_dir / "publication" / "source_data" / "Figure1_and_supplementary_source_data.csv"
    figures_dir = run_dir / "publication" / "figures"
    if not source_path.is_file():
        raise FileNotFoundError("Publication figure source data are missing")
    figures_dir.mkdir(exist_ok=True)
    if any(figures_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite existing publication figures")
    source = pd.read_csv(source_path)
    reports = [
        render_metric(source, figures_dir, metric="auroc", ylabel="AUROC", stem="Figure1_AUROC_by_shot"),
        render_metric(source, figures_dir, metric="auprc", ylabel="AUPRC", stem="FigureS1_AUPRC_by_shot"),
        render_metric(source, figures_dir, metric="brier", ylabel="Brier score (lower is better)", stem="FigureS2_Brier_by_shot"),
    ]
    (figures_dir / "figure_generation_record.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "COMPLETE", "figures": reports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
