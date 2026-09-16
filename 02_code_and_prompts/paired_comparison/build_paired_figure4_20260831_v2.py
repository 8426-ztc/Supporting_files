from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\Administrator\Documents\ChatGPT\转化分析")
OUT = ROOT / "outputs"
SKILL_SCRIPTS = Path(r"C:\Users\Administrator\.codex\skills\nature-figure\scripts")
PATIENT_CSV = OUT / "ML_LLM_PAIRED_PATIENT_MASTER_20260831_v1.csv"
REPLICATE_CSV = OUT / "ML_LLM_PRIMARY_REPLICATE_METRICS_20260831_v1.csv"
SUMMARY_CSV = OUT / "ML_LLM_PAIRED_METRICS_SUMMARY_20260831_v1.csv"
STEM = OUT / "Figure4_paired_ML_LLM_external_20260831_v2"
RECORD_JSON = OUT / "Figure4_paired_generation_record_20260831_v2.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(-0.15, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")


def style_axis(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3, color="#606060")


def main() -> None:
    inputs = [PATIENT_CSV, REPLICATE_CSV, SUMMARY_CSV]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    planned = [
        *[Path(str(STEM) + suffix) for suffix in [".svg", ".pdf", ".tiff", ".png", ".alignment.json", ".alignment.svg"]],
        RECORD_JSON,
    ]
    existing = [str(path) for path in planned if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing files:\n" + "\n".join(existing))

    patient = pd.read_csv(PATIENT_CSV)
    replicate = pd.read_csv(REPLICATE_CSV)
    summary = pd.read_csv(SUMMARY_CSV).set_index("estimand")

    delta_auc = float(summary.loc["Paired delta AUROC: LLM minus ExtraTrees", "estimate"])
    delta_auc_low = float(summary.loc["Paired delta AUROC: LLM minus ExtraTrees", "ci_low"])
    delta_auc_high = float(summary.loc["Paired delta AUROC: LLM minus ExtraTrees", "ci_high"])
    brier_gain = float(summary.loc["Brier improvement: ExtraTrees minus LLM", "estimate"])
    brier_low = float(summary.loc["Brier improvement: ExtraTrees minus LLM", "ci_low"])
    brier_high = float(summary.loc["Brier improvement: ExtraTrees minus LLM", "ci_high"])
    rho = float(summary.loc["Spearman probability correlation", "estimate"])

    y = patient["target_mpr90"].to_numpy(dtype=int)
    ml_probability = patient["extratrees_probability"].to_numpy(dtype=float)
    llm_probability = patient["llm_20shot_probability_mean_across_20_replicates"].to_numpy(dtype=float)
    ml_threshold = float(patient["extratrees_threshold"].iloc[0])
    ml_correct = patient["extratrees_correct"].to_numpy(dtype=int).astype(bool)
    llm_correct = patient["llm_replicate_averaged_correct"].to_numpy(dtype=int).astype(bool)
    matrix = np.array(
        [
            [int(np.sum(~llm_correct & ~ml_correct)), int(np.sum(~llm_correct & ml_correct))],
            [int(np.sum(llm_correct & ~ml_correct)), int(np.sum(llm_correct & ml_correct))],
        ]
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": 600,
        }
    )
    dark = "#484878"
    pink_edge = "#D99AAF"
    grid = "#E8E8E8"
    text = "#303030"

    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(7.2, 2.75))
    fig.subplots_adjust(left=0.145, right=0.98, bottom=0.24, top=0.82, wspace=0.62)

    style_axis(axa)
    axa.grid(axis="x", color=grid, linewidth=0.55, zorder=0)
    axa.axvline(0, color="#909090", linestyle=(0, (3, 3)), linewidth=0.8, zorder=1)
    y_positions = np.array([1.0, 0.0])
    replicate_values = [
        replicate["delta_auc_llm_minus_ml"].to_numpy(dtype=float),
        replicate["brier_improvement_ml_minus_llm"].to_numpy(dtype=float),
    ]
    for y0, values in zip(y_positions, replicate_values):
        jitter = np.linspace(-0.16, 0.16, len(values))
        axa.scatter(values, np.full(len(values), y0) + jitter, s=10, color="#D8D8D8", edgecolor="white", linewidth=0.3, zorder=2)
    estimates = np.array([delta_auc, brier_gain])
    lows = np.array([delta_auc_low, brier_low])
    highs = np.array([delta_auc_high, brier_high])
    for y0, estimate, low_ci, high_ci in zip(y_positions, estimates, lows, highs):
        axa.errorbar(estimate, y0, xerr=[[estimate - low_ci], [high_ci - estimate]], fmt="o", color=dark, ecolor=dark, markersize=4.2, elinewidth=1.0, capsize=2.2, zorder=4)
    extent = max(abs(np.r_[lows, highs, replicate_values[0], replicate_values[1]]).max() + 0.025, 0.10)
    axa.set_xlim(-extent, extent)
    axa.set_ylim(-0.48, 1.48)
    axa.set_yticks(y_positions, ["AUROC Δ\n(LLM − ML)", "Brier improvement\n(ML − LLM)"])
    axa.set_xlabel("Oriented paired difference\n(positive favours LLM)")
    axa.set_title("Paired performance differences", loc="left", fontweight="bold")
    panel_label(axa, "a")

    style_axis(axb)
    axb.grid(color=grid, linewidth=0.5, zorder=0)
    axb.plot([0, 1], [0, 1], color="#B8B8B8", linestyle=(0, (3, 3)), linewidth=0.8, zorder=1)
    axb.axvline(ml_threshold, color=pink_edge, linestyle=(0, (2, 2)), linewidth=0.75, zorder=1)
    axb.axhline(0.5, color=pink_edge, linestyle=(0, (2, 2)), linewidth=0.75, zorder=1)
    non_event = y == 0
    event = y == 1
    axb.scatter(ml_probability[non_event], llm_probability[non_event], s=18, facecolor="#D8D8D8", edgecolor="white", linewidth=0.45, zorder=3)
    axb.scatter(ml_probability[event], llm_probability[event], s=20, facecolor=dark, edgecolor="white", linewidth=0.45, zorder=4)
    axb.set_xlim(0, 1)
    axb.set_ylim(0, 1)
    axb.set_xlabel("Locked ExtraTrees probability")
    axb.set_ylabel("Mean 20-shot LLM probability")
    axb.set_title(f"Probability agreement (ρ = {rho:.2f})", loc="left", fontweight="bold")
    panel_label(axb, "b")

    cmap = LinearSegmentedColormap.from_list("correctness", ["#F3F3F6", "#B4C0E4", dark])
    axc.imshow(matrix, cmap=cmap, vmin=0, vmax=int(matrix.max()), aspect="auto")
    for row in range(2):
        for col in range(2):
            count = int(matrix[row, col])
            colour = "white" if count > matrix.max() * 0.55 else text
            axc.text(col, row, f"{count}\n({count / 57:.0%})", ha="center", va="center", fontsize=7, color=colour, fontweight="bold")
    axc.set_xticks([0, 1], ["No", "Yes"])
    axc.set_yticks([0, 1], ["No", "Yes"])
    axc.set_xlabel("ExtraTrees correct")
    axc.set_ylabel("LLM mean correct")
    axc.set_title("Correctness overlap", loc="left", fontweight="bold")
    axc.tick_params(width=0.7, length=3, color="#606060")
    for spine in axc.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#606060")
    panel_label(axc, "c")

    sys.path.insert(0, str(SKILL_SCRIPTS))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        panel_ids=["a", "b", "c"],
        json_out=str(STEM) + ".alignment.json",
        overlay_svg=str(STEM) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )

    outputs = []
    for suffix, kwargs in [
        (".svg", {}),
        (".pdf", {}),
        (".tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
        (".png", {"dpi": 300}),
    ]:
        path = Path(str(STEM) + suffix)
        fig.savefig(path, facecolor="white", **kwargs)
        outputs.append(path)
    plt.close(fig)

    generated = outputs + [Path(str(STEM) + ".alignment.json"), Path(str(STEM) + ".alignment.svg")]
    record = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "backend": "Python/matplotlib",
        "inputs": {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in inputs},
        "outputs": {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in generated},
        "layout_changes_from_v1": [
            "shortened panel c title to prevent page clipping",
            "moved Spearman rho into panel b title",
            "removed redundant in-panel outcome legend; colour definitions remain in the figure legend",
        ],
        "overwrite_policy": "all planned outputs were asserted absent before generation",
    }
    RECORD_JSON.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "outputs": [str(path) for path in generated + [RECORD_JSON]]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
