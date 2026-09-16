from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\Administrator\Documents\ChatGPT\转化分析")
OUT = ROOT / "outputs"
ML_ROOT = Path(r"D:\gpt\mpr90_stable_rf_rfe_outputs")
LLM_ROOT = Path(r"D:\gpt\publication_run_20260829_v3_batched")
SKILL_SCRIPTS = Path(r"C:\Users\Administrator\.codex\skills\nature-figure\scripts")

ML_PREDICTIONS = ML_ROOT / "predictions_MPR90_for_RFS.csv"
ML_MANIFEST = ML_ROOT / "analysis_manifest.json"
ML_PRIMARY_REPORT = ML_ROOT / "PRIMARY_REPORT_OOF_AND_EXTERNAL_AUC.csv"
ML_DETAILED = ML_ROOT / "locked_primary_model_detailed_performance.csv"
ML_FIGURE_SVG = ML_ROOT / "figures" / "Fig2C_locked_external_ROC.svg"
LLM_FEATURES = LLM_ROOT / "operator" / "validation_features_BLINDED.csv"
LLM_GROUND_TRUTH = LLM_ROOT / "private" / "validation_ground_truth_LOCKED.csv"
LLM_MASTER = Path(
    r"D:\gpt\data\2026-08-31\referenced-chatgpt-conversation-this-is-an-2"
    r"\outputs\LLM_MASTER_PREDICTIONS_20260831_v2.csv"
)

STEM = OUT / "Figure4_paired_ML_LLM_external_20260831_v1"
PATIENT_CSV = OUT / "ML_LLM_PAIRED_PATIENT_MASTER_20260831_v1.csv"
REPLICATE_CSV = OUT / "ML_LLM_PRIMARY_REPLICATE_METRICS_20260831_v1.csv"
BOOTSTRAP_CSV = OUT / "ML_LLM_HIERARCHICAL_BOOTSTRAP_10000_20260831_v1.csv"
SUMMARY_CSV = OUT / "ML_LLM_PAIRED_METRICS_SUMMARY_20260831_v1.csv"
PAIR_QC_JSON = OUT / "ML_LLM_PAIRING_QC_20260831_v1.json"
RESULTS_MD = OUT / "Results_ML_LLM_paired_20260831_v1.md"
METHODS_MD = OUT / "Statistical_analysis_ML_LLM_paired_20260831_v1.md"
LEGEND_MD = OUT / "Figure4_paired_legend_20260831_v1.md"
CONSISTENCY_MD = OUT / "ML_RESULTS_CONSISTENCY_AUDIT_20260831_v1.md"
RECORD_JSON = OUT / "ML_LLM_paired_generation_record_20260831_v1.json"

N_BOOT = 10_000
BOOT_SEED = 20260831


def assert_new(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing files:\n" + "\n".join(existing))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def auc_binary(y: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    positive = probability[y == 1]
    negative = probability[y == 0]
    if len(positive) == 0 or len(negative) == 0:
        return math.nan
    differences = positive[:, None] - negative[None, :]
    return float(np.mean((differences > 0) + 0.5 * (differences == 0)))


def auc_rows_from_strata(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    differences = positive[:, :, None] - negative[:, None, :]
    return np.mean((differences > 0) + 0.5 * (differences == 0), axis=(1, 2))


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def style_axis(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3, color="#606060")


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    input_paths = [
        ML_PREDICTIONS,
        ML_MANIFEST,
        ML_PRIMARY_REPORT,
        ML_DETAILED,
        ML_FIGURE_SVG,
        LLM_FEATURES,
        LLM_GROUND_TRUTH,
        LLM_MASTER,
    ]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))

    planned = [
        *[Path(str(STEM) + suffix) for suffix in [".svg", ".pdf", ".tiff", ".png", ".alignment.json", ".alignment.svg"]],
        PATIENT_CSV,
        REPLICATE_CSV,
        BOOTSTRAP_CSV,
        SUMMARY_CSV,
        PAIR_QC_JSON,
        RESULTS_MD,
        METHODS_MD,
        LEGEND_MD,
        CONSISTENCY_MD,
        RECORD_JSON,
    ]
    assert_new(planned)

    ml_manifest = json.loads(ML_MANIFEST.read_text(encoding="utf-8"))
    ml_threshold = float(ml_manifest["locked_threshold"])
    ml_all = pd.read_csv(ML_PREDICTIONS)
    ml = ml_all.loc[ml_all["dataset"].eq("external")].copy()
    features = pd.read_csv(LLM_FEATURES)
    truth = pd.read_csv(LLM_GROUND_TRUTH)
    llm_master = pd.read_csv(LLM_MASTER, low_memory=False)

    keys = ["bmi", "tumorsizeb", "ast", "tbil"]
    for key in keys:
        features[key] = pd.to_numeric(features[key], errors="raise").round(8)
        ml[key] = pd.to_numeric(ml[key], errors="raise").round(8)

    feature_duplicates = int(features.duplicated(keys, keep=False).sum())
    ml_duplicates = int(ml.duplicated(keys, keep=False).sum())
    matched = features.merge(ml, on=keys, how="outer", indicator=True, suffixes=("_llm", "_ml"))
    merge_counts = {str(key): int(value) for key, value in matched["_merge"].value_counts().to_dict().items()}
    if feature_duplicates or ml_duplicates or merge_counts != {"both": 57, "left_only": 0, "right_only": 0}:
        raise AssertionError(f"Pairing failed: duplicates={feature_duplicates}/{ml_duplicates}, merge={merge_counts}")
    paired = matched.loc[matched["_merge"].eq("both")].merge(truth, on="public_id", how="left")
    if paired["public_id"].nunique() != 57:
        raise AssertionError("Public IDs are not unique after pairing")
    if paired["id"].nunique() != 57:
        raise AssertionError("ML IDs are not unique after pairing")
    if not paired["mpr_binary"].astype(int).eq(paired["target_mpr90"].astype(int)).all():
        raise AssertionError("Outcome disagreement after pairing")
    ml_probability = pd.to_numeric(paired["predicted_probability_MPR"], errors="raise")
    if not ml_probability.between(0, 1, inclusive="both").all():
        raise AssertionError("Invalid ML probability")
    source_ml_class = pd.to_numeric(paired["predicted_MPR_at_locked_threshold"], errors="raise").astype(int)
    recomputed_ml_class = (ml_probability >= ml_threshold).astype(int)
    if not source_ml_class.eq(recomputed_ml_class).all():
        raise AssertionError("ML threshold labels do not reproduce")

    llm20 = llm_master.loc[
        llm_master["cohort"].eq("validation") & pd.to_numeric(llm_master["shot_size"]).eq(20)
    ].copy()
    llm20["replicate"] = pd.to_numeric(llm20["replicate"], errors="raise").astype(int)
    llm20["target_mpr90"] = pd.to_numeric(llm20["target_mpr90"], errors="raise").astype(int)
    llm20["llm_prob_mpr90"] = pd.to_numeric(llm20["llm_prob_mpr90"], errors="raise")
    llm_pivot = llm20.pivot(index="public_id", columns="replicate", values="llm_prob_mpr90")
    if llm_pivot.shape != (57, 20) or llm_pivot.isna().any().any():
        raise AssertionError(f"Unexpected 20-shot matrix shape or missingness: {llm_pivot.shape}")

    paired = paired.sort_values("public_id").reset_index(drop=True)
    llm_pivot = llm_pivot.loc[paired["public_id"], list(range(1, 21))]
    y = paired["target_mpr90"].to_numpy(dtype=int)
    ml_probability_array = ml_probability.loc[paired.index].to_numpy(dtype=float)
    # ml_probability was created before sorting; align explicitly by public ID.
    ml_probability_array = paired["predicted_probability_MPR"].to_numpy(dtype=float)
    llm_probabilities = llm_pivot.to_numpy(dtype=float).T

    if int(y.sum()) != 25 or int((1 - y).sum()) != 32:
        raise AssertionError("Unexpected outcome counts")

    ml_auc = auc_binary(y, ml_probability_array)
    ml_brier = float(np.mean((ml_probability_array - y) ** 2))
    llm_auc_by_rep = np.array([auc_binary(y, row) for row in llm_probabilities])
    llm_brier_by_rep = np.mean((llm_probabilities - y[None, :]) ** 2, axis=1)
    llm_auc_mean = float(llm_auc_by_rep.mean())
    llm_brier_mean = float(llm_brier_by_rep.mean())
    delta_auc = llm_auc_mean - ml_auc
    brier_improvement = ml_brier - llm_brier_mean

    rng = np.random.default_rng(BOOT_SEED)
    positive_indices = np.flatnonzero(y == 1)
    negative_indices = np.flatnonzero(y == 0)
    bootstrap_rows: list[dict[str, float | int]] = []
    for draw in range(1, N_BOOT + 1):
        sampled_positive = rng.choice(positive_indices, size=len(positive_indices), replace=True)
        sampled_negative = rng.choice(negative_indices, size=len(negative_indices), replace=True)
        sampled_patients = np.concatenate([sampled_positive, sampled_negative])
        sampled_replicates = rng.integers(0, llm_probabilities.shape[0], size=llm_probabilities.shape[0])

        ml_auc_draw = auc_binary(y[sampled_patients], ml_probability_array[sampled_patients])
        selected = llm_probabilities[sampled_replicates]
        llm_auc_draws = auc_rows_from_strata(selected[:, sampled_positive], selected[:, sampled_negative])
        llm_auc_draw = float(llm_auc_draws.mean())
        ml_brier_draw = float(np.mean((ml_probability_array[sampled_patients] - y[sampled_patients]) ** 2))
        llm_brier_draw = float(
            np.mean((selected[:, sampled_patients] - y[sampled_patients][None, :]) ** 2)
        )
        bootstrap_rows.append(
            {
                "bootstrap_draw": draw,
                "ml_auc": ml_auc_draw,
                "llm_mean_auc": llm_auc_draw,
                "delta_auc_llm_minus_ml": llm_auc_draw - ml_auc_draw,
                "ml_brier": ml_brier_draw,
                "llm_mean_brier": llm_brier_draw,
                "brier_improvement_ml_minus_llm": ml_brier_draw - llm_brier_draw,
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    delta_auc_ci = percentile_interval(bootstrap["delta_auc_llm_minus_ml"].to_numpy())
    brier_improvement_ci = percentile_interval(bootstrap["brier_improvement_ml_minus_llm"].to_numpy())
    ml_auc_ci = percentile_interval(bootstrap["ml_auc"].to_numpy())
    llm_auc_ci = percentile_interval(bootstrap["llm_mean_auc"].to_numpy())
    ml_brier_ci = percentile_interval(bootstrap["ml_brier"].to_numpy())
    llm_brier_ci = percentile_interval(bootstrap["llm_mean_brier"].to_numpy())

    llm_probability_mean = llm_probabilities.mean(axis=0)
    llm_probability_sd = llm_probabilities.std(axis=0, ddof=1)
    llm_avg_auc = auc_binary(y, llm_probability_mean)
    llm_avg_brier = float(np.mean((llm_probability_mean - y) ** 2))
    ml_ranks = pd.Series(ml_probability_array).rank(method="average").to_numpy()
    llm_ranks = pd.Series(llm_probability_mean).rank(method="average").to_numpy()
    spearman_rho = float(np.corrcoef(ml_ranks, llm_ranks)[0, 1])

    ml_class = (ml_probability_array >= ml_threshold).astype(int)
    llm_avg_class = (llm_probability_mean >= 0.5).astype(int)
    ml_correct = ml_class == y
    llm_correct = llm_avg_class == y
    correctness_matrix = np.array(
        [
            [int(np.sum(~llm_correct & ~ml_correct)), int(np.sum(~llm_correct & ml_correct))],
            [int(np.sum(llm_correct & ~ml_correct)), int(np.sum(llm_correct & ml_correct))],
        ]
    )
    disagreement_n = int(np.sum(ml_class != llm_avg_class))

    patient_output = pd.DataFrame(
        {
            "public_id": paired["public_id"],
            "target_mpr90": y,
            "extratrees_probability": ml_probability_array,
            "extratrees_threshold": ml_threshold,
            "extratrees_class": ml_class,
            "llm_20shot_probability_mean_across_20_replicates": llm_probability_mean,
            "llm_20shot_probability_sd_across_20_replicates": llm_probability_sd,
            "llm_threshold": 0.5,
            "llm_replicate_averaged_class": llm_avg_class,
            "extratrees_correct": ml_correct.astype(int),
            "llm_replicate_averaged_correct": llm_correct.astype(int),
            "threshold_classification_disagreement": (ml_class != llm_avg_class).astype(int),
        }
    )
    replicate_output = pd.DataFrame(
        {
            "shot_size": 20,
            "replicate": np.arange(1, 21),
            "n_patients": 57,
            "llm_auc": llm_auc_by_rep,
            "extratrees_auc": ml_auc,
            "delta_auc_llm_minus_ml": llm_auc_by_rep - ml_auc,
            "llm_brier": llm_brier_by_rep,
            "extratrees_brier": ml_brier,
            "brier_improvement_ml_minus_llm": ml_brier - llm_brier_by_rep,
        }
    )
    summary_output = pd.DataFrame(
        [
            {"estimand": "ExtraTrees AUROC", "estimate": ml_auc, "ci_low": ml_auc_ci[0], "ci_high": ml_auc_ci[1], "orientation": "higher is better", "analysis_role": "locked comparator"},
            {"estimand": "Mean 20-shot LLM AUROC", "estimate": llm_auc_mean, "ci_low": llm_auc_ci[0], "ci_high": llm_auc_ci[1], "orientation": "higher is better", "analysis_role": "prespecified primary LLM"},
            {"estimand": "Paired delta AUROC: LLM minus ExtraTrees", "estimate": delta_auc, "ci_low": delta_auc_ci[0], "ci_high": delta_auc_ci[1], "orientation": "positive favours LLM", "analysis_role": "primary paired contrast"},
            {"estimand": "ExtraTrees Brier", "estimate": ml_brier, "ci_low": ml_brier_ci[0], "ci_high": ml_brier_ci[1], "orientation": "lower is better", "analysis_role": "locked comparator"},
            {"estimand": "Mean 20-shot LLM Brier", "estimate": llm_brier_mean, "ci_low": llm_brier_ci[0], "ci_high": llm_brier_ci[1], "orientation": "lower is better", "analysis_role": "secondary probability quality"},
            {"estimand": "Brier improvement: ExtraTrees minus LLM", "estimate": brier_improvement, "ci_low": brier_improvement_ci[0], "ci_high": brier_improvement_ci[1], "orientation": "positive favours LLM", "analysis_role": "secondary paired contrast"},
            {"estimand": "Replicate-averaged LLM AUROC", "estimate": llm_avg_auc, "ci_low": np.nan, "ci_high": np.nan, "orientation": "higher is better", "analysis_role": "descriptive patient-level display only"},
            {"estimand": "Replicate-averaged LLM Brier", "estimate": llm_avg_brier, "ci_low": np.nan, "ci_high": np.nan, "orientation": "lower is better", "analysis_role": "descriptive patient-level display only"},
            {"estimand": "Spearman probability correlation", "estimate": spearman_rho, "ci_low": np.nan, "ci_high": np.nan, "orientation": "descriptive", "analysis_role": "descriptive patient-level agreement"},
            {"estimand": "Threshold classification disagreement count", "estimate": disagreement_n, "ci_low": np.nan, "ci_high": np.nan, "orientation": "descriptive count", "analysis_role": "descriptive patient-level agreement"},
        ]
    )

    patient_output.to_csv(PATIENT_CSV, index=False, encoding="utf-8-sig")
    replicate_output.to_csv(REPLICATE_CSV, index=False, encoding="utf-8-sig")
    bootstrap.to_csv(BOOTSTRAP_CSV, index=False, encoding="utf-8-sig")
    summary_output.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    pairing_qc = {
        "status": "PASS",
        "matching_method": "exact equality on four locked preoperative inputs after rounding numeric representation to 8 decimals",
        "matching_features": ["BMI", "baseline tumour size", "AST", "total bilirubin"],
        "llm_rows": int(len(features)),
        "ml_external_rows": int(len(ml)),
        "llm_duplicate_matching_keys": feature_duplicates,
        "ml_duplicate_matching_keys": ml_duplicates,
        "merge_counts": merge_counts,
        "matched_public_ids": int(paired["public_id"].nunique()),
        "matched_ml_records": int(paired["id"].nunique()),
        "outcome_disagreements": int((paired["mpr_binary"].astype(int) != paired["target_mpr90"].astype(int)).sum()),
        "ml_probability_missing": int(paired["predicted_probability_MPR"].isna().sum()),
        "ml_probability_out_of_range": int((~paired["predicted_probability_MPR"].between(0, 1)).sum()),
        "ml_threshold_label_inconsistency": int((source_ml_class.to_numpy() != recomputed_ml_class.to_numpy()).sum()),
        "llm_20shot_matrix_shape": list(llm_pivot.shape),
        "raw_ml_identifiers_exported": False,
    }
    PAIR_QC_JSON.write_text(json.dumps(pairing_qc, indent=2, ensure_ascii=False), encoding="utf-8")

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
            "legend.fontsize": 6,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": 600,
        }
    )
    dark = "#484878"
    pink = "#F0C0CC"
    pink_edge = "#D99AAF"
    grey = "#A8A8A8"
    light = "#E4E4F0"
    grid = "#E8E8E8"
    text = "#303030"

    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(7.2, 2.75))
    fig.subplots_adjust(left=0.145, right=0.98, bottom=0.24, top=0.82, wspace=0.62)

    # Panel a: paired effect estimates.
    style_axis(axa)
    axa.grid(axis="x", color=grid, linewidth=0.55, zorder=0)
    axa.axvline(0, color="#909090", linestyle=(0, (3, 3)), linewidth=0.8, zorder=1)
    y_positions = np.array([1.0, 0.0])
    replicate_values = [replicate_output["delta_auc_llm_minus_ml"].to_numpy(), replicate_output["brier_improvement_ml_minus_llm"].to_numpy()]
    for y0, values in zip(y_positions, replicate_values):
        jitter = np.linspace(-0.16, 0.16, len(values))
        axa.scatter(values, np.full(len(values), y0) + jitter, s=10, color="#D8D8D8", edgecolor="white", linewidth=0.3, zorder=2)
    estimates = np.array([delta_auc, brier_improvement])
    lows = np.array([delta_auc_ci[0], brier_improvement_ci[0]])
    highs = np.array([delta_auc_ci[1], brier_improvement_ci[1]])
    for y0, estimate, low_ci, high_ci in zip(y_positions, estimates, lows, highs):
        axa.errorbar(
            estimate,
            y0,
            xerr=[[estimate - low_ci], [high_ci - estimate]],
            fmt="o",
            color=dark,
            ecolor=dark,
            markersize=4.2,
            elinewidth=1.0,
            capsize=2.2,
            zorder=4,
        )
    extent = max(abs(np.r_[lows, highs, replicate_values[0], replicate_values[1]]).max() + 0.025, 0.10)
    axa.set_xlim(-extent, extent)
    axa.set_ylim(-0.48, 1.48)
    axa.set_yticks(y_positions, ["AUROC Δ\n(LLM − ML)", "Brier improvement\n(ML − LLM)"])
    axa.set_xlabel("Oriented paired difference\n(positive favours LLM)")
    axa.set_title("Paired performance differences", loc="left", fontweight="bold")
    panel_label(axa, "a")

    # Panel b: descriptive patient-level agreement.
    style_axis(axb)
    axb.grid(color=grid, linewidth=0.5, zorder=0)
    axb.plot([0, 1], [0, 1], color="#B8B8B8", linestyle=(0, (3, 3)), linewidth=0.8, zorder=1)
    axb.axvline(ml_threshold, color=pink_edge, linestyle=(0, (2, 2)), linewidth=0.75, zorder=1)
    axb.axhline(0.5, color=pink_edge, linestyle=(0, (2, 2)), linewidth=0.75, zorder=1)
    non_event = y == 0
    event = y == 1
    axb.scatter(ml_probability_array[non_event], llm_probability_mean[non_event], s=18, facecolor="#D8D8D8", edgecolor="white", linewidth=0.45, label="Non-MPR", zorder=3)
    axb.scatter(ml_probability_array[event], llm_probability_mean[event], s=20, facecolor=dark, edgecolor="white", linewidth=0.45, label="MPR", zorder=4)
    axb.set_xlim(0, 1)
    axb.set_ylim(0, 1)
    axb.set_xlabel("Locked ExtraTrees probability")
    axb.set_ylabel("Mean 20-shot LLM probability")
    axb.set_title("Patient-level probability agreement", loc="left", fontweight="bold")
    axb.text(0.97, 0.04, f"Spearman ρ = {spearman_rho:.2f}", transform=axb.transAxes, ha="right", va="bottom", fontsize=6.1, color=text)
    axb.legend(loc="upper left", handletextpad=1.2, borderpad=0.1, labelspacing=0.45)
    panel_label(axb, "b")

    # Panel c: descriptive overlap of correctness.
    cmap = LinearSegmentedColormap.from_list("correctness", ["#F3F3F6", "#B4C0E4", dark])
    image = axc.imshow(correctness_matrix, cmap=cmap, vmin=0, vmax=int(correctness_matrix.max()), aspect="auto")
    for row in range(2):
        for col in range(2):
            count = int(correctness_matrix[row, col])
            colour = "white" if count > correctness_matrix.max() * 0.55 else text
            axc.text(col, row, f"{count}\n({count / 57:.0%})", ha="center", va="center", fontsize=7, color=colour, fontweight="bold")
    axc.set_xticks([0, 1], ["No", "Yes"])
    axc.set_yticks([0, 1], ["No", "Yes"])
    axc.set_xlabel("ExtraTrees correct")
    axc.set_ylabel("LLM mean correct")
    axc.set_title("Threshold-level correctness overlap", loc="left", fontweight="bold")
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
    figure_outputs = []
    for suffix, kwargs in [
        (".svg", {}),
        (".pdf", {}),
        (".tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
        (".png", {"dpi": 300}),
    ]:
        path = Path(str(STEM) + suffix)
        fig.savefig(path, facecolor="white", **kwargs)
        figure_outputs.append(path)
    plt.close(fig)

    both_wrong = int(correctness_matrix[0, 0])
    ml_only = int(correctness_matrix[0, 1])
    llm_only = int(correctness_matrix[1, 0])
    both_correct = int(correctness_matrix[1, 1])

    results = f"""# Paired ML–LLM external-validation Results

## English manuscript-ready draft

### Patient-level pairing showed no clear performance difference between the locked ExtraTrees model and the prespecified LLM analysis

All 57 external-validation patients were matched one-to-one between the locked machine-learning and LLM analyses, with complete agreement in outcome labels (25 MPR and 32 non-MPR). The locked ExtraTrees model achieved an AUROC of {ml_auc:.3f}, whereas the mean AUROC across the 20 prespecified 20-shot LLM prompt replicates was {llm_auc_mean:.3f}. In a paired hierarchical bootstrap that resampled patients within outcome strata and prompt replicates over 10,000 iterations, the AUROC difference for LLM minus ExtraTrees was {delta_auc:+.3f} (95% CI, {delta_auc_ci[0]:+.3f} to {delta_auc_ci[1]:+.3f}; Fig. 4a). The corresponding ExtraTrees and mean LLM Brier scores were {ml_brier:.3f} and {llm_brier_mean:.3f}, respectively; the oriented Brier improvement for ExtraTrees minus LLM was {brier_improvement:+.3f} (95% CI, {brier_improvement_ci[0]:+.3f} to {brier_improvement_ci[1]:+.3f}). Both intervals spanned zero, providing no clear evidence that either approach performed better, while not establishing equivalence.

### Aggregate similarity concealed only moderate patient-level concordance

For descriptive patient-level analysis, LLM probabilities were averaged across the 20 prompt replicates to obtain one probability per patient. These probabilities correlated only moderately with the locked ExtraTrees probabilities (Spearman ρ={spearman_rho:.2f}; Fig. 4b). Using the locked ExtraTrees threshold of {ml_threshold:.3f} and an LLM threshold of 0.5, the two methods assigned different binary classifications to {disagreement_n} of 57 patients. Both methods classified {both_correct} patients correctly and {both_wrong} incorrectly; ExtraTrees alone was correct for {ml_only} patients, whereas the replicate-averaged LLM alone was correct for {llm_only} patients (Fig. 4c). This partial error overlap is compatible with differences in patient-level scoring, but it does not establish that an ensemble would improve performance because no combined model was trained or evaluated in an independent cohort.

## 中文证据边界

- 主比较仍然是锁定 ExtraTrees 与预设 20-shot LLM，不改用 full-shot，也不挑选表现最好的 prompt replicate。
- AUROC 主效应定义为“20个 LLM replicate 的平均 AUROC − ExtraTrees AUROC”。
- Brier 差异为“ExtraTrees Brier − LLM 平均 Brier”，因此正值才表示 LLM 更好。
- 两个差异区间均跨越0：可以写“未观察到明确差异”，不能写“等效”。
- 概率散点和正确性矩阵使用20次 LLM 概率均值，仅用于稳定的患者级描述，不替代主分析估计量。
- 错误不完全重叠不能直接证明集成增益；若要研究组合模型，必须在开发集预先定义并在独立数据上验证。

## Terminology ledger

| Canonical term | Definition |
|---|---|
| locked ExtraTrees model | Development-selected conventional machine-learning comparator |
| prespecified 20-shot LLM condition | Primary LLM analysis with 20 demonstration examples |
| prompt replicate | One complete prediction run using a sampled demonstration set |
| replicate-averaged LLM probability | Patient-level mean probability across 20 prompt replicates, used descriptively |
| MPR | Major pathological response, pathological necrosis ≥90% |
"""
    RESULTS_MD.write_text(results, encoding="utf-8")

    methods = f"""# Statistical analysis for paired ML–LLM comparison

The independent analysis unit was the patient. The locked ExtraTrees and LLM external-validation records were paired by exact equality of the four locked preoperative inputs used by both analyses (BMI, baseline tumour size, AST and total bilirubin). These four-variable keys were unique in both datasets; all 57 patients matched one-to-one, and all binary outcome labels agreed. Raw machine-learning identifiers were not exported to the paired Source Data.

The primary LLM estimand was the expected performance of the prespecified 20-shot condition over demonstration sampling, represented by the mean metric across 20 complete prompt replicates. The primary paired contrast was the mean LLM AUROC minus the locked ExtraTrees AUROC. Brier score was analysed as a secondary probability-quality endpoint and oriented as ExtraTrees Brier score minus mean LLM Brier score, so positive values favoured the LLM for both displayed contrasts.

Uncertainty was quantified using a hierarchical stratified nonparametric bootstrap with {N_BOOT:,} resamples and random seed {BOOT_SEED}. In each resample, patients were sampled with replacement separately within the MPR and non-MPR strata, preserving the observed stratum sizes, and the 20 prompt replicates were independently sampled with replacement. ExtraTrees performance and the mean LLM performance across the sampled prompt replicates were recomputed on the same sampled patients. Percentile 95% confidence intervals were obtained from the 2.5th and 97.5th percentiles of the paired bootstrap distribution. Confidence intervals spanning zero were interpreted as absence of a clear performance difference, not as evidence of equivalence. No dichotomous null-hypothesis significance test or multiplicity-adjusted P value was used.

For descriptive patient-level displays, each patient’s LLM probability was averaged across the 20 prompt replicates. Spearman rank correlation summarized probability concordance. Threshold-level correctness used the development-locked ExtraTrees threshold ({ml_threshold:.6f}) and an LLM threshold of 0.5. These analyses were descriptive and were not used to fit or select a combined model.

The paired analysis was run in Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} using NumPy {np.__version__}, pandas {pd.__version__} and matplotlib {mpl.__version__}. Custom auditable code implemented AUROC, Brier score and the hierarchical paired bootstrap.
"""
    METHODS_MD.write_text(methods, encoding="utf-8")

    legend = f"""# Figure 4 legend — paired external validation

**Figure 4 | Paired comparison of the locked ExtraTrees model and prespecified 20-shot LLM condition.** **a,** Paired external-validation performance differences. Small grey points show the 20 individual prompt-replicate contrasts. Large points show the prespecified summary contrasts, and error bars show percentile 95% confidence intervals from 10,000 hierarchical stratified bootstrap resamples of patients and prompt replicates. AUROC Δ is mean LLM AUROC minus ExtraTrees AUROC; Brier improvement is ExtraTrees Brier minus mean LLM Brier, so positive values favour the LLM for both rows. **b,** Locked ExtraTrees probabilities versus patient-level LLM probabilities averaged across the 20 prompt replicates. Points denote patients and are coloured by the observed MPR outcome; dashed pink lines denote the locked ExtraTrees threshold ({ml_threshold:.3f}) and the LLM threshold (0.5), and the grey diagonal denotes equal probabilities. Spearman ρ={spearman_rho:.2f}. **c,** Descriptive overlap of threshold-level correct classifications using the same thresholds; cells show patient counts and percentages. The independent unit was the patient; n=57 (25 MPR and 32 non-MPR). All patients were paired one-to-one. Replicate-averaged LLM probabilities in b and c were used only for descriptive patient-level visualization and were not substituted for the primary prompt-replicate estimand.
"""
    LEGEND_MD.write_text(legend, encoding="utf-8")

    primary_report = pd.read_csv(ML_PRIMARY_REPORT)
    primary_ml = primary_report.loc[primary_report["model"].eq("ExtraTrees")].iloc[0]
    detailed = pd.read_csv(ML_DETAILED)
    detailed_ml = detailed.loc[detailed["dataset"].eq("external_locked_validation")].iloc[0]
    consistency = f"""# ML results consistency audit

## Finding requiring author decision before Figure 2 is edited

The locked external AUROC point estimate is consistent at {ml_auc:.4f}, but three different 95% confidence intervals appear in the supplied ML deliverables:

- Current Figure 2 panel c SVG: 0.526–0.804.
- Primary report CSV: {float(primary_ml['external_AUC_95CI_low']):.3f}–{float(primary_ml['external_AUC_95CI_high']):.3f}.
- Detailed performance CSV: {float(detailed_ml['AUC_95CI_low']):.3f}–{float(detailed_ml['AUC_95CI_high']):.3f}.
- New paired patient-bootstrap interval generated for the comparison package: {ml_auc_ci[0]:.3f}–{ml_auc_ci[1]:.3f}.

These intervals likely arise from different bootstrap seeds, resample counts or interval implementations, but the supplied files do not identify which interval is intended as the single authoritative value. This is a cross-file reporting inconsistency (P1) rather than a change in the AUROC estimate.

No existing Figure 2 file or ML result was modified. Before submission, the author should choose one predefined interval procedure and authorize synchronized replacement across Figure 2, Results, tables and Source Data. The paired Figure 4 avoids this ambiguity by reporting the newly specified hierarchical paired-bootstrap difference interval.
"""
    CONSISTENCY_MD.write_text(consistency, encoding="utf-8")

    generated = [
        *figure_outputs,
        Path(str(STEM) + ".alignment.json"),
        Path(str(STEM) + ".alignment.svg"),
        PATIENT_CSV,
        REPLICATE_CSV,
        BOOTSTRAP_CSV,
        SUMMARY_CSV,
        PAIR_QC_JSON,
        RESULTS_MD,
        METHODS_MD,
        LEGEND_MD,
        CONSISTENCY_MD,
    ]
    record = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "backend": "Python/matplotlib",
        "analysis": {
            "independent_unit": "patient",
            "n": 57,
            "events": 25,
            "non_events": 32,
            "primary_llm_condition": "20-shot",
            "prompt_replicates": 20,
            "bootstrap_resamples": N_BOOT,
            "bootstrap_seed": BOOT_SEED,
            "pairing_qc": pairing_qc,
        },
        "inputs": {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in input_paths},
        "outputs": {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in generated},
        "overwrite_policy": "all planned outputs were asserted absent before generation",
    }
    RECORD_JSON.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "delta_auc": {"estimate": delta_auc, "ci": delta_auc_ci},
                "brier_improvement": {"estimate": brier_improvement, "ci": brier_improvement_ci},
                "spearman_rho": spearman_rho,
                "correctness_matrix": correctness_matrix.tolist(),
                "outputs": [str(path) for path in generated + [RECORD_JSON]],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
