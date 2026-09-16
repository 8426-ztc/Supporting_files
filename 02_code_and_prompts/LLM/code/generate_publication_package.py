"""Create publication tables, text, source data, and workbook inputs after unblinding."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd


SHOT_ORDER = [0, 4, 8, 10, 20, 40, 80, 109]
METRICS = ["auroc", "auprc", "brier", "log_loss", "accuracy", "sensitivity", "specificity"]
PROHIBITED_EXACT_COLUMNS = {"original_id", "patient_id", "mpr", "status", "rfs"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


def json_safe_series(series: pd.Series) -> dict:
    return {
        str(key): (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
        for key, value in series.items()
    }


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}")


def build(run_dir: Path) -> dict:
    evaluation_dir = run_dir / "evaluation"
    prediction_dir = run_dir / "predictions"
    private_dir = run_dir / "private"
    publication_dir = run_dir / "publication"
    if publication_dir.exists() and any(publication_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing publication package")
    publication_dir.mkdir(exist_ok=True)
    tables_dir = publication_dir / "tables"
    source_dir = publication_dir / "source_data"
    workbook_source_dir = publication_dir / "workbook_source"
    figures_dir = publication_dir / "figures"
    for directory in (tables_dir, source_dir, workbook_source_dir, figures_dir):
        directory.mkdir()

    predictions_path = prediction_dir / "llm_predictions_VALIDATED.csv"
    freeze_path = predictions_path.with_suffix(predictions_path.suffix + ".freeze.json")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    require_hash(predictions_path, freeze["predictions_sha256"], "frozen predictions")
    if freeze["status"] != "FROZEN_OUTCOME_BLIND" or freeze["patient_outputs"] != 20_252:
        raise ValueError("Prediction freeze is not complete")

    evaluation_manifest_path = evaluation_dir / "evaluation_manifest.json"
    evaluation_manifest = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
    for name, expected in evaluation_manifest["files_sha256"].items():
        require_hash(evaluation_dir / name, expected, f"evaluation file {name}")

    unblinding_path = private_dir / "unblinding_record_LOCKED.json"
    require_hash(
        unblinding_path,
        evaluation_manifest["unblinding_record_sha256"],
        "unblinding record",
    )
    unblinding = json.loads(unblinding_path.read_text(encoding="utf-8"))
    if unblinding["event"] != "FIRST_AND_ONLY_VALIDATION_UNBLINDING":
        raise ValueError("Unexpected unblinding event")

    scored = pd.read_csv(evaluation_dir / "scored_predictions.csv")
    unsafe_columns = sorted(PROHIBITED_EXACT_COLUMNS.intersection(c.lower() for c in scored.columns))
    if unsafe_columns:
        raise ValueError(f"Prohibited columns present in scored predictions: {unsafe_columns}")
    if len(scored) != 20_252 or scored[["condition_id", "public_id"]].duplicated().any():
        raise ValueError("Scored predictions are incomplete or duplicated")
    if set(scored["shot_size"].astype(int)) != set(SHOT_ORDER):
        raise ValueError("Unexpected shot-size set")

    validation_shot = pd.read_csv(evaluation_dir / "validation_shot_summary.csv")
    validation_condition = pd.read_csv(evaluation_dir / "validation_condition_metrics.csv")
    development_shot = pd.read_csv(evaluation_dir / "development_shot_summary.csv")
    development_condition = pd.read_csv(evaluation_dir / "development_condition_metrics.csv")
    baseline = pd.read_csv(evaluation_dir / "baseline_validation_metrics.csv")
    primary = json.loads((evaluation_dir / "primary_analysis.json").read_text(encoding="utf-8"))
    if validation_shot["shot_size"].astype(int).tolist() != SHOT_ORDER:
        raise ValueError("Validation shot summary order or coverage is invalid")

    validation_table = validation_shot.copy()
    validation_table.insert(1, "analysis_role", ["primary" if int(x) == 20 else "exploratory" for x in validation_table["shot_size"]])
    validation_table.insert(3, "independent_unit", "patient")
    validation_table.insert(4, "uncertainty", "hierarchical stratified bootstrap over patients and prompt replicates; 1000 resamples")
    validation_table.to_csv(tables_dir / "Table1_validation_performance_by_shot.csv", index=False)
    validation_condition.to_csv(tables_dir / "TableS1_validation_condition_metrics.csv", index=False)
    development_condition.to_csv(tables_dir / "TableS2_development_condition_metrics.csv", index=False)
    development_shot.to_csv(tables_dir / "TableS3_development_performance_by_shot.csv", index=False)
    baseline.to_csv(tables_dir / "TableS4_logistic_baseline_validation.csv", index=False)

    figure_source = validation_shot.copy()
    for metric in METRICS:
        figure_source[f"baseline_{metric}"] = float(baseline.iloc[0][metric])
        figure_source[f"baseline_{metric}_ci_low"] = float(baseline.iloc[0][f"{metric}_ci_low"])
        figure_source[f"baseline_{metric}_ci_high"] = float(baseline.iloc[0][f"{metric}_ci_high"])
    figure_source.to_csv(source_dir / "Figure1_and_supplementary_source_data.csv", index=False)
    validation_condition.to_csv(source_dir / "Prompt_replicate_source_data.csv", index=False)

    safe_columns = [
        "condition_id", "round_number", "shot_size", "replicate", "query_order",
        "cohort", "public_id", "label_exposed", "eligible_for_development_metric",
        "pred_label", "prob_mpr90", "attempt", "model", "reasoning_effort",
        "prompt_sha256", "response_sha256", "events_sha256", "completed_utc",
        "target_mpr90",
    ]
    missing = [column for column in safe_columns if column not in scored.columns]
    if missing:
        raise ValueError(f"Missing workbook prediction columns: {missing}")
    workbook_columns = safe_columns.copy()
    workbook_rows = scored[workbook_columns].copy()
    workbook_rows["metric_evaluable"] = (
        (workbook_rows["cohort"] == "validation")
        | (workbook_rows["eligible_for_development_metric"].astype(int) == 1)
    ).astype(int)
    workbook_rows["threshold_consistent"] = (
        workbook_rows["pred_label"].astype(int)
        == (workbook_rows["prob_mpr90"].astype(float) >= 0.5).astype(int)
    ).astype(int)
    workbook_columns = [
        "condition_id", "round_number", "shot_size", "replicate", "query_order",
        "cohort", "public_id", "label_exposed", "metric_evaluable",
        "eligible_for_development_metric", "target_mpr90", "pred_label", "prob_mpr90",
        "threshold_consistent", "attempt", "model", "reasoning_effort", "prompt_sha256",
        "response_sha256", "events_sha256", "completed_utc",
    ]
    for shot in SHOT_ORDER:
        subset = workbook_rows.loc[workbook_rows["shot_size"].astype(int) == shot, workbook_columns]
        subset = subset.sort_values(["round_number", "query_order"], kind="stable")
        subset.to_csv(workbook_source_dir / f"shot_{shot}.csv", index=False)

    validation_unique = scored.loc[scored["cohort"] == "validation", ["public_id", "target_mpr90"]].drop_duplicates()
    development_unique = scored.loc[scored["cohort"] == "development", ["public_id", "target_mpr90"]].drop_duplicates()
    if len(validation_unique) != 57 or len(development_unique) != 109:
        raise ValueError("Unexpected cohort sizes after unblinding")

    latest_audit_path = sorted((run_dir / "audit").glob("inference_audit_*.json"))[-1]
    latest_audit = json.loads(latest_audit_path.read_text(encoding="utf-8"))
    audit_summary = {
        "experiment_status": "COMPLETE_AND_UNBLINDED_AFTER_FREEZE",
        "completed_rounds": int(unblinding["completed_rounds"]),
        "patient_outputs": int(unblinding["patient_outputs"]),
        "development_patients": int(len(development_unique)),
        "validation_patients": int(len(validation_unique)),
        "validation_positive": int(validation_unique["target_mpr90"].sum()),
        "validation_negative": int((1 - validation_unique["target_mpr90"]).sum()),
        "forbidden_tool_events": int(unblinding["forbidden_tool_events"]),
        "prediction_sha256": freeze["predictions_sha256"],
        "prediction_freeze_sha256": sha256_file(freeze_path),
        "unblinding_record_sha256": sha256_file(unblinding_path),
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
        "final_inference_audit_sha256": sha256_file(latest_audit_path),
        "execution_epochs": latest_audit.get("execution_epochs", []),
        "model": freeze["model"],
        "reasoning_effort": freeze["reasoning_effort"],
        "decision_threshold": 0.5,
        "bootstrap_replicates": int(evaluation_manifest["bootstrap_replicates"]),
        "validation_truth_first_access_utc": unblinding["timestamp_utc"],
        "prohibited_columns_absent": True,
        "continuous_mpr_status_rfs_not_exported": True,
    }
    (publication_dir / "audit_summary.json").write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    shot_summaries = {}
    for shot in SHOT_ORDER:
        val = validation_shot.loc[validation_shot["shot_size"].astype(int) == shot].iloc[0]
        dev = development_shot.loc[development_shot["shot_size"].astype(int) == shot].iloc[0]
        shot_summaries[str(shot)] = {
            "validation": json_safe_series(val),
            "development": json_safe_series(dev),
            "development_note": (
                "Not evaluable: all development labels were exposed as demonstrations."
                if shot == 109 else
                "Exploratory analysis restricted to non-demonstration development patients."
            ),
        }
    workbook_manifest = {
        "title": "MPR90 frozen cohort-batched LLM experiment",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shot_order": SHOT_ORDER,
        "audit": audit_summary,
        "primary": primary,
        "baseline": json_safe_series(baseline.iloc[0]),
        "shot_summaries": shot_summaries,
        "prediction_columns": workbook_columns,
    }
    (workbook_source_dir / "workbook_manifest.json").write_text(
        json.dumps(workbook_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    primary_text = (
        f"In the prespecified primary 20-shot external-validation analysis (n = 57 patients; "
        f"{int(primary['prompt_replicates'])} independent prompt replicates), mean AUROC was "
        f"{fmt(primary['llm_auroc_mean'])} (hierarchical bootstrap 95% CI "
        f"{fmt(primary['llm_auroc_ci_low'])}–{fmt(primary['llm_auroc_ci_high'])}). "
        f"The paired mean AUROC difference versus the frozen logistic-regression baseline was "
        f"{fmt(primary['auroc_delta_vs_baseline'])} (95% CI "
        f"{fmt(primary['auroc_delta_ci_low'])}–{fmt(primary['auroc_delta_ci_high'])})."
    )
    methods = f"""# Statistical analysis and reproducibility methods

The binary endpoint was MPR90. The independent clinical unit was the patient: 109 development patients and 57 held-out validation patients. Stochastic prompt replicates quantified inference variability and were not treated as additional patients. Predictions were generated in 122 prespecified shot-by-replicate conditions using {freeze['model']} with {freeze['reasoning_effort']} reasoning. Each condition was executed as one independent ephemeral cohort-batched call returning all 166 ordered pseudonymized patient predictions. The decision threshold (probability >= 0.5) was fixed before validation unblinding.

Development labels were available only for prespecified demonstrations. A development prediction was excluded from that condition's development performance calculation whenever its label had been exposed; the 109-shot development condition was therefore not evaluable. Validation labels remained inaccessible until all 20,252 prediction rows had passed schema, order, uniqueness, threshold-consistency, hash and forbidden-tool audits and the merged prediction file had been frozen by SHA-256.

The prespecified primary analysis was mean AUROC across the 20-shot validation prompt replicates. Secondary metrics were AUPRC, Brier score, log loss, accuracy, sensitivity and specificity. For each shot size, point estimates were means across independently generated prompt replicates. Ninety-five percent confidence intervals were percentile intervals from 1,000 hierarchical bootstrap resamples, stratified by binary outcome, with resampling at both the patient and prompt-replicate levels. The AUROC difference versus the frozen logistic-regression baseline used the same patient resample for both models. Other shot sizes and endpoints were treated as exploratory; no confirmatory multiplicity claim was made. Analyses used Python {platform.python_version()}, pandas {package_version('pandas')} and NumPy {package_version('numpy')} with a fixed random seed of 2026.
"""
    results = f"""# Results

{primary_text}

Across the eight prespecified shot sizes, validation performance is reported as the mean across prompt replicates with hierarchical 95% confidence intervals. AUROC, AUPRC, Brier score, log loss, accuracy, sensitivity and specificity are provided without selective omission in Table 1 and the Source Data. Development-set results are explicitly exploratory and exclude every patient used as a demonstration in the corresponding condition; no development estimate is reported for 109-shot prompting because all 109 development labels were exposed.

The experiment completed {audit_summary['completed_rounds']} conditions and {audit_summary['patient_outputs']:,} patient-level outputs with {audit_summary['forbidden_tool_events']} forbidden-tool events. Validation outcomes were first accessed only after the complete prediction file had been frozen. Failed transport or quota attempts were retained in the audit archive and were never selected in place of a completed response.
"""
    (publication_dir / "Statistical_analysis.md").write_text(methods, encoding="utf-8")
    (publication_dir / "Results.md").write_text(results, encoding="utf-8")

    figure_contract = """# Figure contract

Core conclusion: External-validation discrimination varies with demonstration count, with the prespecified 20-shot estimate interpreted against a frozen logistic-regression baseline.
Results-level question: How does cohort-batched LLM performance change across the prespecified shot sizes, and how uncertain is each estimate across patients and independent prompt replicates?
Figure archetype: quantitative single-panel forest/trend plot.
Target/output: Nature-family style; 89-mm single-column; editable SVG and PDF plus 600-dpi TIFF.
Backend: Python (matplotlib) exclusively.
Evidence hierarchy: 20-shot AUROC is primary; other shot sizes and AUPRC/Brier are exploratory supporting evidence.
Statistics: mean across prompt replicates; hierarchical stratified 95% bootstrap CI; n = 57 independent validation patients; prompt replicate count shown in Source Data.
Reviewer risks: small validation cohort, stochastic prompt variability, transductive cohort-batched inference boundary, and non-confirmatory interpretation of secondary shot sizes.
"""
    (publication_dir / "Figure_contract.md").write_text(figure_contract, encoding="utf-8")

    captions = """# Figure legends

## Fig. 1 | External-validation discrimination across demonstration counts
Mean AUROC is shown for each prespecified shot size; error bars denote hierarchical percentile-bootstrap 95% confidence intervals from 1,000 resamples over 57 independent validation patients and the available independent prompt replicates. The 20-shot condition is the prespecified primary analysis. The dashed line and shaded band indicate the frozen logistic-regression baseline estimate and its patient-level 95% bootstrap confidence interval. Other shot sizes are exploratory. Source data are provided as a Source Data file.

## Supplementary Fig. 1 | Precision-recall performance across demonstration counts
Mean AUPRC and hierarchical 95% confidence intervals are shown as in Fig. 1. The horizontal reference denotes the frozen logistic-regression baseline. All shot-size comparisons in this panel are supporting analyses.

## Supplementary Fig. 2 | Probabilistic error across demonstration counts
Mean Brier score and hierarchical 95% confidence intervals are shown as in Fig. 1; lower values indicate better probabilistic accuracy. The horizontal reference denotes the frozen logistic-regression baseline. All shot-size comparisons in this panel are supporting analyses.
"""
    (publication_dir / "Figure_legends.md").write_text(captions, encoding="utf-8")

    reviewer_notes = f"""# Reviewer-facing audit facts

- Formal run: cohort-batched frozen experiment only; the superseded patient-wise run is excluded.
- Prespecified conditions: 122; completed: {audit_summary['completed_rounds']}.
- Required outputs: 20,252; frozen outputs: {audit_summary['patient_outputs']}.
- Independent clinical units: 109 development patients and 57 held-out validation patients.
- Validation truth access: first and only after full audit and SHA-256 prediction freeze.
- Model configuration: {freeze['model']}, reasoning effort {freeze['reasoning_effort']}; any operational CLI epoch change was separately documented while scientific settings remained fixed.
- Leakage controls: pseudonymized identifiers, no continuous MPR/status/RFS export, no tools/files/web/memory/connectors in inference calls, exact ordered-ID/schema/threshold/hash checks, and explicit label-exposure flags.
- Development demonstrations: label_exposed = 1 and excluded from condition-specific development metrics.
- Failed attempts: preserved by hash; no answer shopping or prompt changes.
- Main limitation: cohort-batched inference is transductive at the cohort level and the validation cohort is modest; results do not establish clinical utility.
"""
    (publication_dir / "Reviewer_audit_notes.md").write_text(reviewer_notes, encoding="utf-8")

    manifest_files = {}
    for path in sorted(publication_dir.rglob("*")):
        if path.is_file() and path.name != "publication_manifest.json":
            manifest_files[str(path.relative_to(publication_dir)).replace("\\", "/")] = sha256_file(path)
    publication_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PUBLICATION_PACKAGE_PREPARED",
        "source_prediction_sha256": freeze["predictions_sha256"],
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "python": sys.version,
        "files_sha256": manifest_files,
    }
    (publication_dir / "publication_manifest.json").write_text(
        json.dumps(publication_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "status": publication_manifest["status"],
        "publication_dir": str(publication_dir),
        "files": len(manifest_files) + 1,
        "primary": primary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run_dir.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
