"""Single authorized unblinding and preregistered analysis of frozen batched rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from audit_codex_inference import verify_chain
from audit_round_inference import audit
from llm_icl_benchmark import (
    RANDOM_SEED,
    _hierarchical_shot_summary,
    _metric_row,
    _metrics,
)


THRESHOLD = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(run_dir: Path, bootstrap_replicates: int) -> dict:
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    prediction_dir = run_dir / "predictions"
    private_dir = run_dir / "private"
    output_dir = run_dir / "evaluation"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite a prior unblinded evaluation")
    predictions_path = prediction_dir / "llm_predictions_VALIDATED.csv"
    freeze_path = predictions_path.with_suffix(predictions_path.suffix + ".freeze.json")
    if not predictions_path.is_file() or not freeze_path.is_file():
        raise ValueError("Complete frozen batched predictions are required before unblinding")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN_OUTCOME_BLIND" or sha256_file(predictions_path) != freeze[
        "predictions_sha256"
    ]:
        raise ValueError("Prediction freeze is missing or invalid")
    if freeze["rounds"] != 122 or freeze["patient_outputs"] != 20_252:
        raise ValueError("Frozen prediction dimensions are incomplete")
    audit_result = audit(run_dir, "gpt-5.6-sol", "high")
    if audit_result["remaining_rounds"] != 0 or audit_result["audit_status"] != "PASS":
        raise ValueError("Final outcome-blind inference audit did not pass")
    verify_chain(prediction_dir / "prediction_freeze_ledger.jsonl")

    validation_truth_path = private_dir / "validation_ground_truth_LOCKED.csv"
    development_truth_path = private_dir / "development_ground_truth_LOCKED.csv"
    development_truth_audit = private_dir / "development_ground_truth_LOCKED.audit.json"
    development_lock = json.loads(development_truth_audit.read_text(encoding="utf-8"))
    if sha256_file(development_truth_path) != development_lock["output_sha256"]:
        raise ValueError("Locked development outcomes were modified")
    output_dir.mkdir()
    unblinding_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": "FIRST_AND_ONLY_VALIDATION_UNBLINDING",
        "predictions_sha256": sha256_file(predictions_path),
        "prediction_freeze_sha256": sha256_file(freeze_path),
        "validation_ground_truth_sha256": sha256_file(validation_truth_path),
        "development_ground_truth_sha256": sha256_file(development_truth_path),
        "completed_rounds": audit_result["completed_rounds"],
        "patient_outputs": audit_result["completed_patient_outputs"],
        "forbidden_tool_events": audit_result["forbidden_tool_events"],
        "evaluation_code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (private_dir / "unblinding_record_LOCKED.json").write_text(
        json.dumps(unblinding_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    predictions = pd.read_csv(predictions_path)
    validation_truth = pd.read_csv(validation_truth_path)[["public_id", "target_mpr90"]]
    development_truth = pd.read_csv(development_truth_path)
    truth = pd.concat(
        [
            development_truth.assign(cohort="development"),
            validation_truth.assign(cohort="validation"),
        ],
        ignore_index=True,
    )
    scored = predictions.merge(truth, on=["cohort", "public_id"], how="left", validate="many_to_one")
    if scored["target_mpr90"].isna().any():
        raise ValueError("A frozen prediction lacks a binary outcome after unblinding")
    scored["target_mpr90"] = scored["target_mpr90"].astype(int)
    scored.to_csv(output_dir / "scored_predictions.csv", index=False)

    baseline_path = prediction_dir / "baseline_logistic.csv"
    baseline_freeze_path = baseline_path.with_suffix(baseline_path.suffix + ".freeze.json")
    baseline_freeze = json.loads(baseline_freeze_path.read_text(encoding="utf-8"))
    if sha256_file(baseline_path) != baseline_freeze["sha256"]:
        raise ValueError("Frozen logistic baseline was modified")
    baseline = validation_truth.merge(
        pd.read_csv(baseline_path), on="public_id", how="inner", validate="one_to_one"
    ).sort_values("public_id")
    if len(baseline) != 57:
        raise ValueError("Frozen validation baseline does not contain 57 patients")
    y_validation = baseline["target_mpr90"].to_numpy(dtype=int)
    baseline_probability = baseline["prob_mpr90"].to_numpy(dtype=float)
    baseline_metrics = _metric_row(
        y_validation,
        baseline_probability,
        THRESHOLD,
        bootstrap_replicates,
        RANDOM_SEED,
    )
    pd.DataFrame([{"model": "logistic_regression", "n": 57, **baseline_metrics}]).to_csv(
        output_dir / "baseline_validation_metrics.csv", index=False
    )

    validation = scored.loc[scored["cohort"] == "validation"].copy()
    validation_condition_rows = []
    for index, (condition_id, group) in enumerate(
        validation.groupby("condition_id", sort=True), start=1
    ):
        group = group.sort_values("public_id")
        if len(group) != 57 or group["label_exposed"].astype(int).sum() != 0:
            raise ValueError("Validation condition is incomplete or label-exposed")
        validation_condition_rows.append(
            {
                "condition_id": condition_id,
                "shot_size": int(group["shot_size"].iloc[0]),
                "replicate": int(group["replicate"].iloc[0]),
                "n": len(group),
                **_metric_row(
                    group["target_mpr90"].to_numpy(dtype=int),
                    group["prob_mpr90"].to_numpy(dtype=float),
                    THRESHOLD,
                    bootstrap_replicates,
                    RANDOM_SEED + index,
                ),
            }
        )
    validation_condition = pd.DataFrame(validation_condition_rows)
    validation_condition.to_csv(output_dir / "validation_condition_metrics.csv", index=False)

    validation_shot_rows = []
    validation_ids = list(baseline["public_id"])
    for shot_size, shot_group in validation.groupby("shot_size", sort=True):
        replicate_probabilities = []
        for _, condition_group in shot_group.groupby("condition_id", sort=True):
            aligned = condition_group.set_index("public_id").loc[validation_ids]
            replicate_probabilities.append(aligned["prob_mpr90"].to_numpy(dtype=float))
        summary = _hierarchical_shot_summary(
            y_validation,
            baseline_probability,
            np.stack(replicate_probabilities),
            THRESHOLD,
            bootstrap_replicates,
            RANDOM_SEED + 200_000 + int(shot_size),
        )
        validation_shot_rows.append(
            {"shot_size": int(shot_size), "n": 57, **summary}
        )
    validation_shot = pd.DataFrame(validation_shot_rows)
    validation_shot.to_csv(output_dir / "validation_shot_summary.csv", index=False)

    development = scored.loc[scored["cohort"] == "development"].copy()
    development_condition_rows = []
    metric_names = list(_metrics(np.array([0, 1]), np.array([0.25, 0.75]), THRESHOLD))
    for index, (condition_id, group) in enumerate(
        development.groupby("condition_id", sort=True), start=1
    ):
        eligible = group.loc[group["eligible_for_development_metric"].astype(int) == 1].copy()
        shot_size = int(group["shot_size"].iloc[0])
        base = {
            "condition_id": condition_id,
            "shot_size": shot_size,
            "replicate": int(group["replicate"].iloc[0]),
            "n_total": len(group),
            "n_label_exposed_excluded": int(group["label_exposed"].astype(int).sum()),
            "n_evaluable": len(eligible),
        }
        if len(eligible) == 0:
            base["not_evaluable_reason"] = "all development patients were labeled demonstrations"
            for metric in metric_names:
                base[metric] = np.nan
                base[f"{metric}_ci_low"] = np.nan
                base[f"{metric}_ci_high"] = np.nan
        else:
            base["not_evaluable_reason"] = ""
            base.update(
                _metric_row(
                    eligible["target_mpr90"].to_numpy(dtype=int),
                    eligible["prob_mpr90"].to_numpy(dtype=float),
                    THRESHOLD,
                    bootstrap_replicates,
                    RANDOM_SEED + 500_000 + index,
                )
            )
        development_condition_rows.append(base)
    development_condition = pd.DataFrame(development_condition_rows)
    development_condition.to_csv(output_dir / "development_condition_metrics.csv", index=False)

    development_shot_rows = []
    for shot_size, group in development_condition.groupby("shot_size", sort=True):
        row = {
            "shot_size": int(shot_size),
            "prompt_replicates": len(group),
            "development_inference": "exploratory_non_demo_subset",
            "n_evaluable_min": int(group["n_evaluable"].min()),
            "n_evaluable_max": int(group["n_evaluable"].max()),
        }
        for metric in metric_names:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_prompt_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) == 1 else np.nan)
            )
            row[f"{metric}_min"] = float(values.min()) if len(values) else np.nan
            row[f"{metric}_max"] = float(values.max()) if len(values) else np.nan
        development_shot_rows.append(row)
    pd.DataFrame(development_shot_rows).to_csv(
        output_dir / "development_shot_summary.csv", index=False
    )

    primary = validation_shot.loc[validation_shot["shot_size"] == 20].iloc[0].to_dict()
    primary_record = {
        "primary_condition": "20-shot external validation",
        "primary_metric": "AUROC",
        "validation_n": 57,
        "prompt_replicates": int(primary["prompt_replicates"]),
        "llm_auroc_mean": float(primary["auroc_mean"]),
        "llm_auroc_ci_low": float(primary["auroc_ci_low"]),
        "llm_auroc_ci_high": float(primary["auroc_ci_high"]),
        "baseline_auroc": float(baseline_metrics["auroc"]),
        "auroc_delta_vs_baseline": float(primary["auroc_delta_vs_baseline"]),
        "auroc_delta_ci_low": float(primary["auroc_delta_ci_low"]),
        "auroc_delta_ci_high": float(primary["auroc_delta_ci_high"]),
        "bootstrap_replicates": bootstrap_replicates,
        "design_boundary": "cohort-batched transductive inference",
    }
    (output_dir / "primary_analysis.json").write_text(
        json.dumps(primary_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    evaluation_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE",
        "unblinding_record_sha256": sha256_file(
            private_dir / "unblinding_record_LOCKED.json"
        ),
        "evaluation_code_sha256": sha256_file(Path(__file__).resolve()),
        "bootstrap_replicates": bootstrap_replicates,
        "files_sha256": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return primary_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.run_dir.resolve(), args.bootstrap_replicates),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

