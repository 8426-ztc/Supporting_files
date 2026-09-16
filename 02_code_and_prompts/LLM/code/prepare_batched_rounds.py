"""Prepare one leakage-audited cohort-batched prompt per shot/replicate round."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from llm_icl_benchmark import FEATURES, TARGET_COL, _format_case, _system_prompt


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Batched run directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def build_prompt(
    exemplars: pd.DataFrame,
    development_queries: pd.DataFrame,
    validation_queries: pd.DataFrame,
) -> str:
    if exemplars.empty:
        demo_block = "No labeled demonstrations are provided. This is the zero-shot condition."
    else:
        demos = "\n".join(
            f"{index + 1}. {_format_case(row, include_label=True)}"
            for index, (_, row) in enumerate(exemplars.iterrows())
        )
        demo_block = "Labeled demonstrations:\n" + demos
    development_block = "\n".join(
        f"{index + 1}. {_format_case(row)}"
        for index, (_, row) in enumerate(development_queries.iterrows())
    )
    validation_block = "\n".join(
        f"{index + 1 + len(development_queries)}. {_format_case(row)}"
        for index, (_, row) in enumerate(validation_queries.iterrows())
    )
    return f"""{demo_block}

Unlabeled development query cases:
{development_block}

Unlabeled external-validation query cases:
{validation_block}

Return predictions for all 166 query cases in the displayed order. Return only the required JSON array."""


def prepare(source_run: Path, output_run: Path) -> dict:
    source_operator = source_run / "operator"
    dev_path = source_operator / "development_labeled_pool.csv"
    val_path = source_operator / "validation_features_BLINDED.csv"
    exemplar_path = source_operator / "exemplars.csv"
    manifest_path = source_operator / "experiment_manifest.json"
    addendum_path = source_run.parent / "PREREGISTRATION_ADDENDUM_06.md"
    for path in [dev_path, val_path, exemplar_path, manifest_path, addendum_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    dev = pd.read_csv(dev_path).sort_values("public_id").reset_index(drop=True)
    val = pd.read_csv(val_path).sort_values("public_id").reset_index(drop=True)
    exemplar_table = pd.read_csv(exemplar_path)
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    conditions = source_manifest["conditions"]
    if len(dev) != 109 or len(val) != 57 or len(conditions) != 122:
        raise ValueError("Frozen cohort or condition count changed")
    if dev["public_id"].duplicated().any() or val["public_id"].duplicated().any():
        raise ValueError("Duplicate pseudonymous ID within a cohort")
    if set(dev["public_id"]) & set(val["public_id"]):
        raise ValueError("Development and validation pseudonyms overlap")
    if list(dev.columns) != ["public_id", *FEATURES, TARGET_COL]:
        raise ValueError("Unexpected development operator schema")
    if list(val.columns) != ["public_id", *FEATURES]:
        raise ValueError("Unexpected validation operator schema")

    require_empty(output_run)
    operator_dir = output_run / "operator"
    private_dir = output_run / "private"
    prediction_dir = output_run / "predictions"
    operator_dir.mkdir()
    private_dir.mkdir()
    prediction_dir.mkdir()

    query_dev = dev[["public_id", *FEATURES]].copy()
    query_val = val[["public_id", *FEATURES]].copy()
    prompt_path = operator_dir / "prompts.jsonl"
    expected_rows = []
    round_rows = []
    with prompt_path.open("w", encoding="utf-8") as handle:
        for round_number, condition in enumerate(conditions, start=1):
            condition_id = condition["condition_id"]
            shot_size = int(condition["shot_size"])
            replicate = int(condition["replicate"])
            exemplars = exemplar_table.loc[
                exemplar_table["condition_id"] == condition_id,
                ["public_id", *FEATURES, TARGET_COL],
            ].reset_index(drop=True)
            if len(exemplars) != shot_size:
                raise ValueError(f"Demonstration count mismatch: {condition_id}")
            demonstration_ids = set(exemplars["public_id"])
            user_prompt = build_prompt(exemplars, query_dev, query_val)
            development_query_section = user_prompt.split(
                "Unlabeled development query cases:\n", 1
            )[1].split("\n\nUnlabeled external-validation query cases:", 1)[0]
            validation_query_section = user_prompt.split(
                "Unlabeled external-validation query cases:\n", 1
            )[1].split("\n\nReturn predictions", 1)[0]
            if "observed_label=" in development_query_section or "observed_label=" in validation_query_section:
                raise ValueError("A target label leaked into an unlabeled query section")

            record = {
                "condition_id": condition_id,
                "round_number": round_number,
                "shot_size": shot_size,
                "replicate": replicate,
                "query_count": 166,
                "development_query_count": 109,
                "validation_query_count": 57,
                "label_exposed_development_count": len(demonstration_ids),
                "system_prompt": _system_prompt(166),
                "user_prompt": user_prompt,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            round_rows.append(
                {
                    key: record[key]
                    for key in [
                        "condition_id",
                        "round_number",
                        "shot_size",
                        "replicate",
                        "query_count",
                        "development_query_count",
                        "validation_query_count",
                        "label_exposed_development_count",
                    ]
                }
            )
            for query_order, public_id in enumerate(
                list(query_dev["public_id"]) + list(query_val["public_id"]), start=1
            ):
                cohort = "development" if query_order <= len(query_dev) else "validation"
                expected_rows.append(
                    {
                        "condition_id": condition_id,
                        "round_number": round_number,
                        "shot_size": shot_size,
                        "replicate": replicate,
                        "query_order": query_order,
                        "cohort": cohort,
                        "public_id": public_id,
                        "label_exposed": int(
                            cohort == "development" and public_id in demonstration_ids
                        ),
                        "eligible_for_development_metric": int(
                            cohort == "development" and public_id not in demonstration_ids
                        ),
                    }
                )

    expected_path = operator_dir / "expected_predictions.csv"
    rounds_path = operator_dir / "rounds.csv"
    demonstrations_path = operator_dir / "demonstrations.csv"
    pd.DataFrame(expected_rows).to_csv(expected_path, index=False)
    pd.DataFrame(round_rows).to_csv(rounds_path, index=False)
    exemplar_table.to_csv(demonstrations_path, index=False)
    query_dev.to_csv(operator_dir / "development_features_UNLABELED.csv", index=False)
    query_val.to_csv(operator_dir / "validation_features_BLINDED.csv", index=False)
    if len(expected_rows) != 122 * 166:
        raise ValueError("Expected-output count must equal 20,252")
    expected_frame = pd.DataFrame(expected_rows)
    if expected_frame.duplicated(["condition_id", "public_id"]).any():
        raise ValueError("Duplicate condition/patient output key")
    if expected_frame.query("cohort == 'validation'")["label_exposed"].sum() != 0:
        raise ValueError("A validation case was marked label-exposed")

    source_private = source_run / "private"
    for name in ["validation_ground_truth_LOCKED.csv", "leakage_audit_LOCKED.json"]:
        shutil.copy2(source_private / name, private_dir / name)
    source_predictions = source_run / "predictions"
    for name in [
        "baseline_logistic.csv",
        "baseline_logistic.csv.freeze.json",
        "baseline_model_audit.json",
    ]:
        shutil.copy2(source_predictions / name, prediction_dir / name)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_OUTCOME_BLIND_BATCHED_ROUNDS",
        "source_run": source_run.name,
        "source_run_predictions_used": False,
        "superseded_patientwise_predictions": 37,
        "rounds": 122,
        "queries_per_round": 166,
        "expected_patient_predictions": 20_252,
        "development_n": 109,
        "validation_n": 57,
        "primary_shot": 20,
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "validation_outcomes_accessed": False,
        "batched_transductive_design": True,
        "validation_label_exposure": 0,
        "development_label_exposure_policy": (
            "retain and flag demonstration-member outputs; exclude them from development metrics"
        ),
        "preregistration_addendum": addendum_path.name,
        "preregistration_addendum_sha256": sha256_file(addendum_path),
        "preparation_code_sha256": sha256_file(Path(__file__).resolve()),
        "source_artifacts_sha256": {
            path.name: sha256_file(path)
            for path in [dev_path, val_path, exemplar_path, manifest_path]
        },
        "operator_artifacts_sha256": {
            path.name: sha256_file(path)
            for path in sorted(operator_dir.iterdir())
            if path.is_file()
        },
        "private_artifacts_sha256": {
            path.name: sha256_file(path)
            for path in sorted(private_dir.iterdir())
            if path.is_file()
        },
        "baseline_artifacts_sha256": {
            path.name: sha256_file(path)
            for path in sorted(prediction_dir.iterdir())
            if path.is_file()
        },
    }
    manifest_output = private_dir / "batched_run_manifest_LOCKED.json"
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "rounds": manifest["rounds"],
                "expected_patient_predictions": manifest["expected_patient_predictions"],
                "validation_label_exposure": 0,
                "output_run": str(output_run),
            },
            ensure_ascii=False,
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_run.resolve(), args.output_run.resolve())


if __name__ == "__main__":
    main()

