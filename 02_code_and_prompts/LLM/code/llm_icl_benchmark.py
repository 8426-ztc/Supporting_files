"""Leakage-resistant benchmark for four-feature MPR90 prediction.

The workflow is deliberately split into four commands:

1. ``prepare`` reads outcomes and creates separated operator/private packages.
2. ``baseline`` trains a fixed logistic-regression baseline from operator files.
3. ``validate-predictions`` validates LLM outputs without reading outcomes.
4. ``evaluate`` verifies file integrity and performs the final unblinding.

No command calls an LLM. Only the ``operator`` directory may be supplied to an
approved model endpoint; the ``private`` directory must remain with the data
custodian until every prediction is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import platform
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 2026
DEFAULT_DATA_FILE = Path("zh.csv")
DEFAULT_RUN_DIR = Path("llm_evaluation_pipeline")
FEATURES = ["bmi", "tumorsizeb", "ast", "tbil"]
TARGET_COL = "target_mpr90"
DEV_CENTER = "yz"
VAL_CENTER = "zy"
SHOT_SIZES = [0, 4, 8, 10, 20, 40, 80]
REPLICATES = 20
# One validation patient per prompt prevents predictions from depending on the
# composition or order of other validation patients.
VALIDATION_BATCH_SIZE = 1
BOOTSTRAP_REPLICATES = 1000

FEATURE_LABELS = {
    "bmi": "BMI (kg/m^2)",
    "tumorsizeb": "baseline tumor maximum diameter (mm)",
    "ast": "AST (U/L)",
    "tbil": "TBIL (umol/L)",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _freeze_file(path: Path) -> None:
    _write_json(
        path.with_suffix(path.suffix + ".freeze.json"),
        {"frozen_utc": _utc_now(), "sha256": _sha256(path)},
    )


def _verify_frozen_file(path: Path) -> None:
    freeze_path = path.with_suffix(path.suffix + ".freeze.json")
    if not freeze_path.exists():
        raise ValueError(f"Missing prediction freeze record: {freeze_path}")
    record = json.loads(freeze_path.read_text(encoding="utf-8"))
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"Prediction file changed after freezing: {path}")


def _append_ledger(run_dir: Path, event: str, details: dict[str, object]) -> None:
    ledger = run_dir / "private" / "run_ledger_LOCKED.jsonl"
    previous_hash = "GENESIS"
    if ledger.exists():
        lines = [line for line in ledger.read_text(encoding="utf-8").splitlines() if line]
        if lines:
            previous_hash = json.loads(lines[-1])["record_hash"]
    record = {
        "timestamp_utc": _utc_now(),
        "event": event,
        "analysis_code_sha256": _sha256(Path(__file__).resolve()),
        "previous_record_hash": previous_hash,
        "details": details,
    }
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
    record["record_hash"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _verify_ledger(run_dir: Path) -> None:
    ledger = run_dir / "private" / "run_ledger_LOCKED.jsonl"
    previous_hash = "GENESIS"
    for number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), start=1):
        record = json.loads(line)
        record_hash = record.pop("record_hash")
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if record["previous_record_hash"] != previous_hash:
            raise ValueError(f"Broken ledger chain at record {number}.")
        if hashlib.sha256(encoded.encode("utf-8")).hexdigest() != record_hash:
            raise ValueError(f"Modified ledger record at line {number}.")
        previous_hash = record_hash


def _require_new_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {path}. Use a new directory so stale "
            "or previously unblinded files cannot be mixed into this run."
        )
    path.mkdir(parents=True, exist_ok=True)


def _public_id(raw_id: object, center: str, key: bytes, prefix: str) -> str:
    message = f"{center}|{raw_id}".encode("utf-8")
    token = hmac.new(key, message, hashlib.sha256).hexdigest()[:16]
    return f"{prefix}-{token}"


def _feature_fingerprints(frame: pd.DataFrame) -> set[str]:
    canonical = frame[FEATURES].round(8).astype("string").fillna("<NA>")
    return {
        hashlib.sha256("|".join(row).encode("utf-8")).hexdigest()
        for row in canonical.itertuples(index=False, name=None)
    }


def _load_and_audit_cohorts(data_file: Path, subject_id_col: str):
    source = pd.read_csv(data_file)
    required = {subject_id_col, "center", "mpr", *FEATURES}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if source[subject_id_col].isna().any():
        raise ValueError(f"{subject_id_col} contains missing values.")
    unexpected_centers = sorted(
        set(source["center"].dropna().astype(str)).difference({DEV_CENTER, VAL_CENTER})
    )
    if source["center"].isna().any() or unexpected_centers:
        raise ValueError(
            f"Center must contain only {DEV_CENTER!r} and {VAL_CENTER!r}; "
            f"unexpected values: {unexpected_centers}."
        )

    duplicated_subjects = source[subject_id_col].astype(str).duplicated(keep=False)
    if duplicated_subjects.any():
        examples = source.loc[duplicated_subjects, subject_id_col].astype(str).unique()[:5]
        raise ValueError(
            f"Subject identifiers are not globally unique; examples: {examples.tolist()}. "
            "Pass a globally stable subject identifier column before claiming a leakage-free split."
        )

    numeric_outcome = pd.to_numeric(source["mpr"], errors="coerce")
    invalid_outcome = source["mpr"].notna() & numeric_outcome.isna()
    if invalid_outcome.any():
        raise ValueError("mpr contains non-numeric, non-missing values.")
    eligible = numeric_outcome.notna()
    frame = source.loc[eligible].copy()
    frame["mpr"] = numeric_outcome.loc[eligible]
    if frame.empty:
        raise ValueError("No rows have a non-missing mpr outcome.")

    for feature in FEATURES:
        if feature == "ast" and frame[feature].dtype == object:
            frame[feature] = frame[feature].astype(str).str.replace("..", ".", regex=False)
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        frame.loc[~np.isfinite(frame[feature]), feature] = np.nan
        if frame[feature].notna().sum() == 0:
            raise ValueError(f"Feature {feature} has no usable numeric values.")

    frame[TARGET_COL] = (pd.to_numeric(frame["mpr"], errors="raise") >= 0.90).astype(int)
    dev = frame.loc[frame["center"].eq(DEV_CENTER)].copy().reset_index(drop=True)
    val = frame.loc[frame["center"].eq(VAL_CENTER)].copy().reset_index(drop=True)
    if dev.empty or val.empty:
        raise ValueError("Both development and validation centers must be present.")

    dev_ids = set(dev[subject_id_col].astype(str))
    val_ids = set(val[subject_id_col].astype(str))
    subject_overlap = dev_ids.intersection(val_ids)
    feature_overlap = _feature_fingerprints(dev).intersection(_feature_fingerprints(val))
    if subject_overlap:
        raise ValueError("The development and validation cohorts share subject identifiers.")
    if feature_overlap:
        raise ValueError(
            "Exact four-feature records occur in both centers. Resolve possible duplicate "
            "patients before running the benchmark; the values are intentionally not logged."
        )

    audit = {
        "created_utc": _utc_now(),
        "source_file_sha256": _sha256(data_file),
        "subject_id_column": subject_id_col,
        "development_center": DEV_CENTER,
        "validation_center": VAL_CENTER,
        "source_rows": int(len(source)),
        "included_rows": int(len(frame)),
        "excluded_missing_mpr": int((~eligible).sum()),
        "exclusions_by_center": {
            str(center): int(count)
            for center, count in source.loc[~eligible, "center"].value_counts().items()
        },
        "inclusion_rule": "center in {yz, zy} and non-missing numeric mpr",
        "other_exclusions": 0,
        "development_n": int(len(dev)),
        "validation_n": int(len(val)),
        "development_class_counts": {
            str(k): int(v) for k, v in dev[TARGET_COL].value_counts().sort_index().items()
        },
        "validation_class_counts_LOCKED": {
            str(k): int(v) for k, v in val[TARGET_COL].value_counts().sort_index().items()
        },
        "missing_feature_values_before_imputation": {
            "development": {f: int(dev[f].isna().sum()) for f in FEATURES},
            "validation": {f: int(val[f].isna().sum()) for f in FEATURES},
        },
        "subject_overlap": 0,
        "exact_feature_record_overlap": 0,
    }

    imputation_medians = dev[FEATURES].median()
    dev[FEATURES] = dev[FEATURES].fillna(imputation_medians)
    val[FEATURES] = val[FEATURES].fillna(imputation_medians)
    audit["development_imputation_medians"] = {
        feature: float(imputation_medians[feature]) for feature in FEATURES
    }
    if dev[TARGET_COL].nunique() != 2:
        raise ValueError("Development outcomes must contain both classes.")
    if val[TARGET_COL].nunique() != 2:
        raise ValueError("Validation outcomes must contain both classes for AUROC evaluation.")
    return dev, val, audit


def _stratified_sample(dev: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.DataFrame:
    if k == 0:
        return dev.iloc[0:0].copy()
    if k > len(dev):
        raise ValueError(f"Requested {k} demonstrations from {len(dev)} cases.")
    n_pos = k // 2
    n_neg = k - n_pos
    pos = dev.loc[dev[TARGET_COL].eq(1)]
    neg = dev.loc[dev[TARGET_COL].eq(0)]
    if n_pos > len(pos) or n_neg > len(neg):
        raise ValueError(f"Cannot create a balanced {k}-shot set from the available classes.")
    chosen = pd.concat(
        [
            pos.sample(n=n_pos, random_state=int(rng.integers(1, 2**31 - 1))),
            neg.sample(n=n_neg, random_state=int(rng.integers(1, 2**31 - 1))),
        ],
        ignore_index=True,
    )
    return chosen.sample(
        frac=1, random_state=int(rng.integers(1, 2**31 - 1))
    ).reset_index(drop=True)


def _format_case(row: pd.Series, include_label: bool = False) -> str:
    values = ", ".join(f"{FEATURE_LABELS[f]}={float(row[f]):.3f}" for f in FEATURES)
    text = f"case_id={row['public_id']}: {values}"
    if include_label:
        text += f"; observed_label={int(row[TARGET_COL])}"
    return text


def _system_prompt(n_validation: int) -> str:
    return f"""You are evaluating a binary prediction task. Predict whether MPR90 is achieved from four pre-treatment features.

MPR90 means MPR >= 0.90. Use the labeled reference cases only as in-context demonstrations. Do not use outside patient information and do not infer validation labels.

Return exactly one JSON array with {n_validation} objects, in the same order as the input cases. Each object must contain only:
  id: the provided case_id string
  pred_label: 0 or 1
  prob_mpr90: number from 0 to 1, rounded to three decimals

Set pred_label=1 when prob_mpr90 >= 0.500. Do not output explanations, confidence scores, markdown, or additional fields."""


def _build_prompt(exemplars: pd.DataFrame, validation: pd.DataFrame) -> str:
    if exemplars.empty:
        demo_block = "No labeled demonstrations are provided. This is the zero-shot condition."
    else:
        demos = "\n".join(
            f"{index + 1}. {_format_case(row, include_label=True)}"
            for index, (_, row) in enumerate(exemplars.iterrows())
        )
        demo_block = "Labeled demonstrations:\n" + demos
    tests = "\n".join(
        f"{index + 1}. {_format_case(row)}"
        for index, (_, row) in enumerate(validation.iterrows())
    )
    return f"""{demo_block}

Unlabeled validation cases:
{tests}

Predict all cases in order and return only the required JSON array."""


def prepare_run(
    data_file: Path,
    run_dir: Path,
    subject_id_col: str = "id",
    primary_shot: int | None = None,
) -> None:
    dev, val, audit = _load_and_audit_cohorts(data_file, subject_id_col)
    class_counts = dev[TARGET_COL].value_counts()
    feasible_shots = [
        k
        for k in SHOT_SIZES
        if k < len(dev)
        and k // 2 <= int(class_counts.get(1, 0))
        and k - k // 2 <= int(class_counts.get(0, 0))
    ]
    shot_sizes = sorted(set(feasible_shots + [len(dev)]))
    if primary_shot is not None and primary_shot not in shot_sizes:
        raise ValueError(f"primary_shot must be one of {shot_sizes}, got {primary_shot}.")

    _require_new_directory(run_dir)
    operator_dir = run_dir / "operator"
    private_dir = run_dir / "private"
    operator_dir.mkdir()
    private_dir.mkdir()
    snapshot_dir = private_dir / "reproducibility_snapshot"
    snapshot_dir.mkdir()
    repository_artifacts = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("test_llm_icl_benchmark.py"),
        Path(__file__).resolve().with_name("EXPERIMENT_PROTOCOL.md"),
        Path(__file__).resolve().with_name("PREREGISTRATION.md"),
        Path(__file__).resolve().with_name("PREREGISTRATION_ADDENDUM_01.md"),
    ]
    for artifact in repository_artifacts:
        if artifact.exists():
            shutil.copy2(artifact, snapshot_dir / artifact.name)
    key = secrets.token_bytes(32)
    dev["public_id"] = [
        _public_id(raw_id, DEV_CENTER, key, "D") for raw_id in dev[subject_id_col]
    ]
    val["public_id"] = [
        _public_id(raw_id, VAL_CENTER, key, "V") for raw_id in val[subject_id_col]
    ]
    dev = dev.sort_values("public_id").reset_index(drop=True)
    val = val.sort_values("public_id").reset_index(drop=True)

    (private_dir / "pseudonym_key.hex").write_text(key.hex(), encoding="ascii")
    pd.concat(
        [
            dev.assign(cohort="development"),
            val.assign(cohort="validation"),
        ],
        ignore_index=True,
    )[["cohort", subject_id_col, "public_id"]].to_csv(
        private_dir / "id_map_LOCKED.csv", index=False
    )
    val[["public_id", "mpr", TARGET_COL]].to_csv(
        private_dir / "validation_ground_truth_LOCKED.csv", index=False
    )
    _write_json(private_dir / "leakage_audit_LOCKED.json", audit)

    allowed_dev = ["public_id", *FEATURES, TARGET_COL]
    allowed_val = ["public_id", *FEATURES]
    dev[allowed_dev].to_csv(operator_dir / "development_labeled_pool.csv", index=False)
    val[allowed_val].to_csv(operator_dir / "validation_features_BLINDED.csv", index=False)

    expected_rows: list[dict[str, object]] = []
    prompt_records: list[dict[str, object]] = []
    exemplar_records: list[pd.DataFrame] = []
    conditions: list[dict[str, object]] = []
    for k in shot_sizes:
        reps = 1 if k in (0, len(dev)) else REPLICATES
        for rep in range(1, reps + 1):
            rng = np.random.default_rng(RANDOM_SEED + 1000 * k + rep - 1)
            exemplars = dev.copy() if k == len(dev) else _stratified_sample(dev, k, rng)
            condition_id = f"shot{k:03d}_rep{rep:02d}"
            exported_exemplars = exemplars[allowed_dev].copy()
            exported_exemplars.insert(0, "condition_id", condition_id)
            exemplar_records.append(exported_exemplars)
            conditions.append(
                {
                    "condition_id": condition_id,
                    "shot_size": int(k),
                    "replicate": int(rep),
                    "full_pool_sensitivity_analysis": bool(k == len(dev)),
                }
            )
            for batch_number, start in enumerate(
                range(0, len(val), VALIDATION_BATCH_SIZE), start=1
            ):
                batch = val.iloc[start : start + VALIDATION_BATCH_SIZE]
                prompt_records.append(
                    {
                        "condition_id": condition_id,
                        "shot_size": int(k),
                        "replicate": int(rep),
                        "batch": int(batch_number),
                        "system_prompt": _system_prompt(len(batch)),
                        "user_prompt": _build_prompt(exemplars, batch),
                    }
                )
                expected_rows.extend(
                    {
                        "condition_id": condition_id,
                        "shot_size": int(k),
                        "replicate": int(rep),
                        "batch": int(batch_number),
                        "public_id": public_id,
                    }
                    for public_id in batch["public_id"]
                )

    pd.concat(exemplar_records, ignore_index=True).to_csv(
        operator_dir / "exemplars.csv", index=False
    )
    pd.DataFrame(expected_rows).to_csv(operator_dir / "expected_predictions.csv", index=False)
    with (operator_dir / "prompts.jsonl").open("w", encoding="utf-8") as handle:
        for record in prompt_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "created_utc": _utc_now(),
        "seed": RANDOM_SEED,
        "development_n": int(len(dev)),
        "validation_n": int(len(val)),
        "features": FEATURES,
        "target_definition": "mpr >= 0.90",
        "shot_sizes": shot_sizes,
        "replicates_per_intermediate_shot": REPLICATES,
        "validation_batch_size": VALIDATION_BATCH_SIZE,
        "demonstration_sampling": "balanced within each intermediate-shot replicate",
        "full_pool_condition": "sensitivity analysis with natural development prevalence",
        "primary_metric": "AUROC",
        "secondary_metrics": ["AUPRC", "Brier", "log loss", "accuracy", "sensitivity", "specificity"],
        "decision_threshold": 0.5,
        "primary_shot": primary_shot,
        "analysis_status": "confirmatory" if primary_shot is not None else "exploratory until primary_shot is preregistered",
        "conditions": conditions,
        "validation_labels_in_operator_files": False,
        "raw_identifiers_in_operator_files": False,
        "software_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }
    _write_json(operator_dir / "experiment_manifest.json", manifest)

    _append_ledger(
        run_dir,
        "PREPARE_COMPLETE",
        {
            "source_file_sha256": audit["source_file_sha256"],
            "development_n": int(len(dev)),
            "validation_n": int(len(val)),
            "primary_shot": primary_shot,
            "operator_files_created": sorted(path.name for path in operator_dir.iterdir()),
        },
    )

    protected_files = sorted(
        path
        for root in (operator_dir, private_dir)
        for path in root.rglob("*")
        if path.is_file()
    )
    integrity = {
        "created_utc": _utc_now(),
        "analysis_code_sha256": _sha256(Path(__file__).resolve()),
        "files": {
            str(path.relative_to(run_dir)): _sha256(path)
            for path in protected_files
            if path.name not in {"integrity_LOCKED.json", "run_ledger_LOCKED.jsonl"}
        },
    }
    _write_json(private_dir / "integrity_LOCKED.json", integrity)


def _read_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "operator" / "experiment_manifest.json").read_text(encoding="utf-8"))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


def _fit_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    l2_penalty: float = 1.0,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> dict[str, object]:
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales[scales == 0] = 1.0
    standardized = (features - means) / scales
    design = np.column_stack([np.ones(len(standardized)), standardized])
    coefficients = np.zeros(design.shape[1])
    penalty = np.diag([0.0, *([l2_penalty] * features.shape[1])])
    converged = False
    for iteration in range(1, max_iterations + 1):
        probability = _sigmoid(design @ coefficients)
        gradient = design.T @ (probability - labels) + penalty @ coefficients
        weights = probability * (1.0 - probability)
        hessian = design.T @ (design * weights[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if np.linalg.norm(step, ord=np.inf) < tolerance:
            converged = True
            break
    if not converged:
        raise RuntimeError("The fixed logistic-regression baseline did not converge.")
    return {
        "means": means,
        "scales": scales,
        "coefficients": coefficients,
        "iterations": iteration,
        "l2_penalty": l2_penalty,
    }


def _predict_logistic(model: dict[str, object], features: np.ndarray) -> np.ndarray:
    standardized = (features - model["means"]) / model["scales"]
    design = np.column_stack([np.ones(len(standardized)), standardized])
    return _sigmoid(design @ model["coefficients"])


def run_baseline(run_dir: Path) -> Path:
    operator_dir = run_dir / "operator"
    output_dir = run_dir / "predictions"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "baseline_logistic.csv"
    if output_file.exists():
        raise FileExistsError(f"Refusing to overwrite frozen predictions: {output_file}")
    manifest = _read_manifest(run_dir)
    dev = pd.read_csv(operator_dir / "development_labeled_pool.csv")
    val = pd.read_csv(operator_dir / "validation_features_BLINDED.csv")
    model = _fit_logistic_regression(
        dev[FEATURES].to_numpy(dtype=float), dev[TARGET_COL].to_numpy(dtype=int)
    )
    probability = _predict_logistic(model, val[FEATURES].to_numpy(dtype=float))
    pd.DataFrame(
        {
            "public_id": val["public_id"],
            "pred_label": (probability >= manifest["decision_threshold"]).astype(int),
            "prob_mpr90": probability,
        }
    ).to_csv(output_file, index=False)
    _write_json(
        output_dir / "baseline_model_audit.json",
        {
            "model": "L2-penalized logistic regression fitted by Newton-Raphson",
            "features": FEATURES,
            "standardization_means": dict(zip(FEATURES, model["means"].tolist())),
            "standardization_scales": dict(zip(FEATURES, model["scales"].tolist())),
            "intercept": float(model["coefficients"][0]),
            "coefficients_standardized": dict(
                zip(FEATURES, model["coefficients"][1:].tolist())
            ),
            "l2_penalty": model["l2_penalty"],
            "iterations": model["iterations"],
            "seed": manifest["seed"],
        },
    )
    _freeze_file(output_file)
    _append_ledger(
        run_dir,
        "BASELINE_PREDICTIONS_FROZEN",
        {
            "prediction_sha256": _sha256(output_file),
            "model_audit_sha256": _sha256(output_dir / "baseline_model_audit.json"),
            "validation_rows": int(len(val)),
        },
    )
    return output_file


def validate_llm_predictions(run_dir: Path, input_file: Path) -> Path:
    expected = pd.read_csv(run_dir / "operator" / "expected_predictions.csv")
    predictions = pd.read_csv(input_file)
    required = ["condition_id", "public_id", "pred_label", "prob_mpr90"]
    missing = sorted(set(required).difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")
    predictions = predictions[required].copy()
    if predictions.duplicated(["condition_id", "public_id"]).any():
        raise ValueError("Prediction file contains duplicate condition_id/public_id rows.")
    probabilities = pd.to_numeric(predictions["prob_mpr90"], errors="coerce")
    labels = pd.to_numeric(predictions["pred_label"], errors="coerce")
    if probabilities.isna().any() or (~probabilities.between(0, 1)).any():
        raise ValueError("prob_mpr90 must contain finite numbers in [0, 1].")
    if labels.isna().any() or (~labels.isin([0, 1])).any():
        raise ValueError("pred_label must contain only 0 or 1.")
    if (labels.astype(int).to_numpy() != (probabilities.to_numpy() >= 0.5)).any():
        raise ValueError("pred_label is inconsistent with the preregistered 0.5 threshold.")
    predictions["pred_label"] = labels.astype(int)
    predictions["prob_mpr90"] = probabilities.astype(float)

    checked = expected.merge(
        predictions,
        on=["condition_id", "public_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    counts = checked["_merge"].value_counts()
    if counts.get("left_only", 0) or counts.get("right_only", 0):
        raise ValueError(
            f"Prediction coverage mismatch: missing={int(counts.get('left_only', 0))}, "
            f"unexpected={int(counts.get('right_only', 0))}."
        )
    checked = checked.drop(columns="_merge")
    output_dir = run_dir / "predictions"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "llm_predictions_VALIDATED.csv"
    if output_file.exists():
        raise FileExistsError(f"Refusing to overwrite frozen predictions: {output_file}")
    checked.to_csv(output_file, index=False)
    _freeze_file(output_file)
    _append_ledger(
        run_dir,
        "LLM_PREDICTIONS_VALIDATED_AND_FROZEN",
        {
            "submitted_file_sha256": _sha256(input_file),
            "validated_file_sha256": _sha256(output_file),
            "prediction_rows": int(len(checked)),
            "conditions": int(checked["condition_id"].nunique()),
        },
    )
    return output_file


def _verify_integrity(run_dir: Path) -> None:
    record = json.loads((run_dir / "private" / "integrity_LOCKED.json").read_text(encoding="utf-8"))
    if _sha256(Path(__file__).resolve()) != record["analysis_code_sha256"]:
        raise ValueError("Analysis code changed after run preparation; start a new run directory.")
    changed = []
    for relative, expected_hash in record["files"].items():
        path = run_dir / relative
        if not path.exists() or _sha256(path) != expected_hash:
            changed.append(relative)
    if changed:
        raise ValueError(f"Protected files changed after preparation: {changed}")
    _verify_ledger(run_dir)


def _metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (probability >= threshold).astype(int)
    positives = y_true == 1
    negatives = ~positives
    ranks = pd.Series(probability).rank(method="average").to_numpy()
    n_positive = int(positives.sum())
    n_negative = int(negatives.sum())
    auroc = (ranks[positives].sum() - n_positive * (n_positive + 1) / 2) / (
        n_positive * n_negative
    )
    order = np.argsort(-probability, kind="stable")
    sorted_labels = y_true[order]
    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)
    distinct = np.r_[probability[order][1:] != probability[order][:-1], True]
    precision = true_positives[distinct] / (true_positives[distinct] + false_positives[distinct])
    recall = true_positives[distinct] / n_positive
    auprc = np.sum(np.diff(np.r_[0.0, recall]) * precision)
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "brier": float(np.mean((probability - y_true) ** 2)),
        "log_loss": float(-np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped))),
        "accuracy": float((predicted == y_true).mean()),
        "sensitivity": float(predicted[positives].mean()),
        "specificity": float((predicted[negatives] == 0).mean()),
    }


def _bootstrap_intervals(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    positive_index = np.flatnonzero(y_true == 1)
    negative_index = np.flatnonzero(y_true == 0)
    samples: dict[str, list[float]] = {}
    for _ in range(iterations):
        index = np.concatenate(
            [
                rng.choice(positive_index, len(positive_index), replace=True),
                rng.choice(negative_index, len(negative_index), replace=True),
            ]
        )
        for name, value in _metrics(y_true[index], probability[index], threshold).items():
            samples.setdefault(name, []).append(value)
    return {
        name: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        for name, values in samples.items()
    }


def _metric_row(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    point = _metrics(y_true, probability, threshold)
    intervals = _bootstrap_intervals(y_true, probability, threshold, iterations, seed)
    row: dict[str, float] = {}
    for name, value in point.items():
        row[name] = value
        row[f"{name}_ci_low"] = intervals[name][0]
        row[f"{name}_ci_high"] = intervals[name][1]
    return row


def _hierarchical_shot_summary(
    y_true: np.ndarray,
    baseline_probability: np.ndarray,
    replicate_probabilities: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    metric_names = list(_metrics(y_true, replicate_probabilities[0], threshold))
    replicate_metrics = [
        _metrics(y_true, probability, threshold)
        for probability in replicate_probabilities
    ]
    baseline_auroc = _metrics(y_true, baseline_probability, threshold)["auroc"]
    result: dict[str, float] = {
        "prompt_replicates": int(len(replicate_probabilities)),
    }
    for metric in metric_names:
        values = np.array([row[metric] for row in replicate_metrics])
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_prompt_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result[f"{metric}_min"] = float(values.min())
        result[f"{metric}_max"] = float(values.max())
    result["auroc_delta_vs_baseline"] = result["auroc_mean"] - baseline_auroc

    rng = np.random.default_rng(seed)
    positive_index = np.flatnonzero(y_true == 1)
    negative_index = np.flatnonzero(y_true == 0)
    bootstrapped = {metric: [] for metric in metric_names}
    delta_samples = []
    n_replicates = len(replicate_probabilities)
    for _ in range(iterations):
        patient_index = np.concatenate(
            [
                rng.choice(positive_index, len(positive_index), replace=True),
                rng.choice(negative_index, len(negative_index), replace=True),
            ]
        )
        replicate_index = rng.choice(n_replicates, n_replicates, replace=True)
        sampled_metrics = [
            _metrics(
                y_true[patient_index],
                replicate_probabilities[index, patient_index],
                threshold,
            )
            for index in replicate_index
        ]
        for metric in metric_names:
            bootstrapped[metric].append(
                float(np.mean([row[metric] for row in sampled_metrics]))
            )
        sampled_baseline_auroc = _metrics(
            y_true[patient_index], baseline_probability[patient_index], threshold
        )["auroc"]
        delta_samples.append(bootstrapped["auroc"][-1] - sampled_baseline_auroc)

    for metric, values in bootstrapped.items():
        result[f"{metric}_ci_low"] = float(np.quantile(values, 0.025))
        result[f"{metric}_ci_high"] = float(np.quantile(values, 0.975))
    result["auroc_delta_ci_low"] = float(np.quantile(delta_samples, 0.025))
    result["auroc_delta_ci_high"] = float(np.quantile(delta_samples, 0.975))
    return result


def evaluate_run(run_dir: Path, bootstrap_iterations: int = BOOTSTRAP_REPLICATES) -> Path:
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be at least 1.")
    _verify_integrity(run_dir)
    evaluation_dir = run_dir / "evaluation"
    _require_new_directory(evaluation_dir)
    manifest = _read_manifest(run_dir)
    threshold = float(manifest["decision_threshold"])
    truth = pd.read_csv(run_dir / "private" / "validation_ground_truth_LOCKED.csv")
    baseline = pd.read_csv(run_dir / "predictions" / "baseline_logistic.csv")
    llm = pd.read_csv(run_dir / "predictions" / "llm_predictions_VALIDATED.csv")
    _verify_frozen_file(run_dir / "predictions" / "baseline_logistic.csv")
    _verify_frozen_file(run_dir / "predictions" / "llm_predictions_VALIDATED.csv")
    if set(baseline["public_id"]) != set(truth["public_id"]):
        raise ValueError("Baseline predictions do not exactly cover the validation cohort.")
    if truth[TARGET_COL].nunique() != 2:
        raise ValueError("Validation outcomes contain only one class; AUROC is undefined.")

    baseline_scored = truth[["public_id", TARGET_COL]].merge(
        baseline, on="public_id", validate="one_to_one"
    ).sort_values("public_id")
    y_true = baseline_scored[TARGET_COL].to_numpy(dtype=int)
    baseline_probability = baseline_scored["prob_mpr90"].to_numpy(dtype=float)
    baseline_row = _metric_row(
        y_true, baseline_probability, threshold, bootstrap_iterations, RANDOM_SEED
    )
    pd.DataFrame([baseline_row]).to_csv(evaluation_dir / "baseline_metrics.csv", index=False)

    llm_scored = llm.merge(
        truth[["public_id", TARGET_COL]], on="public_id", validate="many_to_one"
    )
    replicate_rows = []
    delta_rows = []
    for index, (condition_id, group) in enumerate(llm_scored.groupby("condition_id", sort=True)):
        aligned = truth[["public_id", TARGET_COL]].merge(
            group[["public_id", "prob_mpr90", "shot_size", "replicate"]],
            on="public_id",
            validate="one_to_one",
        ).sort_values("public_id")
        probability = aligned["prob_mpr90"].to_numpy(dtype=float)
        current_y = aligned[TARGET_COL].to_numpy(dtype=int)
        row = {
            "condition_id": condition_id,
            "shot_size": int(aligned["shot_size"].iloc[0]),
            "replicate": int(aligned["replicate"].iloc[0]),
            **_metric_row(
                current_y,
                probability,
                threshold,
                bootstrap_iterations,
                RANDOM_SEED + index + 1,
            ),
        }
        replicate_rows.append(row)

        rng = np.random.default_rng(RANDOM_SEED + 100_000 + index)
        positive_index = np.flatnonzero(current_y == 1)
        negative_index = np.flatnonzero(current_y == 0)
        deltas = []
        for _ in range(bootstrap_iterations):
            sampled = np.concatenate(
                [
                    rng.choice(positive_index, len(positive_index), replace=True),
                    rng.choice(negative_index, len(negative_index), replace=True),
                ]
            )
            deltas.append(
                _metrics(current_y[sampled], probability[sampled], threshold)["auroc"]
                - _metrics(current_y[sampled], baseline_probability[sampled], threshold)["auroc"]
            )
        delta_rows.append(
            {
                "condition_id": condition_id,
                "shot_size": row["shot_size"],
                "replicate": row["replicate"],
                "auroc_delta_vs_baseline": row["auroc"] - baseline_row["auroc"],
                "auroc_delta_ci_low": float(np.quantile(deltas, 0.025)),
                "auroc_delta_ci_high": float(np.quantile(deltas, 0.975)),
            }
        )

    replicate_frame = pd.DataFrame(replicate_rows)
    replicate_frame.to_csv(evaluation_dir / "llm_metrics_by_replicate.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(
        evaluation_dir / "paired_auroc_delta_vs_baseline.csv", index=False
    )
    shot_rows = []
    ordered_truth = truth[["public_id", TARGET_COL]].sort_values("public_id")
    ordered_y = ordered_truth[TARGET_COL].to_numpy(dtype=int)
    for shot_size, shot_group in llm_scored.groupby("shot_size", sort=True):
        probability_rows = []
        for _, condition_group in shot_group.groupby("condition_id", sort=True):
            aligned = ordered_truth[["public_id"]].merge(
                condition_group[["public_id", "prob_mpr90"]],
                on="public_id",
                validate="one_to_one",
            )
            probability_rows.append(aligned["prob_mpr90"].to_numpy(dtype=float))
        shot_rows.append(
            {
                "shot_size": int(shot_size),
                **_hierarchical_shot_summary(
                    ordered_y,
                    baseline_probability,
                    np.vstack(probability_rows),
                    threshold,
                    bootstrap_iterations,
                    RANDOM_SEED + 200_000 + int(shot_size),
                ),
            }
        )
    shot_frame = pd.DataFrame(shot_rows)
    shot_frame.to_csv(evaluation_dir / "llm_metrics_by_shot.csv", index=False)

    primary_shot = manifest["primary_shot"]
    if primary_shot is not None:
        primary_row = shot_frame.loc[shot_frame["shot_size"].eq(primary_shot)]
        if len(primary_row) != 1:
            raise ValueError("The preregistered primary shot condition is missing or duplicated.")
        _write_json(
            evaluation_dir / "primary_comparison.json",
            primary_row.iloc[0].to_dict(),
        )

    _write_json(
        evaluation_dir / "cohort_summary.json",
        {
            "development_n": manifest["development_n"],
            "validation_n": int(len(truth)),
            "validation_mpr90_positive": int(truth[TARGET_COL].sum()),
            "validation_mpr90_negative": int((1 - truth[TARGET_COL]).sum()),
            "independent_unit": "patient",
        },
    )

    audit = {
        "created_utc": _utc_now(),
        "bootstrap_method": "stratified patient-level percentile bootstrap; shot summaries also resample demonstration replicates",
        "bootstrap_replicates": bootstrap_iterations,
        "independent_unit": "validation patient",
        "primary_shot": manifest["primary_shot"],
        "analysis_status": manifest["analysis_status"],
        "multiplicity_note": "All shot-level comparisons are exploratory unless primary_shot was preregistered before unblinding.",
        "prediction_file_hashes": {
            "baseline_logistic.csv": _sha256(run_dir / "predictions" / "baseline_logistic.csv"),
            "llm_predictions_VALIDATED.csv": _sha256(run_dir / "predictions" / "llm_predictions_VALIDATED.csv"),
        },
    }
    _write_json(evaluation_dir / "evaluation_audit.json", audit)
    _append_ledger(
        run_dir,
        "FINAL_UNBLINDING_AND_EVALUATION_COMPLETE",
        {
            "bootstrap_replicates": bootstrap_iterations,
            "primary_shot": primary_shot,
            "evaluation_file_hashes": {
                path.name: _sha256(path)
                for path in sorted(evaluation_dir.iterdir())
                if path.is_file()
            },
        },
    )
    return evaluation_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create blinded operator and locked private packages.")
    prepare.add_argument("--data", type=Path, default=DEFAULT_DATA_FILE)
    prepare.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    prepare.add_argument("--subject-id-column", default="id")
    prepare.add_argument("--primary-shot", type=int)

    baseline = subparsers.add_parser("baseline", help="Run the blinded logistic-regression baseline.")
    baseline.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)

    validate = subparsers.add_parser("validate-predictions", help="Validate and freeze LLM prediction CSV.")
    validate.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    validate.add_argument("--input", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate", help="Verify integrity and unblind final evaluation.")
    evaluate.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    evaluate.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        prepare_run(args.data, args.run_dir, args.subject_id_column, args.primary_shot)
        print(f"Prepared blinded run at {args.run_dir.resolve()}")
    elif args.command == "baseline":
        print(f"Frozen baseline predictions: {run_baseline(args.run_dir).resolve()}")
    elif args.command == "validate-predictions":
        print(f"Frozen validated LLM predictions: {validate_llm_predictions(args.run_dir, args.input).resolve()}")
    elif args.command == "evaluate":
        print(f"Final evaluation: {evaluate_run(args.run_dir, args.bootstrap_replicates).resolve()}")


if __name__ == "__main__":
    main()
