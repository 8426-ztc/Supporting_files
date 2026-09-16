"""Merge and outcome-blind freeze all 122 cohort-batched round outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from audit_round_inference import audit
from codex_batch_inference import append_chain
from codex_round_inference import ROUND_FIELDS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(run_dir: Path, model: str, effort: str) -> dict:
    prediction_dir = run_dir / "predictions"
    output = prediction_dir / "llm_predictions_VALIDATED.csv"
    freeze_path = output.with_suffix(output.suffix + ".freeze.json")
    if output.exists() or freeze_path.exists():
        if not output.exists() or not freeze_path.exists():
            raise ValueError("Incomplete pre-existing prediction freeze")
        previous = json.loads(freeze_path.read_text(encoding="utf-8"))
        if sha256_file(output) != previous["predictions_sha256"]:
            raise ValueError("Frozen batched prediction file was modified")
        return previous

    audit_result = audit(run_dir, model, effort)
    if audit_result["audit_status"] != "PASS" or audit_result["remaining_rounds"] != 0:
        raise ValueError("All 122 rounds must pass the outcome-blind audit before freezing")
    rows = []
    round_hashes = {}
    for directory in sorted(prediction_dir.glob("round_predictions_shard??of04")):
        for path in sorted(directory.glob("*.csv")):
            round_hashes[path.name] = sha256_file(path)
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != ROUND_FIELDS:
                    raise ValueError(f"Unexpected round schema: {path}")
                rows.extend(reader)
    if len(rows) != 20_252:
        raise ValueError(f"Expected 20,252 patient outputs, found {len(rows)}")
    keys = [(row["condition_id"], row["public_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate condition/patient output key during freeze")
    rows.sort(key=lambda row: (int(row["round_number"]), int(row["query_order"])))
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROUND_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_OUTCOME_BLIND",
        "rounds": 122,
        "patient_outputs": len(rows),
        "unique_condition_patient_keys": len(set(keys)),
        "model": model,
        "reasoning_effort": effort,
        "predictions_sha256": sha256_file(output),
        "round_output_sha256": dict(sorted(round_hashes.items())),
        "prompts_sha256": sha256_file(run_dir / "operator" / "prompts.jsonl"),
        "expected_predictions_sha256": sha256_file(
            run_dir / "operator" / "expected_predictions.csv"
        ),
        "run_manifest_sha256": sha256_file(
            run_dir / "private" / "batched_run_manifest_LOCKED.json"
        ),
        "freeze_code_sha256": sha256_file(Path(__file__).resolve()),
        "inference_audit_code_sha256": sha256_file(
            Path(__file__).resolve().with_name("audit_round_inference.py")
        ),
        "validation_outcomes_accessed": False,
        "development_label_exposed_rows_retained_and_flagged": True,
    }
    freeze_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_chain(
        prediction_dir / "prediction_freeze_ledger.jsonl",
        "ALL_BATCHED_PREDICTIONS_FROZEN",
        {
            "predictions_sha256": record["predictions_sha256"],
            "rounds": 122,
            "patient_outputs": len(rows),
            "validation_outcomes_accessed": False,
        },
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    args = parser.parse_args()
    result = freeze(args.run_dir.resolve(), args.model, args.reasoning_effort)
    print(
        json.dumps(
            {
                "status": result["status"],
                "rounds": result["rounds"],
                "patient_outputs": result["patient_outputs"],
                "predictions_sha256": result["predictions_sha256"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

