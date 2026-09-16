"""Independent outcome-blind leakage audit for cohort-batched round prompts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


CASE_PATTERN = re.compile(r"case_id=([DV]-[0-9a-f]{16}):")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(run_dir: Path) -> dict:
    operator = run_dir / "operator"
    private = run_dir / "private"
    manifest_path = private / "batched_run_manifest_LOCKED.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected_hash in manifest["operator_artifacts_sha256"].items():
        if sha256_file(operator / name) != expected_hash:
            raise ValueError(f"Modified operator artifact: {name}")
    for name, expected_hash in manifest["private_artifacts_sha256"].items():
        if sha256_file(private / name) != expected_hash:
            raise ValueError(f"Modified private artifact: {name}")
    source_root = run_dir.parent
    if sha256_file(source_root / "prepare_batched_rounds.py") != manifest["preparation_code_sha256"]:
        raise ValueError("Batched preparation code changed")
    if sha256_file(source_root / "PREREGISTRATION_ADDENDUM_06.md") != manifest[
        "preregistration_addendum_sha256"
    ]:
        raise ValueError("Batched preregistration addendum changed")

    with (operator / "expected_predictions.csv").open(newline="", encoding="utf-8") as handle:
        expected_rows = list(csv.DictReader(handle))
    expected_by_condition: dict[str, list[dict]] = {}
    for row in expected_rows:
        expected_by_condition.setdefault(row["condition_id"], []).append(row)
    if len(expected_rows) != 20_252 or len(expected_by_condition) != 122:
        raise ValueError("Batched expected-output dimensions differ from 122 × 166")
    if len({(row["condition_id"], row["public_id"]) for row in expected_rows}) != len(expected_rows):
        raise ValueError("Duplicate condition/patient expected-output key")

    shot_round_counts: Counter = Counter()
    label_exposed_counts: Counter = Counter()
    prompt_count = 0
    with (operator / "prompts.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            condition_id = record["condition_id"]
            expected = sorted(
                expected_by_condition[condition_id], key=lambda row: int(row["query_order"])
            )
            if len(expected) != 166:
                raise ValueError(f"Round does not contain 166 expected outputs: {condition_id}")
            user_prompt = record["user_prompt"]
            demo_block, remainder = user_prompt.split(
                "\n\nUnlabeled development query cases:\n", 1
            )
            development_block, validation_remainder = remainder.split(
                "\n\nUnlabeled external-validation query cases:\n", 1
            )
            validation_block = validation_remainder.split(
                "\n\nReturn predictions for all 166 query cases", 1
            )[0]
            if "observed_label=" in development_block or "observed_label=" in validation_block:
                raise ValueError("Observed label appears in an unlabeled query block")
            development_ids = CASE_PATTERN.findall(development_block)
            validation_ids = CASE_PATTERN.findall(validation_block)
            query_ids = development_ids + validation_ids
            expected_ids = [row["public_id"] for row in expected]
            if len(development_ids) != 109 or len(validation_ids) != 57:
                raise ValueError("Development/validation query count mismatch")
            if query_ids != expected_ids or len(set(query_ids)) != 166:
                raise ValueError("Prompt query IDs differ from the frozen ordered index")
            demo_ids = CASE_PATTERN.findall(demo_block)
            shot_size = int(record["shot_size"])
            if len(demo_ids) != shot_size or len(set(demo_ids)) != shot_size:
                raise ValueError("Demonstration count or uniqueness mismatch")
            if any(public_id.startswith("V-") for public_id in demo_ids):
                raise ValueError("A validation ID appears in labeled demonstrations")
            if demo_block.count("observed_label=") != shot_size:
                raise ValueError("Demonstration labels differ from shot size")
            exposed_expected = {
                row["public_id"]
                for row in expected
                if row["cohort"] == "development" and int(row["label_exposed"]) == 1
            }
            if exposed_expected != set(demo_ids):
                raise ValueError("Label-exposure flags differ from demonstration membership")
            if any(
                int(row["label_exposed"]) != 0
                for row in expected
                if row["cohort"] == "validation"
            ):
                raise ValueError("Validation label-exposure flag is nonzero")
            shot_round_counts[shot_size] += 1
            label_exposed_counts[shot_size] += len(exposed_expected)
            prompt_count += 1

    if prompt_count != 122:
        raise ValueError("Prompt count differs from 122 rounds")
    if shot_round_counts != Counter({0: 1, 4: 20, 8: 20, 10: 20, 20: 20, 40: 20, 80: 20, 109: 1}):
        raise ValueError("Shot/replicate schedule changed")
    return {
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rounds_verified": prompt_count,
        "patient_outputs_expected": len(expected_rows),
        "development_queries_per_round": 109,
        "validation_queries_per_round": 57,
        "validation_label_exposure": 0,
        "query_label_leakage": 0,
        "rounds_by_shot": {str(key): shot_round_counts[key] for key in sorted(shot_round_counts)},
        "development_label_exposed_outputs_by_shot": {
            str(key): label_exposed_counts[key] for key in sorted(label_exposed_counts)
        },
        "design_boundary": "cohort-batched transductive inference",
        "audit_code_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "privacy_note": "Aggregate counts and hashes only; validation outcomes not read.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    report = audit(run_dir)
    audit_dir = run_dir / "audit"
    audit_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = audit_dir / f"preparation_audit_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "rounds_verified": report["rounds_verified"],
                "patient_outputs_expected": report["patient_outputs_expected"],
                "validation_label_exposure": 0,
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

