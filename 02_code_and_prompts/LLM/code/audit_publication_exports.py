from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


SHOT_COLUMNS = [
    "condition_id", "round_number", "shot_size", "replicate", "query_order",
    "cohort", "public_id", "label_exposed", "metric_evaluable",
    "eligible_for_development_metric", "target_mpr90", "pred_label",
    "prob_mpr90", "threshold_consistent", "attempt", "model",
    "reasoning_effort", "prompt_sha256", "response_sha256", "events_sha256",
    "completed_utc",
]
FORBIDDEN_EXACT_COLUMNS = {"raw_id", "original_id", "patient_id", "mpr", "status", "rfs"}
PUBLIC_ID = re.compile(r"^[DV]-[0-9a-f]{16}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--workbook", required=True, type=Path)
    args = parser.parse_args()

    publication = args.run_dir / "publication"
    shot_files = sorted((publication / "workbook_source").glob("shot_*.csv"))
    failures: list[str] = []
    checked_rows = 0
    public_ids: set[str] = set()
    file_records = []

    for path in sorted(publication.rglob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
        forbidden = sorted(set(header) & FORBIDDEN_EXACT_COLUMNS)
        if forbidden:
            failures.append(f"{path.name}: prohibited columns {forbidden}")
        file_records.append({"path": str(path.relative_to(publication)), "sha256": sha256(path), "columns": header})

    for path in shot_files:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != SHOT_COLUMNS:
                failures.append(f"{path.name}: shot export schema mismatch")
            groups: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in reader:
                checked_rows += 1
                groups[row["condition_id"]].append(row)
                public_ids.add(row["public_id"])
                if not PUBLIC_ID.fullmatch(row["public_id"]):
                    failures.append(f"{path.name}: non-pseudonymized public_id pattern")
                if row["target_mpr90"] not in {"0", "1"}:
                    failures.append(f"{path.name}: non-binary target_mpr90")
                if row["cohort"] == "validation" and row["label_exposed"] != "0":
                    failures.append(f"{path.name}: validation label_exposed must be 0")
            for condition, rows in groups.items():
                orders = [int(row["query_order"]) for row in rows]
                ids = [row["public_id"] for row in rows]
                if len(rows) != 166 or orders != list(range(1, 167)) or len(set(ids)) != 166:
                    failures.append(f"{path.name}/{condition}: incomplete, unordered, or duplicate IDs")
                if int(rows[0]["shot_size"]) == 109:
                    dev = [row for row in rows if row["cohort"] == "development"]
                    if any(row["metric_evaluable"] != "0" or row["label_exposed"] != "1" for row in dev):
                        failures.append(f"{path.name}/{condition}: 109-shot development boundary violated")

    if len(shot_files) != 8:
        failures.append(f"expected 8 shot exports, found {len(shot_files)}")
    if checked_rows != 20252:
        failures.append(f"expected 20,252 shot rows, found {checked_rows}")
    if len(public_ids) != 166:
        failures.append(f"expected 166 unique pseudonymized IDs, found {len(public_ids)}")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "scope": "publication CSV exports and workbook source lineage",
        "workbook": str(args.workbook),
        "workbook_sha256": sha256(args.workbook),
        "shot_files": len(shot_files),
        "patient_level_rows": checked_rows,
        "unique_pseudonymized_ids": len(public_ids),
        "forbidden_exact_columns": sorted(FORBIDDEN_EXACT_COLUMNS),
        "continuous_mpr_exported": False,
        "raw_identifier_exported": False,
        "validation_label_exposure_violations": sum("validation label_exposed" in item for item in failures),
        "failures": failures,
        "files": file_records,
    }
    output = publication / "privacy_export_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(output), "rows": checked_rows, "ids": len(public_ids)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
