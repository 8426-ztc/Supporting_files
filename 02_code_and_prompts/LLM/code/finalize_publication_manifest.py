"""Finalize the publication manifest after figure and workbook QA."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    workbook = args.workbook.resolve()
    publication_dir = run_dir / "publication"
    prediction_path = run_dir / "predictions" / "llm_predictions_VALIDATED.csv"
    freeze_path = prediction_path.with_suffix(prediction_path.suffix + ".freeze.json")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if sha256_file(prediction_path) != freeze["predictions_sha256"]:
        raise ValueError("Frozen prediction hash mismatch during publication finalization")
    if not workbook.is_file():
        raise FileNotFoundError("Final workbook is missing")
    files = {}
    for path in sorted(publication_dir.rglob("*")):
        if path.is_file() and path.name != "publication_manifest.json":
            files[str(path.relative_to(publication_dir)).replace("\\", "/")] = sha256_file(path)
    record = {
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FINALIZED_FOR_MANUSCRIPT_AND_REVIEW",
        "source_prediction_sha256": freeze["predictions_sha256"],
        "prediction_freeze_sha256": sha256_file(freeze_path),
        "files_sha256": files,
        "workbook": {
            "name": workbook.name,
            "sha256": sha256_file(workbook),
        },
        "privacy_boundary": "pseudonymized IDs and binary outcomes only; no raw ID, continuous MPR, status or RFS",
    }
    manifest_path = publication_dir / "publication_manifest.json"
    manifest_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "manifest": str(manifest_path), "files": len(files), "workbook_sha256": record["workbook"]["sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
