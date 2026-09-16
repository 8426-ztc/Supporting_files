"""Archive failed Codex attempt artifacts by ledger hash before filename reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


LEGACY_MISSING = {
    (
        "shot000_rep01",
        1,
        1,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "e12570f755a80ca2335f8ef18aac4c595bdd739fca8e18c42c68838d39f4f9a2",
    ),
    (
        "shot000_rep01",
        1,
        2,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "e12570f755a80ca2335f8ef18aac4c595bdd739fca8e18c42c68838d39f4f9a2",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def failed_records(prediction_dir: Path) -> list[dict]:
    records = []
    for ledger in sorted(prediction_dir.glob("codex_inference_ledger*.jsonl")):
        with ledger.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["event"] == "PROMPT_ATTEMPT_FAILED":
                    records.append(record["details"])
    return records


def matching_file(paths: list[Path], expected_hash: str) -> Path | None:
    for path in paths:
        if path.is_file() and sha256_file(path) == expected_hash:
            return path
    return None


def archive(run_dir: Path) -> dict:
    allowed_legacy_missing = (
        LEGACY_MISSING
        if run_dir.name == "publication_run_20260829_v2"
        else set()
    )
    prediction_dir = run_dir / "predictions"
    archive_dir = prediction_dir / "failed_attempt_archive"
    archive_dir.mkdir(exist_ok=True)
    event_dirs = [
        path for path in prediction_dir.glob("codex_events*") if path.is_dir()
    ]
    historical_dirs = [
        prediction_dir / "preflight_schema_failed",
    ]
    archived = 0
    already_preserved = 0
    legacy_missing = 0
    for details in failed_records(prediction_dir):
        condition = details["condition_id"]
        batch = int(details["batch"])
        attempt = int(details["attempt"])
        event_hash = details["events_sha256"]
        stderr_hash = details["stderr_sha256"]
        stem = f"{condition}_batch{batch:03d}_attempt{attempt:02d}"
        exception_key = (
            condition,
            batch,
            attempt,
            event_hash,
            stderr_hash,
        )
        event_name = stem + ".jsonl"
        stderr_name = stem + ".stderr.txt"
        event_candidates = [directory / event_name for directory in event_dirs + historical_dirs]
        event_candidates.extend(archive_dir.glob(f"*_{event_name}"))
        stderr_candidates = [directory / stderr_name for directory in event_dirs + historical_dirs]
        stderr_candidates.extend(archive_dir.glob(f"*_{stderr_name}"))
        event_path = matching_file(event_candidates, event_hash)
        stderr_path = matching_file(stderr_candidates, stderr_hash)
        if event_path is None or stderr_path is None:
            if exception_key in allowed_legacy_missing:
                legacy_missing += 1
                continue
            raise ValueError(f"Unpreserved failed-attempt artifact: {stem}")

        event_target = archive_dir / f"{event_hash[:16]}_{event_name}"
        stderr_target = archive_dir / f"{stderr_hash[:16]}_{stderr_name}"
        if event_target.exists() and sha256_file(event_target) != event_hash:
            raise ValueError(f"Archive hash collision: {event_target}")
        if stderr_target.exists() and sha256_file(stderr_target) != stderr_hash:
            raise ValueError(f"Archive hash collision: {stderr_target}")
        copied = False
        if not event_target.exists():
            shutil.copy2(event_path, event_target)
            copied = True
        if not stderr_target.exists():
            shutil.copy2(stderr_path, stderr_target)
            copied = True
        if copied:
            archived += 1
        else:
            already_preserved += 1

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "failed_ledger_records": len(failed_records(prediction_dir)),
        "newly_archived_record_pairs": archived,
        "already_archived_record_pairs": already_preserved,
        "documented_legacy_missing_record_pairs": legacy_missing,
        "addendum": "PREREGISTRATION_ADDENDUM_04.md",
        "archive_guard_sha256": sha256_file(Path(__file__).resolve()),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = archive_dir / f"archive_guard_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(archive(args.run_dir.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
