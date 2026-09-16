"""Outcome-blind integrity audit for one-call-per-round cohort-batched inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from audit_codex_inference import (
    FORBIDDEN_ITEM_TYPES,
    append_audit_index,
    read_jsonl,
    verify_chain,
)
from codex_round_inference import (
    PROMPT_SUFFIX,
    ROUND_FIELDS,
    load_expected,
    load_prompts,
    validate_response,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(run_dir: Path, model: str, effort: str) -> dict:
    operator = run_dir / "operator"
    prediction_dir = run_dir / "predictions"
    source_root = run_dir.parent
    prompt_path = operator / "prompts.jsonl"
    expected_path = operator / "expected_predictions.csv"
    schema_path = source_root / "llm_round_prediction_schema.json"
    runner_path = source_root / "codex_round_inference.py"
    epoch02_runner_path = source_root / "codex_round_inference_epoch02.py"
    epoch02_addendum_path = source_root / "PREREGISTRATION_ADDENDUM_07.md"
    epoch03_runner_path = source_root / "codex_round_inference_epoch03.py"
    epoch03_addendum_path = source_root / "PREREGISTRATION_ADDENDUM_08.md"
    dependency_path = source_root / "codex_batch_inference.py"
    prompts = load_prompts(prompt_path)
    prompt_by_condition = {row["condition_id"]: row for row in prompts}
    expected = load_expected(expected_path)

    state_paths = sorted(prediction_dir.glob("codex_inference_state_shard??of04.json"))
    if len(state_paths) != 4:
        raise ValueError("Four batched-round shard states are required")
    states = [json.loads(path.read_text(encoding="utf-8")) for path in state_paths]
    common_fields = [
        "runner_type",
        "model",
        "reasoning_effort",
        "codex_cli",
        "codex_version",
        "ephemeral_sessions",
        "sandbox",
        "training_control_confirmed_by_user",
        "queries_per_round",
        "provider_sampling_seed_supported",
        "prompts_sha256",
        "expected_predictions_sha256",
        "schema_sha256",
        "runner_sha256",
        "runner_dependency_sha256",
        "shard_count",
    ]
    for field in common_fields:
        if len({json.dumps(state[field], sort_keys=True) for state in states}) != 1:
            raise ValueError(f"Batched-round state mismatch: {field}")
    state = states[0]
    if state["runner_type"] != "cohort_batched_round":
        raise ValueError("Unexpected runner type")
    if state["model"] != model or state["reasoning_effort"] != effort:
        raise ValueError("Frozen model or reasoning effort changed")
    if not state["ephemeral_sessions"] or state["sandbox"] != "read-only empty workspace":
        raise ValueError("Ephemeral-session or sandbox control changed")
    if not state["training_control_confirmed_by_user"]:
        raise ValueError("Training-control confirmation is absent")
    if state["queries_per_round"] != 166 or state["provider_sampling_seed_supported"]:
        raise ValueError("Round size or seed-support record changed")
    if {state["shard_index"] for state in states} != set(range(4)):
        raise ValueError("Shard indices are incomplete")
    frozen_hashes = {
        "prompts_sha256": sha256_file(prompt_path),
        "expected_predictions_sha256": sha256_file(expected_path),
        "schema_sha256": sha256_file(schema_path),
        "runner_sha256": sha256_file(runner_path),
        "runner_dependency_sha256": sha256_file(dependency_path),
    }
    for field, actual_hash in frozen_hashes.items():
        if state[field] != actual_hash:
            raise ValueError(f"Frozen batched artifact changed: {field}")

    execution_epochs = {
        "1": {
            "state_files": [path.name for path in state_paths],
            "codex_cli": state["codex_cli"],
            "codex_version": state["codex_version"],
            "runner_sha256": state["runner_sha256"],
        }
    }
    all_state_paths = list(state_paths)
    epoch02_state_paths = sorted(
        prediction_dir.glob("codex_inference_state_epoch02_shard??of04.json")
    )
    if epoch02_state_paths:
        if len(epoch02_state_paths) != 4:
            raise ValueError("Execution epoch 2 requires four shard states")
        epoch02_states = [
            json.loads(path.read_text(encoding="utf-8")) for path in epoch02_state_paths
        ]
        epoch02_common_fields = common_fields + [
            "execution_epoch",
            "previous_epoch_state_sha256",
            "operational_change_addendum_sha256",
        ]
        for field in epoch02_common_fields:
            if field == "previous_epoch_state_sha256":
                continue
            if len(
                {json.dumps(epoch_state[field], sort_keys=True) for epoch_state in epoch02_states}
            ) != 1:
                raise ValueError(f"Execution epoch 2 state mismatch: {field}")
        if {epoch_state["shard_index"] for epoch_state in epoch02_states} != set(range(4)):
            raise ValueError("Execution epoch 2 shard indices are incomplete")
        epoch02_hashes = {
            "prompts_sha256": sha256_file(prompt_path),
            "expected_predictions_sha256": sha256_file(expected_path),
            "schema_sha256": sha256_file(schema_path),
            "runner_sha256": sha256_file(epoch02_runner_path),
            "runner_dependency_sha256": sha256_file(dependency_path),
            "operational_change_addendum_sha256": sha256_file(epoch02_addendum_path),
        }
        for epoch_state, epoch_state_path, old_state_path in zip(
            epoch02_states, epoch02_state_paths, state_paths
        ):
            for field, actual_hash in epoch02_hashes.items():
                if epoch_state[field] != actual_hash:
                    raise ValueError(f"Execution epoch 2 frozen artifact changed: {field}")
            if epoch_state["previous_epoch_state_sha256"] != sha256_file(old_state_path):
                raise ValueError(
                    f"Execution epoch 2 predecessor state mismatch: {epoch_state_path.name}"
                )
            for field in [
                "runner_type",
                "model",
                "reasoning_effort",
                "ephemeral_sessions",
                "sandbox",
                "training_control_confirmed_by_user",
                "queries_per_round",
                "provider_sampling_seed_supported",
                "prompts_sha256",
                "expected_predictions_sha256",
                "schema_sha256",
                "runner_dependency_sha256",
                "shard_count",
                "shard_index",
            ]:
                if epoch_state[field] != states[epoch_state["shard_index"]][field]:
                    raise ValueError(f"Scientific setting changed across execution epochs: {field}")
        if epoch02_states[0]["execution_epoch"] != 2:
            raise ValueError("Execution epoch 2 marker is invalid")
        execution_epochs["2"] = {
            "state_files": [path.name for path in epoch02_state_paths],
            "codex_cli": epoch02_states[0]["codex_cli"],
            "codex_version": epoch02_states[0]["codex_version"],
            "runner_sha256": epoch02_states[0]["runner_sha256"],
            "addendum": epoch02_addendum_path.name,
            "addendum_sha256": epoch02_hashes["operational_change_addendum_sha256"],
        }
        all_state_paths.extend(epoch02_state_paths)

    epoch03_state_paths = sorted(
        prediction_dir.glob("codex_inference_state_epoch03_shard??of04.json")
    )
    if epoch03_state_paths:
        if len(epoch03_state_paths) != 4 or not epoch02_state_paths:
            raise ValueError("Execution epoch 3 requires four shard states and execution epoch 2")
        epoch03_states = [
            json.loads(path.read_text(encoding="utf-8")) for path in epoch03_state_paths
        ]
        epoch03_common_fields = common_fields + [
            "execution_epoch",
            "previous_epoch_state_sha256",
            "operational_change_addendum_sha256",
            "timeout_seconds",
            "max_attempts",
        ]
        for field in epoch03_common_fields:
            if field == "previous_epoch_state_sha256":
                continue
            if len(
                {json.dumps(epoch_state[field], sort_keys=True) for epoch_state in epoch03_states}
            ) != 1:
                raise ValueError(f"Execution epoch 3 state mismatch: {field}")
        if {epoch_state["shard_index"] for epoch_state in epoch03_states} != set(range(4)):
            raise ValueError("Execution epoch 3 shard indices are incomplete")
        epoch03_hashes = {
            "prompts_sha256": sha256_file(prompt_path),
            "expected_predictions_sha256": sha256_file(expected_path),
            "schema_sha256": sha256_file(schema_path),
            "runner_sha256": sha256_file(epoch03_runner_path),
            "runner_dependency_sha256": sha256_file(dependency_path),
            "operational_change_addendum_sha256": sha256_file(epoch03_addendum_path),
        }
        for epoch_state, epoch_state_path, old_state_path in zip(
            epoch03_states, epoch03_state_paths, epoch02_state_paths
        ):
            for field, actual_hash in epoch03_hashes.items():
                if epoch_state[field] != actual_hash:
                    raise ValueError(f"Execution epoch 3 frozen artifact changed: {field}")
            if epoch_state["previous_epoch_state_sha256"] != sha256_file(old_state_path):
                raise ValueError(
                    f"Execution epoch 3 predecessor state mismatch: {epoch_state_path.name}"
                )
            for field in [
                "runner_type",
                "model",
                "reasoning_effort",
                "ephemeral_sessions",
                "sandbox",
                "training_control_confirmed_by_user",
                "queries_per_round",
                "provider_sampling_seed_supported",
                "prompts_sha256",
                "expected_predictions_sha256",
                "schema_sha256",
                "runner_dependency_sha256",
                "shard_count",
                "shard_index",
            ]:
                if epoch_state[field] != epoch02_states[epoch_state["shard_index"]][field]:
                    raise ValueError(f"Scientific setting changed across execution epochs: {field}")
        if epoch03_states[0]["execution_epoch"] != 3:
            raise ValueError("Execution epoch 3 marker is invalid")
        if epoch03_states[0]["timeout_seconds"] != 1800 or epoch03_states[0]["max_attempts"] != 2:
            raise ValueError("Execution epoch 3 timeout or attempt limit is invalid")
        execution_epochs["3"] = {
            "state_files": [path.name for path in epoch03_state_paths],
            "codex_cli": epoch03_states[0]["codex_cli"],
            "codex_version": epoch03_states[0]["codex_version"],
            "runner_sha256": epoch03_states[0]["runner_sha256"],
            "timeout_seconds": epoch03_states[0]["timeout_seconds"],
            "max_attempts": epoch03_states[0]["max_attempts"],
            "addendum": epoch03_addendum_path.name,
            "addendum_sha256": epoch03_hashes["operational_change_addendum_sha256"],
        }
        all_state_paths.extend(epoch03_state_paths)

    ledger_entries = {}
    failure_entries = []
    ledger_summaries = {}
    for ledger_path in sorted(prediction_dir.glob("codex_inference_ledger*shard??of04.jsonl")):
        record_count, event_counts = verify_chain(ledger_path)
        ledger_summaries[ledger_path.name] = {
            "records": record_count,
            "events": dict(sorted(event_counts.items())),
        }
        for record in read_jsonl(ledger_path):
            if record["event"] == "ROUND_COMPLETE":
                condition_id = record["details"]["condition_id"]
                if condition_id in ledger_entries:
                    raise ValueError(f"Duplicate ROUND_COMPLETE ledger entry: {condition_id}")
                ledger_entries[condition_id] = record["details"]
            elif record["event"] == "PROMPT_ATTEMPT_FAILED":
                failure_entries.append(record["details"])

    round_files = {}
    for directory in prediction_dir.glob("round_predictions_shard??of04"):
        for path in directory.glob("*.csv"):
            if path.stem in round_files:
                raise ValueError(f"Duplicate completed round file: {path.stem}")
            round_files[path.stem] = path
    if set(round_files) != set(ledger_entries):
        raise ValueError("Completed round files and ledger entries differ")

    raw_files = {
        path.name: path
        for directory in prediction_dir.glob("codex_raw_shard??of04")
        for path in directory.glob("*.json")
    }
    event_files = {
        path.name: path
        for directory in prediction_dir.glob("codex_events_shard??of04")
        for path in directory.glob("*.jsonl")
    }
    stderr_files = {
        path.name: path
        for directory in prediction_dir.glob("codex_events_shard??of04")
        for path in directory.glob("*.stderr.txt")
    }
    archived_event_files = list((prediction_dir / "failed_attempt_archive").glob("*.jsonl"))
    archived_stderr_files = list(
        (prediction_dir / "failed_attempt_archive").glob("*.stderr.txt")
    )
    accounted_events = set()
    patient_output_count = 0
    shot_counts: Counter = Counter()
    attempt_counts: Counter = Counter()
    for condition_id, output_path in round_files.items():
        with output_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ROUND_FIELDS:
                raise ValueError(f"Unexpected round-output schema: {output_path}")
            rows = list(reader)
        if len(rows) != 166:
            raise ValueError(f"Round output does not contain 166 rows: {condition_id}")
        expected_rows = expected[(condition_id, 1)]
        prompt_record = prompt_by_condition[condition_id]
        prompt = (
            prompt_record["system_prompt"]
            + "\n\n"
            + prompt_record["user_prompt"]
            + "\n\n"
            + PROMPT_SUFFIX
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        attempts = {int(row["attempt"]) for row in rows}
        if len(attempts) != 1:
            raise ValueError(f"Mixed attempts within one round: {condition_id}")
        attempt = attempts.pop()
        stem = f"{condition_id}_batch001_attempt{attempt:02d}"
        raw_name = stem + ".json"
        event_name = stem + ".jsonl"
        stderr_name = stem + ".stderr.txt"
        if raw_name not in raw_files or event_name not in event_files or stderr_name not in stderr_files:
            raise ValueError(f"Missing raw/event/stderr artifact: {condition_id}")
        raw_path = raw_files[raw_name]
        event_path = event_files[event_name]
        predictions = validate_response(raw_path, expected_rows)
        response_hash = sha256_file(raw_path)
        events_hash = sha256_file(event_path)
        for row, expected_row, prediction in zip(rows, expected_rows, predictions):
            for field in [
                "condition_id",
                "round_number",
                "shot_size",
                "replicate",
                "query_order",
                "cohort",
                "public_id",
                "label_exposed",
                "eligible_for_development_metric",
            ]:
                if row[field] != expected_row[field]:
                    raise ValueError(f"Round output metadata differs from frozen index: {condition_id}")
            if int(row["pred_label"]) != prediction["pred_label"] or float(
                row["prob_mpr90"]
            ) != prediction["prob_mpr90"]:
                raise ValueError(f"Round CSV differs from raw response: {condition_id}")
            if row["model"] != model or row["reasoning_effort"] != effort:
                raise ValueError(f"Round model setting mismatch: {condition_id}")
            if (
                row["prompt_sha256"] != prompt_hash
                or row["response_sha256"] != response_hash
                or row["events_sha256"] != events_hash
            ):
                raise ValueError(f"Round row hash mismatch: {condition_id}")
        ledger = ledger_entries[condition_id]
        if (
            ledger["prediction_rows"] != 166
            or ledger["prompt_sha256"] != prompt_hash
            or ledger["response_sha256"] != response_hash
            or ledger["events_sha256"] != events_hash
            or ledger["round_output_sha256"] != sha256_file(output_path)
        ):
            raise ValueError(f"Round ledger hash mismatch: {condition_id}")
        accounted_events.add(event_name)
        patient_output_count += 166
        shot_counts[int(rows[0]["shot_size"])] += 1
        attempt_counts[attempt] += 1

    for details in failure_entries:
        stem = (
            f"{details['condition_id']}_batch{int(details['batch']):03d}"
            f"_attempt{int(details['attempt']):02d}"
        )
        event_name = stem + ".jsonl"
        stderr_name = stem + ".stderr.txt"
        event_candidates = ([event_files[event_name]] if event_name in event_files else []) + [
            path for path in archived_event_files if path.name.endswith(event_name)
        ]
        stderr_candidates = ([stderr_files[stderr_name]] if stderr_name in stderr_files else []) + [
            path for path in archived_stderr_files if path.name.endswith(stderr_name)
        ]
        matched_event = next(
            (path for path in event_candidates if sha256_file(path) == details["events_sha256"]),
            None,
        )
        matched_stderr = next(
            (path for path in stderr_candidates if sha256_file(path) == details["stderr_sha256"]),
            None,
        )
        if matched_event is None or matched_stderr is None:
            raise ValueError(f"Missing failed-round artifact: {stem}")
        if event_name in event_files and matched_event == event_files[event_name]:
            accounted_events.add(event_name)

    if set(event_files) != accounted_events:
        raise ValueError("An event log is absent from the success/failure ledger")
    if {name.removesuffix(".stderr.txt") + ".jsonl" for name in stderr_files} != set(event_files):
        raise ValueError("Event and stderr coverage differs")
    forbidden_events = 0
    transport_errors = 0
    item_types: Counter = Counter()
    for event_path in list(event_files.values()) + archived_event_files:
        for event in read_jsonl(event_path):
            if event.get("type") == "error":
                transport_errors += 1
            item_type = (event.get("item") or {}).get("type")
            if item_type:
                item_types[item_type] += 1
                if item_type in FORBIDDEN_ITEM_TYPES:
                    forbidden_events += 1
    if forbidden_events:
        raise ValueError(f"Forbidden tool events found: {forbidden_events}")
    workspace = run_dir / "codex_empty_workspace"
    if not workspace.is_dir() or any(workspace.iterdir()):
        raise ValueError("Read-only Codex workspace is absent or non-empty")

    completed_rounds = len(round_files)
    return {
        "audit_status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": "outcome-blind cohort-batched round inference",
        "completed_rounds": completed_rounds,
        "remaining_rounds": 122 - completed_rounds,
        "completed_patient_outputs": patient_output_count,
        "expected_patient_outputs": 20_252,
        "duplicate_completed_rounds": 0,
        "rounds_by_shot": {str(key): shot_counts[key] for key in sorted(shot_counts)},
        "attempt_counts": {str(key): attempt_counts[key] for key in sorted(attempt_counts)},
        "model": model,
        "reasoning_effort": effort,
        "ephemeral_sessions": True,
        "provider_sampling_seed_supported": False,
        "forbidden_tool_events": 0,
        "transport_error_events_preserved": transport_errors,
        "observed_event_item_types": dict(sorted(item_types.items())),
        "failed_attempt_records": len(failure_entries),
        "raw_responses_verified": completed_rounds,
        "round_output_files_verified": completed_rounds,
        "event_logs_verified": len(event_files) + len(archived_event_files),
        "state_files_verified": [path.name for path in all_state_paths],
        "execution_epochs": execution_epochs,
        "frozen_file_hashes": frozen_hashes,
        "ledger_summaries": ledger_summaries,
        "validation_outcomes_accessed": False,
        "design_boundary": "cohort-batched transductive inference",
        "audit_code_sha256": sha256_file(Path(__file__).resolve()),
        "python_version": sys.version.split()[0],
        "privacy_note": "Aggregate counts and hashes only; no outcomes or individual predictions.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-model", default="gpt-5.6-sol")
    parser.add_argument("--expected-effort", default="high")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    report = audit(run_dir, args.expected_model, args.expected_effort)
    audit_dir = run_dir / "audit"
    audit_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = audit_dir / f"inference_audit_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_hash = sha256_file(report_path)
    append_audit_index(audit_dir / "audit_index.jsonl", report_path, report_hash)
    print(
        json.dumps(
            {
                "status": report["audit_status"],
                "completed_rounds": report["completed_rounds"],
                "remaining_rounds": report["remaining_rounds"],
                "patient_outputs": report["completed_patient_outputs"],
                "forbidden_tool_events": 0,
                "report": str(report_path),
                "report_sha256": report_hash,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
