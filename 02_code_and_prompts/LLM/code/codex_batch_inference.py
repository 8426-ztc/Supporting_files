"""Resume-safe, outcome-blind Codex inference for the frozen MPR90 prompts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_chain(path: Path, event: str, details: dict[str, object]) -> None:
    previous_hash = "GENESIS"
    if path.exists():
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if lines:
            previous_hash = json.loads(lines[-1])["record_hash"]
    record = {
        "timestamp_utc": utc_now(),
        "event": event,
        "previous_record_hash": previous_hash,
        "details": details,
    }
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
    record["record_hash"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_codex(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    command = shutil.which("codex")
    if command:
        return Path(command).resolve()
    candidates = sorted(
        Path(os.environ["LOCALAPPDATA"]).glob("OpenAI/Codex/bin/*/codex.exe"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Codex CLI was not found. Pass --codex explicitly.")
    return candidates[0].resolve()


def codex_version(codex: Path, environment: dict[str, str]) -> str:
    result = subprocess.run(
        [str(codex), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        check=True,
        timeout=30,
    )
    return result.stdout.strip() or result.stderr.strip()


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    keys = [(record["condition_id"], int(record["batch"])) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("prompts.jsonl contains duplicate condition/batch keys.")
    return records


def expected_ids(path: Path) -> dict[tuple[str, int], str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {}
    for row in rows:
        key = (row["condition_id"], int(row["batch"]))
        if key in expected:
            raise ValueError("Each prompt must contain exactly one validation patient.")
        expected[key] = row["public_id"]
    return expected


def completed_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["condition_id"], int(row["batch"]))
            for row in csv.DictReader(handle)
        }


def validate_response(text: str, expected_id: str) -> dict[str, object]:
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"predictions"}:
        raise ValueError("The transport response must contain only predictions.")
    predictions = value["predictions"]
    if not isinstance(predictions, list) or len(predictions) != 1 or not isinstance(predictions[0], dict):
        raise ValueError("predictions must be a one-object JSON array.")
    item = predictions[0]
    if set(item) != {"id", "pred_label", "prob_mpr90"}:
        raise ValueError("The final response has missing or additional fields.")
    if item["id"] != expected_id:
        raise ValueError("The returned ID does not match the frozen prompt.")
    if item["pred_label"] not in (0, 1):
        raise ValueError("pred_label must be 0 or 1.")
    probability = float(item["prob_mpr90"])
    if not 0 <= probability <= 1:
        raise ValueError("prob_mpr90 must be in [0, 1].")
    if int(probability >= 0.5) != item["pred_label"]:
        raise ValueError("pred_label conflicts with the frozen 0.5 threshold.")
    return {
        "id": expected_id,
        "pred_label": int(item["pred_label"]),
        "prob_mpr90": probability,
    }


def reject_tool_events(path: Path) -> None:
    forbidden = {
        "command_execution",
        "file_change",
        "mcp_tool_call",
        "web_search",
        "computer_initialize_state",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        item = event.get("item") if isinstance(event, dict) else None
        item_type = item.get("type") if isinstance(item, dict) else None
        if item_type in forbidden:
            raise ValueError(f"Forbidden tool event detected: {item_type}")


def append_prediction(path: Path, row: dict[str, object]) -> None:
    fields = [
        "condition_id",
        "shot_size",
        "replicate",
        "batch",
        "public_id",
        "pred_label",
        "prob_mpr90",
        "attempt",
        "model",
        "prompt_sha256",
        "response_sha256",
        "completed_utc",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    if not args.confirm_training_disabled:
        raise ValueError(
            "Refusing to send patient features. Confirm the account/workspace training "
            "control is disabled with --confirm-training-disabled."
        )
    run_dir = args.run_dir.resolve()
    operator_dir = run_dir / "operator"
    prediction_dir = run_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require 0 <= shard_index < shard_count.")
    tag = "" if args.shard_count == 1 else f"_shard{args.shard_index:02d}of{args.shard_count:02d}"
    raw_dir = prediction_dir / f"codex_raw{tag}"
    raw_dir.mkdir(exist_ok=True)
    event_dir = prediction_dir / f"codex_events{tag}"
    event_dir.mkdir(exist_ok=True)
    sandbox_dir = run_dir / "codex_empty_workspace"
    sandbox_dir.mkdir(exist_ok=True)
    if any(sandbox_dir.iterdir()):
        raise ValueError("The Codex inference workspace must remain empty.")

    prompts_file = operator_dir / "prompts.jsonl"
    expected_file = operator_dir / "expected_predictions.csv"
    schema_file = args.schema.resolve()
    partial_file = prediction_dir / f"llm_predictions_PARTIAL{tag}.csv"
    ledger = prediction_dir / f"codex_inference_ledger{tag}.jsonl"
    state_file = prediction_dir / f"codex_inference_state{tag}.json"
    records = load_records(prompts_file)
    expected = expected_ids(expected_file)
    if len(records) != len(expected):
        raise ValueError("Prompt and expected-prediction counts differ.")

    environment = os.environ.copy()
    if not environment.get("HOME") and environment.get("USERPROFILE"):
        environment["HOME"] = environment["USERPROFILE"]
    if not environment.get("CODEX_HOME") and environment.get("USERPROFILE"):
        environment["CODEX_HOME"] = str(Path(environment["USERPROFILE"]) / ".codex")
    codex = find_codex(args.codex)
    version = codex_version(codex, environment)
    state = {
        "created_utc": utc_now(),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "codex_cli": str(codex),
        "codex_version": version,
        "ephemeral_sessions": True,
        "sandbox": "read-only empty workspace",
        "training_control_confirmed_by_user": True,
        "prompts_sha256": sha256(prompts_file),
        "expected_predictions_sha256": sha256(expected_file),
        "schema_sha256": sha256(schema_file),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
    }
    if state_file.exists():
        previous = json.loads(state_file.read_text(encoding="utf-8"))
        immutable_fields = [
            "model",
            "reasoning_effort",
            "codex_cli",
            "codex_version",
            "prompts_sha256",
            "expected_predictions_sha256",
            "schema_sha256",
            "runner_sha256",
            "shard_count",
            "shard_index",
        ]
        changed = [field for field in immutable_fields if previous[field] != state[field]]
        if changed:
            raise ValueError(f"Frozen inference settings changed: {changed}")
    else:
        write_json(state_file, state)
        append_chain(ledger, "INFERENCE_SETTINGS_FROZEN", state)

    completed = set()
    for existing_partial in prediction_dir.glob("llm_predictions_PARTIAL*.csv"):
        completed.update(completed_keys(existing_partial))
    sharded_records = [
        record
        for index, record in enumerate(records)
        if index % args.shard_count == args.shard_index
    ]
    pending = [
        record
        for record in sharded_records
        if (record["condition_id"], int(record["batch"])) not in completed
    ]
    if args.max_prompts is not None:
        pending = pending[: args.max_prompts]
    if args.dry_run:
        print(json.dumps({"total": len(records), "completed": len(completed), "selected": len(pending)}))
        return

    for record in pending:
        key = (record["condition_id"], int(record["batch"]))
        expected_id = expected[key]
        stem = f"{record['condition_id']}_batch{int(record['batch']):03d}"
        prompt = (
            record["system_prompt"]
            + "\n\n"
            + record["user_prompt"]
            + "\n\nThis is a frozen, outcome-blind evaluation. Do not use tools, files, "
            "web search, memory, or external information. For transport-schema "
            "compatibility, wrap the required one-object JSON array in an object with "
            "the single key predictions. Return only that JSON object."
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        success = False
        for attempt in range(1, args.max_attempts + 1):
            final_file = raw_dir / f"{stem}_attempt{attempt:02d}.json"
            events_file = event_dir / f"{stem}_attempt{attempt:02d}.jsonl"
            command = [
                str(codex),
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--cd",
                str(sandbox_dir),
                "--model",
                args.model,
                "-c",
                f'model_reasoning_effort="{args.reasoning_effort}"',
                "--output-schema",
                str(schema_file),
                "--json",
                "--output-last-message",
                str(final_file),
                "-",
            ]
            result = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
                timeout=args.timeout_seconds,
            )
            events_file.write_text(result.stdout, encoding="utf-8")
            error_file = event_dir / f"{stem}_attempt{attempt:02d}.stderr.txt"
            error_file.write_text(result.stderr, encoding="utf-8")
            try:
                if result.returncode != 0 or not final_file.exists():
                    raise RuntimeError(f"Codex exited with code {result.returncode}.")
                reject_tool_events(events_file)
                response = validate_response(final_file.read_text(encoding="utf-8"), expected_id)
                append_prediction(
                    partial_file,
                    {
                        "condition_id": record["condition_id"],
                        "shot_size": int(record["shot_size"]),
                        "replicate": int(record["replicate"]),
                        "batch": int(record["batch"]),
                        "public_id": response["id"],
                        "pred_label": response["pred_label"],
                        "prob_mpr90": response["prob_mpr90"],
                        "attempt": attempt,
                        "model": args.model,
                        "prompt_sha256": prompt_hash,
                        "response_sha256": sha256(final_file),
                        "completed_utc": utc_now(),
                    },
                )
                append_chain(
                    ledger,
                    "PROMPT_COMPLETE",
                    {
                        "condition_id": record["condition_id"],
                        "batch": int(record["batch"]),
                        "attempt": attempt,
                        "prompt_sha256": prompt_hash,
                        "response_sha256": sha256(final_file),
                        "events_sha256": sha256(events_file),
                    },
                )
                success = True
                break
            except Exception as error:
                append_chain(
                    ledger,
                    "PROMPT_ATTEMPT_FAILED",
                    {
                        "condition_id": record["condition_id"],
                        "batch": int(record["batch"]),
                        "attempt": attempt,
                        "error": str(error),
                        "events_sha256": sha256(events_file),
                        "stderr_sha256": sha256(error_file),
                    },
                )
        if not success:
            raise RuntimeError(f"Stopped after repeated failure for {stem}; progress is preserved.")

    total_completed_keys = set()
    for existing_partial in prediction_dir.glob("llm_predictions_PARTIAL*.csv"):
        total_completed_keys.update(completed_keys(existing_partial))
    total_completed = len(total_completed_keys)
    append_chain(
        ledger,
        "BATCH_COMPLETE",
        {"total_prompts": len(records), "completed_prompts": total_completed},
    )
    print(json.dumps({"total": len(records), "completed": total_completed}, ensure_ascii=False))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-dir", type=Path, required=True)
    value.add_argument("--model", required=True)
    value.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low")
    value.add_argument("--schema", type=Path, default=Path(__file__).with_name("llm_prediction_schema.json"))
    value.add_argument("--codex", type=Path)
    value.add_argument("--max-prompts", type=int)
    value.add_argument("--max-attempts", type=int, default=2)
    value.add_argument("--timeout-seconds", type=int, default=300)
    value.add_argument("--confirm-training-disabled", action="store_true")
    value.add_argument("--shard-count", type=int, default=1)
    value.add_argument("--shard-index", type=int, default=0)
    value.add_argument("--dry-run", action="store_true")
    return value


if __name__ == "__main__":
    run(parser().parse_args())
