"""Execution epoch 2: resume cohort-batched rounds after a documented CLI update."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

from codex_batch_inference import (
    append_chain,
    codex_version,
    find_codex,
    reject_tool_events,
    sha256,
    utc_now,
    write_json,
)


ROUND_FIELDS = [
    "condition_id",
    "round_number",
    "shot_size",
    "replicate",
    "query_order",
    "cohort",
    "public_id",
    "label_exposed",
    "eligible_for_development_metric",
    "pred_label",
    "prob_mpr90",
    "attempt",
    "model",
    "reasoning_effort",
    "prompt_sha256",
    "response_sha256",
    "events_sha256",
    "completed_utc",
]
PROMPT_SUFFIX = (
    "This is a frozen, outcome-blind cohort-batched evaluation. Do not use tools, "
    "files, web search, memory, connectors, or external information. Do not omit, "
    "add, reorder, or rename any query ID. For transport-schema compatibility, "
    "wrap the required JSON array in an object with the single key predictions. "
    "Return only that JSON object."
)


def load_prompts(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    keys = [(row["condition_id"], int(row.get("batch", 1))) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate batched-round prompt key")
    return records


def load_expected(path: Path) -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["condition_id"], 1)
            grouped.setdefault(key, []).append(row)
    for key, rows in grouped.items():
        rows.sort(key=lambda row: int(row["query_order"]))
        if len(rows) != 166 or len({row["public_id"] for row in rows}) != 166:
            raise ValueError(f"Expected round does not contain 166 unique IDs: {key}")
    return grouped


def read_completed_round(path: Path, expected: dict[tuple[str, int], list[dict]]) -> tuple[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ROUND_FIELDS:
            raise ValueError(f"Unexpected completed-round schema: {path}")
        rows = list(reader)
    if len(rows) != 166:
        raise ValueError(f"Incomplete completed-round file: {path}")
    key = (rows[0]["condition_id"], 1)
    if key not in expected:
        raise ValueError(f"Unexpected completed-round key: {path}")
    if any(row["condition_id"] != key[0] for row in rows):
        raise ValueError(f"Mixed condition IDs in completed-round file: {path}")
    expected_ids = [row["public_id"] for row in expected[key]]
    if [row["public_id"] for row in rows] != expected_ids:
        raise ValueError(f"Completed-round ID order differs from frozen index: {path}")
    return key


def completed_keys(prediction_dir: Path, expected: dict[tuple[str, int], list[dict]]) -> set[tuple[str, int]]:
    keys = []
    for directory in prediction_dir.glob("round_predictions*"):
        if directory.is_dir():
            keys.extend(read_completed_round(path, expected) for path in directory.glob("*.csv"))
    if len(keys) != len(set(keys)):
        raise ValueError("A batched round has more than one completed output file")
    return set(keys)


def validate_response(path: Path, expected_rows: list[dict]) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"predictions"}:
        raise ValueError("Response must contain only the predictions key")
    predictions = value["predictions"]
    if not isinstance(predictions, list) or len(predictions) != 166:
        raise ValueError("Response must contain exactly 166 predictions")
    expected_ids = [row["public_id"] for row in expected_rows]
    returned_ids = []
    checked = []
    for item in predictions:
        if not isinstance(item, dict) or set(item) != {"id", "pred_label", "prob_mpr90"}:
            raise ValueError("A prediction has missing or additional fields")
        if item["pred_label"] not in (0, 1):
            raise ValueError("pred_label must be 0 or 1")
        probability = float(item["prob_mpr90"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError("prob_mpr90 must be in [0, 1]")
        if int(probability >= 0.5) != int(item["pred_label"]):
            raise ValueError("pred_label conflicts with the frozen 0.5 threshold")
        returned_ids.append(item["id"])
        checked.append(
            {
                "id": item["id"],
                "pred_label": int(item["pred_label"]),
                "prob_mpr90": probability,
            }
        )
    if returned_ids != expected_ids or len(set(returned_ids)) != 166:
        raise ValueError("Returned IDs are missing, duplicated, unexpected, or reordered")
    return checked


def write_round_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite a completed or temporary round: {path}")
    with temporary.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROUND_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> None:
    if not args.confirm_training_disabled:
        raise ValueError("Explicit confirmation that model improvement is disabled is required")
    run_dir = args.run_dir.resolve()
    operator_dir = run_dir / "operator"
    prediction_dir = run_dir / "predictions"
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require 0 <= shard_index < shard_count")
    tag = "" if args.shard_count == 1 else f"_shard{args.shard_index:02d}of{args.shard_count:02d}"
    raw_dir = prediction_dir / f"codex_raw{tag}"
    event_dir = prediction_dir / f"codex_events{tag}"
    round_dir = prediction_dir / f"round_predictions{tag}"
    for directory in [raw_dir, event_dir, round_dir]:
        directory.mkdir(exist_ok=True)
    sandbox_dir = run_dir / "codex_empty_workspace"
    sandbox_dir.mkdir(exist_ok=True)
    if any(sandbox_dir.iterdir()):
        raise ValueError("Codex inference workspace must remain empty")

    prompts_path = operator_dir / "prompts.jsonl"
    expected_path = operator_dir / "expected_predictions.csv"
    schema_path = args.schema.resolve()
    previous_state_path = prediction_dir / f"codex_inference_state{tag}.json"
    state_path = prediction_dir / f"codex_inference_state_epoch02{tag}.json"
    ledger_path = prediction_dir / f"codex_inference_ledger_epoch02{tag}.jsonl"
    addendum_path = Path(__file__).resolve().with_name("PREREGISTRATION_ADDENDUM_07.md")
    if not previous_state_path.is_file() or not addendum_path.is_file():
        raise ValueError("Execution epoch 2 requires the frozen epoch-1 state and addendum 07")
    prompts = load_prompts(prompts_path)
    expected = load_expected(expected_path)
    if {(row["condition_id"], 1) for row in prompts} != set(expected):
        raise ValueError("Prompt rounds and expected-output rounds differ")

    environment = os.environ.copy()
    if not environment.get("HOME") and environment.get("USERPROFILE"):
        environment["HOME"] = environment["USERPROFILE"]
    if not environment.get("CODEX_HOME") and environment.get("USERPROFILE"):
        environment["CODEX_HOME"] = str(Path(environment["USERPROFILE"]) / ".codex")
    codex = find_codex(args.codex)
    state = {
        "created_utc": utc_now(),
        "execution_epoch": 2,
        "runner_type": "cohort_batched_round",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "codex_cli": str(codex),
        "codex_version": codex_version(codex, environment),
        "ephemeral_sessions": True,
        "sandbox": "read-only empty workspace",
        "training_control_confirmed_by_user": True,
        "queries_per_round": 166,
        "provider_sampling_seed_supported": False,
        "prompts_sha256": sha256(prompts_path),
        "expected_predictions_sha256": sha256(expected_path),
        "schema_sha256": sha256(schema_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "runner_dependency_sha256": sha256(
            Path(__file__).resolve().with_name("codex_batch_inference.py")
        ),
        "previous_epoch_state_sha256": sha256(previous_state_path),
        "operational_change_addendum_sha256": sha256(addendum_path),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
    }
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        immutable = [key for key in state if key != "created_utc"]
        changed = [key for key in immutable if previous.get(key) != state[key]]
        if changed:
            raise ValueError(f"Frozen batched inference settings changed: {changed}")
    else:
        write_json(state_path, state)
        append_chain(ledger_path, "INFERENCE_SETTINGS_FROZEN", state)

    completed = completed_keys(prediction_dir, expected)
    selected = [
        row
        for index, row in enumerate(prompts)
        if index % args.shard_count == args.shard_index
        and (row["condition_id"], 1) not in completed
    ]
    if args.max_rounds is not None:
        selected = selected[: args.max_rounds]
    if args.dry_run:
        print(
            json.dumps(
                {"rounds": len(prompts), "completed": len(completed), "selected": len(selected)}
            )
        )
        return

    for record in selected:
        condition_id = record["condition_id"]
        key = (condition_id, 1)
        expected_rows = expected[key]
        stem = f"{condition_id}_batch001"
        prompt = record["system_prompt"] + "\n\n" + record["user_prompt"] + "\n\n" + PROMPT_SUFFIX
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        success = False
        for attempt in range(1, args.max_attempts + 1):
            final_path = raw_dir / f"{stem}_attempt{attempt:02d}.json"
            events_path = event_dir / f"{stem}_attempt{attempt:02d}.jsonl"
            stderr_path = event_dir / f"{stem}_attempt{attempt:02d}.stderr.txt"
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
                str(schema_path),
                "--json",
                "--output-last-message",
                str(final_path),
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
            events_path.write_text(result.stdout, encoding="utf-8")
            stderr_path.write_text(result.stderr, encoding="utf-8")
            try:
                if result.returncode != 0 or not final_path.exists():
                    raise RuntimeError(f"Codex exited with code {result.returncode}")
                reject_tool_events(events_path)
                predictions = validate_response(final_path, expected_rows)
                response_hash = sha256(final_path)
                events_hash = sha256(events_path)
                completed_utc = utc_now()
                output_rows = []
                for expected_row, prediction in zip(expected_rows, predictions):
                    output_rows.append(
                        {
                            **{key: expected_row[key] for key in [
                                "condition_id",
                                "round_number",
                                "shot_size",
                                "replicate",
                                "query_order",
                                "cohort",
                                "public_id",
                                "label_exposed",
                                "eligible_for_development_metric",
                            ]},
                            "pred_label": prediction["pred_label"],
                            "prob_mpr90": prediction["prob_mpr90"],
                            "attempt": attempt,
                            "model": args.model,
                            "reasoning_effort": args.reasoning_effort,
                            "prompt_sha256": prompt_hash,
                            "response_sha256": response_hash,
                            "events_sha256": events_hash,
                            "completed_utc": completed_utc,
                        }
                    )
                output_path = round_dir / f"{condition_id}.csv"
                write_round_atomic(output_path, output_rows)
                append_chain(
                    ledger_path,
                    "ROUND_COMPLETE",
                    {
                        "condition_id": condition_id,
                        "batch": 1,
                        "attempt": attempt,
                        "prediction_rows": 166,
                        "prompt_sha256": prompt_hash,
                        "response_sha256": response_hash,
                        "events_sha256": events_hash,
                        "round_output_sha256": sha256(output_path),
                    },
                )
                success = True
                break
            except Exception as error:
                append_chain(
                    ledger_path,
                    "PROMPT_ATTEMPT_FAILED",
                    {
                        "condition_id": condition_id,
                        "batch": 1,
                        "attempt": attempt,
                        "error": str(error),
                        "events_sha256": sha256(events_path),
                        "stderr_sha256": sha256(stderr_path),
                    },
                )
        if not success:
            raise RuntimeError(f"Stopped after repeated failure for {condition_id}")

    total_completed = len(completed_keys(prediction_dir, expected))
    append_chain(
        ledger_path,
        "BATCH_COMPLETE",
        {"total_rounds": len(prompts), "completed_rounds": total_completed},
    )
    print(json.dumps({"rounds": len(prompts), "completed": total_completed}))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-dir", type=Path, required=True)
    value.add_argument("--model", required=True)
    value.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="high")
    value.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).with_name("llm_round_prediction_schema.json"),
    )
    value.add_argument("--codex", type=Path)
    value.add_argument("--max-rounds", type=int)
    value.add_argument("--max-attempts", type=int, default=2)
    value.add_argument("--timeout-seconds", type=int, default=900)
    value.add_argument("--confirm-training-disabled", action="store_true")
    value.add_argument("--shard-count", type=int, default=1)
    value.add_argument("--shard-index", type=int, default=0)
    value.add_argument("--dry-run", action="store_true")
    return value


if __name__ == "__main__":
    run(parser().parse_args())
