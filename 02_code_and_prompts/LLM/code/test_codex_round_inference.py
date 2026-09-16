import csv
import json
import tempfile
import unittest
from pathlib import Path

from codex_round_inference import (
    ROUND_FIELDS,
    read_completed_round,
    validate_response,
    write_round_atomic,
)


def expected_rows():
    ids = [f"D-{index:016x}" for index in range(109)] + [
        f"V-{index:016x}" for index in range(57)
    ]
    return [
        {
            "condition_id": "shot000_rep01",
            "round_number": "1",
            "shot_size": "0",
            "replicate": "1",
            "query_order": str(index),
            "cohort": "development" if index <= 109 else "validation",
            "public_id": public_id,
            "label_exposed": "0",
            "eligible_for_development_metric": "1" if index <= 109 else "0",
        }
        for index, public_id in enumerate(ids, start=1)
    ]


class RoundInferenceTest(unittest.TestCase):
    def write_response(self, path: Path, rows: list[dict]) -> None:
        path.write_text(json.dumps({"predictions": rows}), encoding="utf-8")

    def test_validates_exact_order_and_threshold(self):
        expected = expected_rows()
        predictions = [
            {
                "id": row["public_id"],
                "pred_label": int(index % 2 == 0),
                "prob_mpr90": 0.75 if index % 2 == 0 else 0.25,
            }
            for index, row in enumerate(expected)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.json"
            self.write_response(path, predictions)
            checked = validate_response(path, expected)
        self.assertEqual(len(checked), 166)
        self.assertEqual([row["id"] for row in checked], [row["public_id"] for row in expected])

    def test_rejects_reordered_or_threshold_inconsistent_response(self):
        expected = expected_rows()
        predictions = [
            {"id": row["public_id"], "pred_label": 0, "prob_mpr90": 0.25}
            for row in expected
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.json"
            reordered = predictions.copy()
            reordered[0], reordered[1] = reordered[1], reordered[0]
            self.write_response(path, reordered)
            with self.assertRaises(ValueError):
                validate_response(path, expected)
            predictions[0]["pred_label"] = 1
            self.write_response(path, predictions)
            with self.assertRaises(ValueError):
                validate_response(path, expected)

    def test_atomic_round_file_requires_166_expected_ids(self):
        expected = expected_rows()
        rows = []
        for row in expected:
            rows.append(
                {
                    **row,
                    "pred_label": 0,
                    "prob_mpr90": 0.25,
                    "attempt": 1,
                    "model": "gpt-5.6-sol",
                    "reasoning_effort": "high",
                    "prompt_sha256": "a" * 64,
                    "response_sha256": "b" * 64,
                    "events_sha256": "c" * 64,
                    "completed_utc": "2026-08-30T00:00:00+00:00",
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shot000_rep01.csv"
            write_round_atomic(path, rows)
            key = read_completed_round(path, {("shot000_rep01", 1): expected})
            self.assertEqual(key, ("shot000_rep01", 1))
            with self.assertRaises(FileExistsError):
                write_round_atomic(path, rows)
            with path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(csv.DictReader(handle).fieldnames, ROUND_FIELDS)


if __name__ == "__main__":
    unittest.main()

