import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import llm_icl_benchmark as benchmark


def synthetic_data(path: Path, duplicate_subject: bool = False) -> None:
    rows = []
    for center, count, offset in [("yz", 50, 0), ("zy", 24, 100)]:
        for index in range(count):
            subject_id = index + offset
            rows.append(
                {
                    "id": 1 if duplicate_subject and center == "zy" and index == 0 else subject_id,
                    "center": center,
                    "bmi": 18.0 + 0.13 * subject_id,
                    "tumorsizeb": 10.0 + 0.77 * subject_id,
                    "ast": 20.0 + 0.31 * subject_id,
                    "tbil": 5.0 + 0.19 * subject_id,
                    "mpr": 0.95 if index % 3 == 0 else 0.50,
                    "private_note": f"never-export-{subject_id}",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


class BenchmarkTest(unittest.TestCase):
    def test_prepare_minimizes_and_separates_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_file = root / "data.csv"
            run_dir = root / "run"
            synthetic_data(data_file)
            benchmark.prepare_run(data_file, run_dir, primary_shot=20)

            operator_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in (run_dir / "operator").iterdir()
                if path.suffix in {".csv", ".json", ".jsonl"}
            )
            self.assertNotIn("private_note", operator_text)
            self.assertNotIn("never-export", operator_text)
            self.assertNotIn("validation_ground_truth", operator_text)
            validation = pd.read_csv(run_dir / "operator" / "validation_features_BLINDED.csv")
            self.assertEqual(list(validation.columns), ["public_id", *benchmark.FEATURES])
            self.assertTrue(validation["public_id"].str.startswith("V-").all())
            self.assertTrue((run_dir / "private" / "validation_ground_truth_LOCKED.csv").exists())

    def test_prepare_blocks_cross_cohort_subject_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_file = root / "data.csv"
            synthetic_data(data_file, duplicate_subject=True)
            with self.assertRaisesRegex(ValueError, "not globally unique"):
                benchmark.prepare_run(data_file, root / "run")

    def test_blinded_prediction_and_evaluation_flow(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_file = root / "data.csv"
            run_dir = root / "run"
            synthetic_data(data_file)
            benchmark.prepare_run(data_file, run_dir, primary_shot=20)
            benchmark.run_baseline(run_dir)

            expected = pd.read_csv(run_dir / "operator" / "expected_predictions.csv")
            features = pd.read_csv(run_dir / "operator" / "validation_features_BLINDED.csv")
            predictions = expected.merge(features[["public_id", "bmi"]], on="public_id")
            probability = 1.0 / (1.0 + np.exp(-(predictions["bmi"] - predictions["bmi"].median())))
            predictions["prob_mpr90"] = probability
            predictions["pred_label"] = (probability >= 0.5).astype(int)
            raw_predictions = root / "llm.csv"
            predictions[["condition_id", "public_id", "pred_label", "prob_mpr90"]].to_csv(
                raw_predictions, index=False
            )
            benchmark.validate_llm_predictions(run_dir, raw_predictions)
            evaluation_dir = benchmark.evaluate_run(run_dir, bootstrap_iterations=20)

            self.assertTrue((evaluation_dir / "baseline_metrics.csv").exists())
            self.assertTrue((evaluation_dir / "llm_metrics_by_replicate.csv").exists())
            audit = json.loads((evaluation_dir / "evaluation_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["independent_unit"], "validation patient")
            self.assertEqual(audit["primary_shot"], 20)

    def test_metrics_have_expected_limits(self):
        labels = np.array([0, 0, 1, 1])
        perfect = benchmark._metrics(labels, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
        reversed_scores = benchmark._metrics(labels, np.array([0.9, 0.8, 0.2, 0.1]), 0.5)
        self.assertEqual(perfect["auroc"], 1.0)
        self.assertEqual(perfect["auprc"], 1.0)
        self.assertEqual(reversed_scores["auroc"], 0.0)


if __name__ == "__main__":
    unittest.main()
