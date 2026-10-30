"""Registry consistency and analysis of multiple runs with different data."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.evaluation.analyze_results import analyze_results, collect_results
from src.models.dl.zoo import MODEL_ZOO


class ModelZooTests(unittest.TestCase):
    def test_registry_matches_configs(self):
        self.assertEqual(list(MODEL_ZOO), [f"E{i}" for i in range(1, 13)])
        for spec in MODEL_ZOO.values():
            config = yaml.safe_load(Path(spec.config).read_text())
            self.assertEqual(config["variant"], spec.variant)

    def test_report_preserves_runs_and_separates_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "runs"
            for i, (language, fingerprint, score) in enumerate([
                    ("python", "same-data", .7), ("python", "same-data", .8),
                    ("python", "changed-data", .6), ("java", "same-data", .9),
                    ("python", None, .5)]):
                folder = results / str(i)
                folder.mkdir(parents=True)
                metrics = {"experiment": f"exp{i + 1}", "variant": list(MODEL_ZOO.values())[i].variant,
                           "language": language, "labels": ["a", "b"], "split_source": "processed",
                           "data_fingerprint": fingerprint, "seed": 42,
                           "cross_validation": {"micro_f1": {"mean": score, "std": .02},
                                                "macro_f1": {"mean": score - .1, "std": .03}},
                           "test": {"micro_f1": score, "macro_f1": score - .1,
                                    "macro_average_precision": None,
                                    "per_label": {"a": {"f1": .5, "support": 10}, "b": {"f1": .8}}}}
                if i == 4:
                    # A completed run can intentionally disable test evaluation.
                    metrics.pop("test")
                (folder / "metrics.json").write_text(json.dumps(metrics))
            (results / "metrics.json").write_text("{broken json")
            output = root / "report"
            summary = analyze_results([results, results / "0"], output)
            self.assertEqual(summary["runs"], 5)  # Deduplicate overlapping input roots.
            self.assertEqual(summary["cohorts"], 4)
            self.assertEqual(len(summary["skipped"]), 1)
            with (output / "leaderboard.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row["checkpoint_exists"] == "False" for row in rows))
            with (output / "per_label.csv").open() as file:
                self.assertEqual(len(list(csv.DictReader(file))), 10)
            report = (output / "report.md").read_text()
            self.assertIn("Historical run", report)
            self.assertIn("held-out reporting only", report)
            # Descending CV order within the matching-data cohort.
            self.assertLess(report.index("| R002 |"), report.index("| R001 |"))
            self.assertEqual(len(list(output.glob("*/comparison.png"))), 4)
            self.assertEqual(len(list(output.glob("*/per_label_f1.png"))), 3)
            self.assertTrue(all(path.stat().st_size > 1000 for path in output.glob("*/*.png")))

    def test_no_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "No completed"):
                analyze_results([tmp], Path(tmp) / "report")
            with self.assertRaisesRegex(ValueError, "does not exist"):
                collect_results([Path(tmp) / "missing"])


if __name__ == "__main__":
    unittest.main()
