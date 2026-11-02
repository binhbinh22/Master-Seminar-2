"""Offline checks of split integrity and the two independent training stages."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from experiments.dl import staged_experiment as runner
from src.data.prepare_two_stage import LANGUAGES, prepare, raw_to_rows
from src.data.two_stage import load_stage_data, population_fingerprint
from src.evaluation.two_stage_results import report_final, select_methods
from src.models.dl.zoo import MODEL_ZOO
from test_processed_experiments import OfflineTokenizer, tiny_backbone


def fixture(root):
    raw = root / "raw"
    raw.mkdir()
    labels = ["a", "b", "c"]
    for language in LANGUAGES:
        with (raw / f"{language}.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["comment_sentence_id", "class", "comment_sentence", "category", "instance_type", "partition"])
            for i in range(30):
                for j, label in enumerate(labels):
                    writer.writerow([i, f"class{i % 3}", f"sentence {i if i < 29 else 0}", label,
                                     int((i + j) % 3 == 0), i % 2])
    config = root / "dataset.yaml"
    config.write_text(yaml.safe_dump({"languages": {lang: {"categories": labels} for lang in LANGUAGES}}))
    prepare(raw, root / "data", folds=2, config_path=config)


class TwoStageTests(unittest.TestCase):
    def test_actual_data_matches_raw_and_reproduces(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepare("data/raw", tmp)
            for language in LANGUAGES:
                cv = load_stage_data("data/k_fold_data", language, "kfold")
                final = load_stage_data("data/full_training_data", language, "final")
                rows = raw_to_rows(Path("data/raw/nlbse2023") / f"{language}.csv", list(cv.label_names))
                self.assertEqual(cv.population_fingerprint, population_fingerprint(rows, list(cv.label_names)))
                self.assertEqual(cv.population_fingerprint, final.population_fingerprint)
                self.assertEqual(set(cv.sentence_ids), set(final.sentence_ids))
                self.assertEqual(len(cv.folds), 5)
                self.assertTrue(final.validation_mask[final.validation_indices].all())
                self.assertFalse(final.train_mask[final.validation_indices + final.test_indices].any())
                for folder, stage in (("k_fold_data", "kfold"), ("full_training_data", "final")):
                    reproduced = load_stage_data(Path(tmp) / folder, language, stage)
                    existing = cv if stage == "kfold" else final
                    self.assertEqual(existing.data_fingerprint, reproduced.data_fingerprint)

    def test_rejects_overlap_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                prepare(root / "raw", root / "data", folds=2, config_path=root / "dataset.yaml")
            directory = root / "data/k_fold_data/python"
            folds = json.loads((directory / "folds.json").read_text())
            folds[0]["val_indices"][0] = folds[0]["train_indices"][0]
            (directory / "folds.json").write_text(json.dumps(folds))
            with self.assertRaisesRegex(ValueError, "coverage"):
                load_stage_data(directory.parent, "python", "kfold")

    def test_screen_select_and_final_train_without_pretrained_downloads(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            configs = {}
            with patch("src.models.dl.roberta_baseline.RobertaModel.from_pretrained", side_effect=tiny_backbone), \
                 patch.object(runner.AutoTokenizer, "from_pretrained", return_value=OfflineTokenizer()), \
                 patch.object(runner, "save_training_plots", return_value=[]):
                for key in ("E2", "E4", "E9"):
                    config = yaml.safe_load(Path(MODEL_ZOO[key].config).read_text())
                    config["training"].update(epochs=1, min_epochs=1, batch_size=30, eval_batch_size=30, progress=False)
                    config["tokenizer"]["max_length"] = 8
                    configs[key] = root / f"{key}.yaml"
                    configs[key].write_text(yaml.safe_dump(config))
                    output = root / "screen" / key
                    with patch.object(runner, "evaluate_masked_multilabel") as test_evaluation:
                        cv = runner.run_staged(configs[key], language="python", stage="kfold",
                                               data_dir=root / "data/k_fold_data", output_dir=output, device="cpu")
                        test_evaluation.assert_not_called()
                    self.assertNotIn("test", cv)
                    self.assertEqual(cv["cv_population_sentences"], 30)
                    self.assertFalse((output / "best_model.pt").exists())
                    self.assertEqual(len(cv["folds"]), 2)
                selection_dir = root / "selection"
                manifest = select_methods([root / "screen"], selection_dir, top_k=3)
                self.assertEqual(len(manifest["selected"]["python"]), 3)
                for key in configs:
                    output = root / "final" / key
                    result = runner.run_staged(configs[key], language="python", stage="final",
                                               data_dir=root / "data/full_training_data", output_dir=output,
                                               device="cpu", selection=selection_dir / "selection.json")
                    self.assertEqual(result["fold_count"], 0)
                    self.assertNotIn("cross_validation", result)
                    self.assertEqual(result["test"]["evaluated_targets"], result["test_sentences"] * 3)
                    self.assertEqual(result["validation"]["evaluated_targets"], result["validation_sentences"] * 3)
                    self.assertEqual(result["train_sentences"] + result["validation_sentences"] + result["test_sentences"], 30)
                    self.assertTrue((output / "best_model.pt").exists())
                    if key == "E4":
                        self.assertEqual(result["threshold_tuning"]["source"], "stage2_validation")
                    if key == "E9":
                        self.assertEqual(result["training"]["label_graph"]["train_sentences"], result["train_sentences"])
                    with self.assertRaisesRegex(ValueError, "overwrite"):
                        runner.run_staged(configs[key], language="python", stage="final",
                                          data_dir=root / "data/full_training_data", output_dir=output, device="cpu")
                self.assertEqual(len(report_final([root / "final"], root / "report")), 3)
                config = yaml.safe_load(configs["E2"].read_text())
                config["training"]["learning_rate"] *= 2
                configs["E2"].write_text(yaml.safe_dump(config))
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    runner.run_staged(configs["E2"], language="python", stage="final",
                                      data_dir=root / "data/full_training_data", output_dir=root / "changed",
                                      device="cpu", selection=selection_dir / "selection.json")


if __name__ == "__main__":
    unittest.main()
