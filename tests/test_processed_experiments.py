"""Offline checks with a real, small, randomly initialized RoBERTa backbone."""

import csv
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from transformers import RobertaConfig, RobertaModel

from experiments.dl import roberta_experiment as runner
from src.data.processed import load_processed
from src.evaluation.thresholds import tune_thresholds
from src.models.dl.roberta_variants import BinaryFocalLoss
from src.models.dl.roberta_weighted_bce import compute_pos_weight
from src.training.masked_classifier import fit_masked_classifier
from src.training.devices import resolve_device
from src.models.dl.zoo import MODEL_ZOO


torch.set_num_threads(1)


def tiny_backbone(*args, **kwargs):
    return RobertaModel(RobertaConfig(vocab_size=32, hidden_size=768, num_hidden_layers=1,
                                     num_attention_heads=12, intermediate_size=32,
                                     max_position_embeddings=32, hidden_dropout_prob=0.0,
                                     attention_probs_dropout_prob=0.0))


class OfflineTokenizer:
    def __call__(self, text, **kwargs):
        length = kwargs["max_length"]
        ids = [0] + [3 + ord(c) % 29 for c in text[:length - 2]] + [2]
        mask = [1] * len(ids) + [0] * (length - len(ids))
        ids += [1] * (length - len(ids))
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)


def fixture(root):
    directory = root / "python"
    directory.mkdir()
    labels = ["a", "b", "c"]
    (directory / "label_cols.json").write_text(json.dumps(labels))
    for split, indices in (("train", range(10)), ("test", range(10, 12))):
        with (directory / f"{split}.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["comment_sentence_id", "class", "group_id", "text"] + labels)
            for i in indices:
                writer.writerow([i, f"class{i % 3}", i, f"sentence {i}", i % 2, (i // 2) % 2, 1])
    folds = [{"fold": k, "train_indices": [i for i in range(10) if i % 2 != k],
              "val_indices": [i for i in range(10) if i % 2 == k]} for k in range(2)]
    (directory / "folds.json").write_text(json.dumps(folds))
    return directory


class ProcessedExperimentTests(unittest.TestCase):
    def test_device_selection(self):
        self.assertEqual(str(resolve_device("cpu")), "cpu")
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(str(resolve_device("auto")), "cpu")
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                resolve_device("cuda:1")
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2):
            self.assertEqual(str(resolve_device("auto")), "cuda:0")
            self.assertEqual(str(resolve_device("cuda:1")), "cuda:1")
            with self.assertRaises(ValueError):
                resolve_device("cuda:2")
        with self.assertRaises(ValueError):
            resolve_device("invalid")

    def test_actual_processed_data(self):
        for language, counts in (("python", (2044, 511, 5)), ("java", (1934, 484, 7)),
                                 ("pharo", (1411, 354, 7))):
            with self.subTest(language=language):
                bundle = load_processed("data/processed", language)
                self.assertEqual((bundle.train_size, len(bundle) - bundle.train_size,
                                  len(bundle.label_names)), counts)
                self.assertTrue(bundle.train_mask[:bundle.train_size].all())
                self.assertFalse(bundle.test_mask[:bundle.train_size].any())
                self.assertTrue(bundle.test_mask[bundle.train_size:].all())

    def test_loader_rejects_leakage_and_bad_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = fixture(root)
            original = (directory / "test.csv").read_text()
            (directory / "test.csv").write_text(original.replace("sentence 10", "sentence 0"))
            with self.assertRaises(ValueError):
                load_processed(root, "python")
            (directory / "test.csv").write_text(original)
            folds = json.loads((directory / "folds.json").read_text())
            folds[0]["val_indices"][0] = 99
            (directory / "folds.json").write_text(json.dumps(folds))
            with self.assertRaises(ValueError):
                load_processed(root, "python")

    def test_focal_and_weights(self):
        logits = torch.tensor([[100., -100.], [-1., 1.]], requires_grad=True)
        labels = torch.tensor([[0., 1.], [1., 0.]])
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        self.assertTrue(torch.allclose(BinaryFocalLoss(gamma=0)(logits, labels), bce))
        loss = BinaryFocalLoss(gamma=2, alpha=0.25)(logits, labels).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        y = torch.tensor([[1., 0.], [0., 1.], [0., 1.]])
        self.assertTrue(torch.allclose(compute_pos_weight(y, torch.ones_like(y)), torch.tensor([2., .5])))

    def test_thresholds(self):
        scores = torch.tensor([[.3, .6], [.2, .7], [.1, .1]])
        labels = torch.tensor([[1., 0.], [1., 1.], [0., 0.]])
        tuned = tune_thresholds(scores, labels, [.2, .5, .7])
        self.assertTrue(torch.allclose(tuned, torch.tensor([.2, .7])))

    def test_best_epoch_restored(self):
        class TinyClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, **kwargs):
                from types import SimpleNamespace
                return SimpleNamespace(loss=self.weight.square())

        model = TinyClassifier()
        batch = {k: torch.ones((1, 1)) for k in ("input_ids", "attention_mask", "labels", "label_mask")}
        snapshots = []

        def evaluate(*args, **kwargs):
            snapshots.append(model.weight.detach().clone())
            return {"micro_f1": 1.0 if len(snapshots) == 1 else 0.0}

        training = dict(learning_rate=.1, weight_decay=0., gradient_accumulation_steps=1,
                        epochs=2, warmup_ratio=0., mixed_precision=False, max_grad_norm=1.,
                        early_stopping_patience=2)
        with patch("src.training.masked_classifier.evaluate_masked_multilabel", side_effect=evaluate):
            result = fit_masked_classifier(model, [batch], [batch], device=torch.device("cpu"),
                                           training=training, threshold=.5, label_names=("a",),
                                           report_per_label=True)
        self.assertEqual(result["best_epoch"], 1)
        self.assertTrue(torch.equal(model.weight, snapshots[0]))
        self.assertFalse(torch.equal(snapshots[0], snapshots[1]))

    def test_all_twelve_experiments_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            with patch("src.models.dl.roberta_baseline.RobertaModel.from_pretrained", side_effect=tiny_backbone) as backbone_factory, \
                 patch.object(runner.AutoTokenizer, "from_pretrained", return_value=OfflineTokenizer()) as token_factory, \
                 patch.object(runner, "save_training_plots", return_value=[]):
                for key, spec in MODEL_ZOO.items():
                    number = int(key[1:])
                    with self.subTest(experiment=number):
                        entry = importlib.import_module(f"experiments.dl.train_exp{number}")
                        config = yaml.safe_load(Path(entry.DEFAULT_CONFIG).read_text())
                        config["training"].update(epochs=1, batch_size=5, eval_batch_size=5)
                        config["tokenizer"]["max_length"] = 8
                        config_path = root / f"exp{number}.yaml"
                        config_path.write_text(yaml.safe_dump(config))
                        output = root / f"out{number}"
                        result = entry.run(config_path, data_dir=root, output_dir=output, device="cpu")
                        self.assertEqual(result["device"], "cpu")
                        backbone_factory.assert_called_with(config["model"]["name"])
                        token_factory.assert_called_with(config["model"]["name"], use_fast=True)
                        self.assertEqual(result["labels"], ["a", "b", "c"])
                        self.assertEqual(result["train_sentences"], 10)
                        self.assertEqual(result["test_sentences"], 2)
                        self.assertEqual(result["test"]["evaluated_targets"], 6)
                        self.assertEqual(result["fold_count"], 2)
                        self.assertEqual(result["folds"][0]["validation_indices"], [0, 2, 4, 6, 8])
                        self.assertTrue((output / "best_model.pt").is_file())
                        self.assertTrue((output / "metrics.json").is_file())
                        self.assertEqual(len(json.loads((output / "thresholds.json").read_text())), 3)
                        self.assertTrue(all(torch.isfinite(torch.tensor(f["history"][0]["train_loss"]))
                                            for f in result["folds"]))
                        if number == 2:
                            self.assertEqual(result["folds"][0]["pos_weight"], [1.0, 1.5, 1.0])
                        if number in (9, 12):
                            self.assertEqual(result["folds"][0]["label_graph"]["source"], "current_train_indices")
                            self.assertEqual(len(result["final_training"]["label_graph"]["adjacency"]), 3)
                            graph = json.loads((output / "label_graph.json").read_text())
                            self.assertEqual(graph["num_nodes"], 3)
                            self.assertEqual(graph["labels"], ["a", "b", "c"])
                            self.assertEqual(graph["language"], "python")
                            self.assertEqual(graph["train_sentences"], 10)
                        if number == 11:
                            self.assertEqual(config["model"]["name"], "microsoft/codebert-base")
                        if number == 4:
                            self.assertEqual(result["threshold_tuning"]["source"], "out_of_fold_train")
                            oof = torch.load(output / "oof_predictions.pt", weights_only=True)
                            self.assertEqual(oof["scores"].shape, (10, 3))
                            self.assertTrue(torch.isfinite(oof["scores"]).all())

    def test_attention_masks_padding_and_contrastive_gradients(self):
        with patch("src.models.dl.roberta_baseline.RobertaModel.from_pretrained", side_effect=tiny_backbone):
            from src.models.dl.roberta_variants import RobertaLabelAttention
            model = RobertaLabelAttention(num_labels=3, contrastive_weight=.1)
            model.eval()
            x = torch.tensor([[0, 5, 2, 1, 1]])
            mask = torch.tensor([[1, 1, 1, 0, 0]])
            changed = x.clone()
            changed[:, 3:] = 9
            a = model(x, mask).logits
            b = model(changed, mask).logits
            self.assertTrue(torch.allclose(a, b, atol=1e-6))
            loss = model(x, mask, labels=torch.tensor([[1., 0., 1.]])).loss
            loss.backward()
            for parameter in (model.label_queries, model.sentence_projection.weight,
                              model.attention_projection.weight, model.label_classifier):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
