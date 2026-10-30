"""Regression checks for E9 classifier collapse and premature F1-only stopping."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_processed_experiments import tiny_backbone
from src.models.dl.roberta_baseline import RobertaBaseline
from src.models.dl.advanced_variants import RobertaLabelGNN, compute_label_adjacency
from src.training.masked_classifier import fit_masked_classifier


class E9ConvergenceTests(unittest.TestCase):
    def test_initial_gnn_classifier_matches_baseline_logits(self):
        adjacency = compute_label_adjacency(torch.tensor([[1., 1., 0.], [0., 1., 1.]]))
        with patch("src.models.dl.roberta_baseline.RobertaModel.from_pretrained", side_effect=tiny_backbone):
            torch.manual_seed(42)
            baseline = RobertaBaseline(num_labels=3).eval()
            torch.manual_seed(42)
            gnn = RobertaLabelGNN(num_labels=3, adjacency=adjacency).eval()
        ids = torch.tensor([[0, 3, 4, 2], [0, 7, 9, 2]])
        with torch.no_grad():
            expected = baseline(ids, torch.ones_like(ids)).logits
            actual = gnn(ids, torch.ones_like(ids)).logits
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6))
        self.assertTrue(torch.equal(gnn.graph(gnn.label_embeddings), gnn.label_embeddings))
        self.assertEqual(int(gnn.head_version), 2)
        legacy = gnn.state_dict()
        legacy.pop("head_version")
        with self.assertRaisesRegex(RuntimeError, "head_version"):
            gnn.load_state_dict(legacy)

    def test_zero_initialized_graph_starts_learning_after_optimizer_step(self):
        with patch("src.models.dl.roberta_baseline.RobertaModel.from_pretrained", side_effect=tiny_backbone):
            model = RobertaLabelGNN(num_labels=3, adjacency=torch.eye(3))
        optimizer = torch.optim.AdamW([model.label_embeddings, model.label_bias, *model.graph.parameters()], lr=.01)
        torch.manual_seed(8)
        features = torch.randn(32, 768)
        labels = (features[:, :3] > .7).float()
        for _ in range(3):
            optimizer.zero_grad()
            nodes = model.graph(model.label_embeddings)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(features @ nodes.T + model.label_bias, labels)
            loss.backward()
            optimizer.step()
        self.assertGreater(model.graph.first.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.graph.second.weight.detach().abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(model.label_embeddings.grad).all())

    def fit_fake(self, metrics, *, epochs=10, min_epochs=1, patience=2, tiebreaker=None, max_run_epochs=None):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))

            def forward(self, **kwargs):
                return SimpleNamespace(loss=self.weight.square())

        batch = {key: torch.ones(1, 1) for key in ("input_ids", "attention_mask", "labels", "label_mask")}
        config = dict(learning_rate=.01, weight_decay=0., gradient_accumulation_steps=1,
                      epochs=epochs, min_epochs=min_epochs, warmup_ratio=.1, mixed_precision=False,
                      max_grad_norm=1., early_stopping_patience=patience,
                      validation_tiebreaker=tiebreaker, progress=False)
        with patch("src.training.masked_classifier.evaluate_masked_multilabel", side_effect=metrics):
            return fit_masked_classifier(Model(), [batch], [batch], device=torch.device("cpu"),
                                         training=config, threshold=.5, label_names=("a",),
                                         report_per_label=True, max_run_epochs=max_run_epochs)

    def test_ap_tiebreaker_preserves_learning_when_f1_is_zero(self):
        metrics = [{"micro_f1": 0., "micro_average_precision": ap} for ap in (.1, .2, .3, .25)]
        result = self.fit_fake(metrics, epochs=4, tiebreaker="micro_average_precision")
        self.assertEqual(result["completed_epochs"], 4)
        self.assertEqual(result["best_epoch"], 3)
        self.assertEqual(result["validation"]["micro_average_precision"], .3)

    def test_f1_has_priority_over_ap(self):
        result = self.fit_fake([
            {"micro_f1": .5, "micro_average_precision": .1},
            {"micro_f1": .4, "micro_average_precision": .99},
            {"micro_f1": .5, "micro_average_precision": .2},
        ], epochs=3, tiebreaker="micro_average_precision")
        self.assertEqual(result["best_epoch"], 3)
        self.assertEqual(result["validation"]["micro_f1"], .5)

    def test_min_epochs_and_diagnostic_schedule(self):
        metrics = [{"micro_f1": 0.}] * 10
        result = self.fit_fake(metrics, min_epochs=5, patience=1)
        self.assertEqual(result["completed_epochs"], 5)
        self.assertEqual(result["best_epoch"], 1)
        probe = self.fit_fake(metrics, min_epochs=5, max_run_epochs=1)
        self.assertEqual(probe["planned_epochs"], 10)
        self.assertEqual(probe["completed_epochs"], 1)


if __name__ == "__main__":
    unittest.main()
