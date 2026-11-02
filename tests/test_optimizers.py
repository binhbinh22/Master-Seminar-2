"""Check parameter coverage, layer rates, scheduler ratios and legacy AdamW."""

import copy
import unittest

import torch
from torch import nn
from transformers import RobertaConfig, RobertaModel, get_linear_schedule_with_warmup

from src.training.optimizers import build_optimizer, optimizer_summary


def tiny_model():
    model = nn.Module()
    model.roberta = RobertaModel(RobertaConfig(
        vocab_size=20, hidden_size=12, num_hidden_layers=3,
        num_attention_heads=3, intermediate_size=16))
    model.classifier = nn.Linear(12, 2)
    return model


class OptimizerTests(unittest.TestCase):
    def test_layer_order_parameter_coverage_and_frozen_parameters(self):
        model = tiny_model()
        model.roberta.pooler.dense.bias.requires_grad_(False)
        config = dict(learning_rate=2e-5, weight_decay=.01,
                      head_learning_rate=1e-4, layerwise_lr_decay=.9)
        optimizer = build_optimizer(model, config)
        groups = {g["group_name"]: g for g in optimizer.param_groups}
        self.assertAlmostEqual(groups["embeddings"]["lr"], 2e-5 * .9**3)
        for layer in range(3):
            self.assertAlmostEqual(groups[f"encoder_layer_{layer}"]["lr"], 2e-5 * .9**(2-layer))
        self.assertEqual(groups["head"]["lr"], 1e-4)
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        actual = [id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(expected, set(actual))
        self.assertEqual(len(actual), len(expected))
        self.assertEqual({id(p) for p in groups["head"]["params"]},
                         {id(p) for p in model.classifier.parameters()})
        summary = optimizer_summary(optimizer)
        self.assertEqual(sum(g["parameter_count"] for g in summary["groups"]),
                         sum(p.numel() for p in model.parameters() if p.requires_grad))

    def test_head_only_lr_keeps_all_encoder_rates_equal(self):
        optimizer = build_optimizer(tiny_model(), dict(
            learning_rate=2e-5, encoder_learning_rate=1e-5,
            head_learning_rate=1e-4, weight_decay=.01))
        for group in optimizer.param_groups:
            self.assertEqual(group["lr"], 1e-4 if group["group_name"] == "head" else 1e-5)

    def test_scheduler_preserves_group_ratios_and_initial_metadata(self):
        optimizer = build_optimizer(tiny_model(), dict(
            learning_rate=2e-5, head_learning_rate=1e-4,
            layerwise_lr_decay=.9, weight_decay=.01))
        summary = optimizer_summary(optimizer)
        initial = [g["lr"] for g in optimizer.param_groups]
        scheduler = get_linear_schedule_with_warmup(optimizer, 2, 6)
        for step in range(1, 7):
            optimizer.step()
            scheduler.step()
            multiplier = step / 2 if step < 2 else (6-step) / 4
            for group, base in zip(optimizer.param_groups, initial):
                self.assertAlmostEqual(group["lr"], base * multiplier)
        self.assertEqual([g["initial_learning_rate"] for g in summary["groups"]], initial)

    def test_legacy_optimizer_preserves_exact_update(self):
        first = nn.Linear(3, 2)
        second = copy.deepcopy(first)
        new = build_optimizer(first, dict(learning_rate=2e-5, weight_decay=.01))
        old = torch.optim.AdamW(second.parameters(), lr=2e-5, weight_decay=.01)
        self.assertEqual(len(new.param_groups), 1)
        for _ in range(2):
            for a, b in zip(first.parameters(), second.parameters()):
                a.grad = torch.ones_like(a)
                b.grad = torch.ones_like(b)
            new.step()
            old.step()
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_invalid_rates_and_unsupported_encoder_fail(self):
        model = tiny_model()
        base = dict(learning_rate=2e-5, weight_decay=.01)
        for key in ("learning_rate", "encoder_learning_rate", "head_learning_rate", "layerwise_lr_decay"):
            for value in (0, -1, float("nan"), float("inf"), True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    build_optimizer(model, {**base, key: value})
        with self.assertRaises(ValueError):
            build_optimizer(model, {**base, "layerwise_lr_decay": 1.1})
        with self.assertRaisesRegex(ValueError, "encoder"):
            build_optimizer(nn.Linear(2, 2), {**base, "head_learning_rate": 1e-4})


if __name__ == "__main__":
    unittest.main()
