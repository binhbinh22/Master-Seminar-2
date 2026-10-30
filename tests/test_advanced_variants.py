"""Numerical, gradient and train-only graph tests for E7–E12."""

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F
import yaml

from test_processed_experiments import tiny_backbone, fixture
from src.data.processed import load_processed
from src.models.dl.advanced_variants import (
    AsymmetricLoss, symmetric_bernoulli_kl, compute_label_adjacency,
    RobertaRDrop, RobertaFGM, RobertaLabelGNN, RobertaAttentionGNN,
    embedding_perturbation,
    describe_label_graph,
)
from src.models.dl.zoo import MODEL_ZOO, build_model


class AdvancedVariantTests(unittest.TestCase):
    def test_e9_matches_actual_processed_language_label_spaces(self):
        config = yaml.safe_load(Path(MODEL_ZOO["E9"].config).read_text())
        with patch("src.models.dl.roberta_baseline.RobertaModel.from_pretrained", side_effect=tiny_backbone):
            for language, count in (("python", 5), ("java", 7), ("pharo", 7)):
                with self.subTest(language=language):
                    bundle = load_processed("data/processed", language)
                    indices = bundle.folds[0]["train_indices"]
                    model = build_model(config, bundle, indices)
                    self.assertEqual(model.num_labels, count)
                    self.assertEqual(model.graph.adjacency.shape, (count, count))
                    expected = compute_label_adjacency(bundle.labels[indices], config["model"]["graph_threshold"])
                    self.assertTrue(torch.equal(model.graph.adjacency, expected))
                    metadata = describe_label_graph(model.graph.adjacency, bundle.label_names,
                        language=language, train_size=len(indices), threshold=config["model"]["graph_threshold"])
                    self.assertEqual(metadata["labels"], list(bundle.label_names))
                    self.assertEqual(metadata["num_nodes"], count)
                    self.assertEqual(metadata["train_sentences"], len(indices))
                    x = torch.tensor([[0, 3, 2]])
                    self.assertEqual(model(x, torch.ones_like(x)).logits.shape, (1, count))

    def test_e9_rejects_test_rows_and_mixed_language_graphs(self):
        config = yaml.safe_load(Path(MODEL_ZOO["E9"].config).read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            bundle = load_processed(root, "python")
            for indices in ([], [0, 0], [bundle.train_size], [-1]):
                with self.subTest(indices=indices), self.assertRaises(ValueError):
                    build_model(config, bundle, indices)
            languages = list(bundle.languages)
            languages[1] = "java"
            with self.assertRaisesRegex(ValueError, "one language"):
                build_model(config, replace(bundle, languages=tuple(languages)), [0, 1])

    def test_asl_bce_limit_and_easy_negatives(self):
        z = torch.tensor([[100., -100., 1., -1.]], requires_grad=True)
        y = torch.tensor([[0., 1., 1., 0.]])
        loss = AsymmetricLoss(gamma_pos=0, gamma_neg=0, clip=0)(z, y)
        self.assertTrue(torch.allclose(loss, F.binary_cross_entropy_with_logits(z, y, reduction="none")))
        loss = AsymmetricLoss()(z, y)
        self.assertTrue(torch.isfinite(loss).all())
        loss.mean().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        easy = torch.tensor([-10.], requires_grad=True)
        easy_loss = AsymmetricLoss()(easy, torch.zeros_like(easy))
        self.assertEqual(easy_loss.item(), 0.)
        easy_loss.sum().backward()
        self.assertEqual(easy.grad.item(), 0.)

    def test_asl_matches_probability_definition(self):
        z = torch.tensor([[-2., -.1, .5, 2.]], requires_grad=True)
        y = torch.tensor([[0., 1., 0., 1.]])
        p = z.sigmoid()
        negative = (1 - p + .05).clamp(max=1)
        probability = p * y + negative * (1 - y)
        weight = (1 - probability).pow(y + 4 * (1 - y))
        expected = -weight * (y * p.log() + (1 - y) * negative.log())
        self.assertTrue(torch.allclose(AsymmetricLoss()(z, y), expected))
        # Loss does not change ambient autograd state, including under no_grad.
        with torch.no_grad():
            AsymmetricLoss()(z, y)
            self.assertFalse(torch.is_grad_enabled())

    def test_bernoulli_kl_definition_mask_and_extreme_logits(self):
        a = torch.tensor([[.7, -1.2]], requires_grad=True)
        b = torch.tensor([[-.3, .8]], requires_grad=True)
        p, q = torch.distributions.Bernoulli(logits=a), torch.distributions.Bernoulli(logits=b)
        expected = .5 * (torch.distributions.kl_divergence(p, q) + torch.distributions.kl_divergence(q, p))
        actual = symmetric_bernoulli_kl(a, b)
        self.assertTrue(torch.allclose(actual, expected.mean(), atol=1e-7))
        self.assertEqual(symmetric_bernoulli_kl(a, a).item(), 0.)
        self.assertTrue(torch.allclose(actual, symmetric_bernoulli_kl(b, a)))
        actual.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertTrue(torch.isfinite(b.grad).all())
        extreme = symmetric_bernoulli_kl(torch.tensor([[100., -100.]]), torch.tensor([[-100., 100.]]))
        self.assertTrue(torch.isfinite(extreme))
        mask = torch.tensor([[True, False]])
        self.assertTrue(torch.allclose(symmetric_bernoulli_kl(a, b, mask), expected[0, 0]))

    def test_rdrop_uses_two_stochastic_passes_only_in_training(self):
        with patch("src.models.dl.roberta_baseline.RobertaModel.from_pretrained", side_effect=tiny_backbone):
            model = RobertaRDrop(num_labels=3)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = .3
        logits = []
        hook = model.classifier.register_forward_hook(lambda module, args, output: logits.append(output))
        x = torch.tensor([[0, 3, 4, 2]])
        y = torch.tensor([[1., 0., 1.]])
        out = model(x, torch.ones_like(x), y)
        self.assertEqual(len(logits), 2)
        self.assertFalse(torch.equal(*logits))
        expected = .5 * sum(F.binary_cross_entropy_with_logits(z, y) for z in logits)
        expected += symmetric_bernoulli_kl(*logits)
        self.assertTrue(torch.allclose(out.loss, expected))
        out.loss.backward()
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())
        logits.clear()
        model.eval()
        with torch.no_grad():
            model(x, torch.ones_like(x))
        self.assertEqual(len(logits), 1)
        hook.remove()

    def test_graph_normalization_and_absent_label(self):
        y = torch.tensor([[1., 1., 0.], [1., 0., 0.], [0., 1., 0.]])
        adjacency = compute_label_adjacency(y)
        expected = torch.tensor([[2/3, 1/3, 0.], [1/3, 2/3, 0.], [0., 0., 1.]])
        self.assertTrue(torch.allclose(adjacency, expected))
        self.assertTrue(torch.equal(adjacency, adjacency.T))
        self.assertTrue(torch.isfinite(compute_label_adjacency(torch.zeros_like(y))).all())

    def test_graph_never_reads_validation_or_test_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            bundle = load_processed(root, "python")
            indices = bundle.folds[0]["train_indices"]
            excluded = [i for i in range(len(bundle)) if i not in indices]
            labels = bundle.labels.clone()
            labels[excluded] = 1 - labels[excluded]
            with patch("src.models.dl.roberta_baseline.RobertaModel.from_pretrained", side_effect=tiny_backbone):
                for key in ("E9", "E12"):
                    config = yaml.safe_load(Path(MODEL_ZOO[key].config).read_text())
                    first = build_model(config, bundle, indices)
                    second = build_model(config, replace(bundle, labels=labels), indices)
                    self.assertTrue(torch.equal(first.graph.adjacency, second.graph.adjacency))

    def test_graph_heads_support_nineteen_labels_and_receive_gradients(self):
        y = torch.zeros(4, 19)
        y[:, :3] = 1
        adjacency = compute_label_adjacency(y)
        x = torch.tensor([[0, 3, 2]])
        labels = torch.zeros(1, 19)
        labels[0, 0] = 1
        with patch("src.models.dl.roberta_baseline.RobertaModel.from_pretrained", side_effect=tiny_backbone):
            for cls in (RobertaLabelGNN, RobertaAttentionGNN):
                model = cls(adjacency=adjacency, num_labels=19)
                out = model(x, torch.ones_like(x), labels)
                self.assertEqual(out.logits.shape, (1, 19))
                out.loss.backward()
                for parameter in model.graph.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                state = model.state_dict()
                self.assertIn("graph.adjacency", state)

    def test_fgm_padding_norm_and_no_parameter_mutation(self):
        gradient = torch.ones(2, 3, 4)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        noise = embedding_perturbation(gradient, mask, .7)
        self.assertTrue(torch.allclose(noise.flatten(1).norm(dim=1), torch.full((2,), .7)))
        self.assertTrue((noise[~mask.bool()] == 0).all())
        self.assertTrue(torch.isfinite(embedding_perturbation(torch.zeros_like(gradient), mask, 1.)).all())
        with patch("src.models.dl.roberta_baseline.RobertaModel.from_pretrained", side_effect=tiny_backbone):
            model = RobertaFGM(num_labels=3, epsilon=.7)
        original = model.roberta.get_input_embeddings().weight.detach().clone()
        embeddings = []
        hook = model.roberta.register_forward_pre_hook(
            lambda module, args, kwargs: embeddings.append(kwargs["inputs_embeds"].detach().clone()),
            with_kwargs=True)
        x = torch.tensor([[0, 3, 2, 1]])
        attention_mask = torch.tensor([[1, 1, 1, 0]])
        out = model(x, attention_mask, torch.tensor([[1., 0., 1.]]))
        self.assertEqual(len(embeddings), 2)
        difference = embeddings[1] - embeddings[0]
        self.assertAlmostEqual(difference.norm().item(), .7, places=5)
        self.assertTrue((difference[:, 3] == 0).all())
        self.assertTrue(torch.equal(original, model.roberta.get_input_embeddings().weight))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        out.loss.backward()
        self.assertTrue(torch.isfinite(model.roberta.get_input_embeddings().weight.grad).all())
        hook.remove()
        model.eval()
        with torch.no_grad():
            self.assertEqual(model(x, attention_mask).logits.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
