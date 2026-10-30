"""E7–E12: multi-label loss, consistency, graph and embedding regularization.

Method definitions and primary references are documented in docs/experiments_e7_e12.md.
All data-dependent graph statistics must be estimated from the current fold train.
"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

from src.models.dl.roberta_baseline import RobertaBaseline
from src.models.dl.roberta_variants import RobertaLabelAttention


def masked_mean(values, mask=None):
    if mask is None:
        return values.mean()
    if mask.shape != values.shape or not mask.bool().any():
        raise ValueError("label_mask must match loss shape and select targets")
    return values[mask.bool()].mean()


class AsymmetricLoss(nn.Module):
    """Unreduced sigmoid ASL, with clipped negatives and asymmetric focusing.

    Uses log-sigmoid for positive targets and the unclipped negative case, so
    saturated logits remain finite without globally changing grad-enabled state.
    """

    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, clip=0.05, detach_focal=True):
        super().__init__()
        if gamma_pos < 0 or gamma_neg < 0 or not 0 <= clip < 1:
            raise ValueError("Require nonnegative ASL gammas and 0 <= clip < 1")
        self.gamma_pos, self.gamma_neg = gamma_pos, gamma_neg
        self.clip, self.detach_focal = clip, detach_focal
        self.register_buffer("pos_weight", None)

    def forward(self, logits, labels):
        logits, labels = logits.float(), labels.float()
        positive = torch.sigmoid(logits)
        negative = (torch.sigmoid(-logits) + self.clip).clamp(max=1)
        log_negative = negative.clamp_min(1e-8).log() if self.clip else F.logsigmoid(-logits)
        log_likelihood = labels * F.logsigmoid(logits) + (1 - labels) * log_negative
        probability = labels * positive + (1 - labels) * negative
        gamma = labels * self.gamma_pos + (1 - labels) * self.gamma_neg
        weight = (1 - probability).pow(gamma)
        if self.detach_focal:
            weight = weight.detach()
        return -weight * log_likelihood


class RobertaASL(RobertaBaseline):
    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, clip=.05, detach_focal=True, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = AsymmetricLoss(gamma_pos, gamma_neg, clip, detach_focal)


def symmetric_bernoulli_kl(first, second, mask=None):
    """Symmetric KL for independent Bernoulli labels, never softmax over labels."""
    first, second = first.float(), second.float()
    p, q = first.sigmoid(), second.sigmoid()
    log_p, log_q = F.logsigmoid(first), F.logsigmoid(second)
    log_not_p, log_not_q = F.logsigmoid(-first), F.logsigmoid(-second)
    pq = p * (log_p - log_q) + (1 - p) * (log_not_p - log_not_q)
    qp = q * (log_q - log_p) + (1 - q) * (log_not_q - log_not_p)
    return masked_mean((.5 * (pq + qp)).clamp_min(0), mask)


class RobertaRDrop(RobertaBaseline):
    def __init__(self, rdrop_alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        if rdrop_alpha < 0:
            raise ValueError("rdrop_alpha must be nonnegative")
        self.rdrop_alpha = rdrop_alpha

    def forward(self, input_ids=None, attention_mask=None, labels=None, label_mask=None,
                inputs_embeds=None):
        first = super().forward(input_ids, attention_mask, labels, label_mask, inputs_embeds)
        if not self.training or labels is None:
            return first
        second = super().forward(input_ids, attention_mask, labels, label_mask, inputs_embeds)
        consistency = symmetric_bernoulli_kl(first.logits, second.logits, label_mask)
        return SequenceClassifierOutput(loss=.5 * (first.loss + second.loss) + self.rdrop_alpha * consistency,
                                        logits=.5 * (first.logits + second.logits))


def compute_label_adjacency(labels, threshold=.1):
    """Cosine co-occurrence graph, self loops, then symmetric degree normalization."""
    if labels.ndim != 2 or not labels.shape[0] or not labels.shape[1]:
        raise ValueError("Expected a non-empty sentence x label matrix")
    if not 0 <= threshold < 1 or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Require binary labels and graph threshold in [0, 1)")
    y = labels.detach().float()
    cooccurrence = y.T @ y
    support = y.sum(0)
    scale = (support[:, None] * support[None, :]).sqrt().clamp_min(1)
    adjacency = cooccurrence / scale
    adjacency.fill_diagonal_(0)
    adjacency = torch.where(adjacency >= threshold, adjacency, torch.zeros_like(adjacency))
    adjacency += torch.eye(y.shape[1], device=y.device)
    inverse_degree = adjacency.sum(-1).rsqrt()
    return inverse_degree[:, None] * adjacency * inverse_degree[None, :]


def describe_label_graph(adjacency, label_names, *, language, train_size, threshold):
    """Human-readable graph metadata in the exact processed label order."""
    adjacency = adjacency.detach().cpu()
    off_diagonal = adjacency > 0
    off_diagonal.fill_diagonal_(False)
    return {
        "source": "current_train_indices", "label_space": "per_language",
        "language": language, "labels": list(label_names), "num_nodes": len(label_names),
        "train_sentences": train_size, "threshold": threshold,
        "normalization": "symmetric_degree_normalized_cosine_cooccurrence_with_self_loops",
        "num_edges": int(off_diagonal.triu(diagonal=1).sum()),
        "isolated_labels": [name for name, connected in zip(label_names, off_diagonal.any(dim=1))
                            if not connected],
        "adjacency": adjacency.tolist(),
    }


class LabelGCN(nn.Module):
    """Two dense graph convolution layers with a residual path; supports B x C x D."""

    def __init__(self, adjacency, hidden_size=128, dimension=768):
        super().__init__()
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32).detach().clone()
        if (adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]
                or not torch.isfinite(adjacency).all() or (adjacency < 0).any()
                or hidden_size < 1):
            raise ValueError("Expected a finite, square, nonnegative graph and positive hidden size")
        self.register_buffer("adjacency", adjacency)
        self.first = nn.Linear(dimension, hidden_size)
        self.second = nn.Linear(hidden_size, dimension)

    def forward(self, nodes):
        adjacency = self.adjacency.to(dtype=nodes.dtype)
        hidden = F.relu(self.first(adjacency @ nodes))
        return nodes + self.second(adjacency @ hidden)


class RobertaLabelGNN(RobertaBaseline):
    """E9 v2: a baseline-scale classifier with a zero-initialized GCN residual.

    Starts exactly as the ordinary CLS linear head; graph correlations are
    learned as corrections rather than an initially dominant shared bias.
    """

    def __init__(self, adjacency, graph_hidden=128, **kwargs):
        super().__init__(**kwargs)
        if adjacency.shape != (self.num_labels, self.num_labels):
            raise ValueError("Graph size must match the label head")
        self.label_embeddings = nn.Parameter(self.classifier.weight.detach().clone())
        self.label_bias = nn.Parameter(self.classifier.bias.detach().clone())
        del self.classifier
        self.graph = LabelGCN(adjacency, graph_hidden)
        nn.init.zeros_(self.graph.second.weight)
        nn.init.zeros_(self.graph.second.bias)
        # Strict loading rejects old E9 checkpoints whose logit scale differed.
        self.register_buffer("head_version", torch.tensor(2))

    def forward(self, input_ids, attention_mask=None, labels=None, label_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        nodes = self.graph(self.label_embeddings)
        logits = outputs.last_hidden_state[:, 0] @ nodes.T + self.label_bias
        loss = None if labels is None else masked_mean(self.loss_fn(logits, labels.to(logits)), label_mask)
        return SequenceClassifierOutput(loss=loss, logits=logits)


class RobertaAttentionGNN(RobertaLabelAttention):
    """E12: label-aware token contexts, followed by graph message passing and BCE."""

    def __init__(self, adjacency, graph_hidden=128, **kwargs):
        super().__init__(**kwargs)
        if adjacency.shape != (self.num_labels, self.num_labels):
            raise ValueError("Graph size must match the label head")
        self.graph = LabelGCN(adjacency, graph_hidden)

    def refine_contexts(self, contexts):
        return self.graph(contexts)


def embedding_perturbation(gradient, attention_mask, epsilon):
    """Per-sentence L2-normalized FGM perturbation, excluding padding positions."""
    gradient = gradient.detach().float()
    if attention_mask is not None:
        gradient = gradient * attention_mask.unsqueeze(-1)
    norm = gradient.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
    return epsilon * gradient / norm


class RobertaFGM(RobertaBaseline):
    """E10: clean/adversarial BCE on input embeddings, without modifying parameters.

    autograd.grad obtains the current batch embedding gradient without polluting
    accumulated parameter gradients; the outer loop handles AMP/accumulation.
    """

    def __init__(self, epsilon=1.0, adversarial_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        if epsilon < 0 or adversarial_weight < 0:
            raise ValueError("FGM epsilon and adversarial weight must be nonnegative")
        self.epsilon, self.adversarial_weight = epsilon, adversarial_weight

    def forward(self, input_ids=None, attention_mask=None, labels=None, label_mask=None,
                inputs_embeds=None):
        if not self.training or labels is None or not self.epsilon or not self.adversarial_weight:
            return super().forward(input_ids, attention_mask, labels, label_mask, inputs_embeds)
        embeddings = self.roberta.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        if not embeddings.requires_grad:
            embeddings = embeddings.detach().requires_grad_(True)
        clean = super().forward(None, attention_mask, labels, label_mask, embeddings)
        gradient, = torch.autograd.grad(clean.loss, embeddings, retain_graph=True)
        noise = embedding_perturbation(gradient, attention_mask, self.epsilon).to(embeddings)
        adversarial = super().forward(None, attention_mask, labels, label_mask, embeddings + noise)
        loss = (clean.loss + self.adversarial_weight * adversarial.loss) / (1 + self.adversarial_weight)
        return SequenceClassifierOutput(loss=loss, logits=clean.logits)
