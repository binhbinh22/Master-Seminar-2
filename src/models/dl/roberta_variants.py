"""Focal BCE and label-aware token attention, optionally with label contrast."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

from src.models.dl.roberta_baseline import RobertaBaseline


class BinaryFocalLoss(nn.Module):
    """Unreduced alpha-balanced focal loss; alpha=None disables alpha weighting."""

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        if gamma < 0 or (alpha is not None and not 0 < alpha < 1):
            raise ValueError("Focal loss requires gamma >= 0 and 0 < alpha < 1")
        self.gamma, self.alpha = gamma, alpha
        self.register_buffer("pos_weight", None)

    def forward(self, logits, labels):
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss = (1 - torch.exp(-bce)).pow(self.gamma) * bce
        if self.alpha is not None:
            loss = loss * (self.alpha * labels + (1 - self.alpha) * (1 - labels))
        return loss


class RobertaFocal(RobertaBaseline):
    def __init__(self, gamma=2.0, alpha=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = BinaryFocalLoss(gamma, alpha)


class RobertaLabelAttention(RobertaBaseline):
    """Learned label queries attend over non-padding tokens.

    Optional contrastive term aligns projected CLS with all positive label
    queries using a uniform multi-positive target and cosine/temperature logits.
    """

    def __init__(self, contrastive_weight=0.0, temperature=0.1, **kwargs):
        super().__init__(**kwargs)
        if contrastive_weight < 0 or temperature <= 0:
            raise ValueError("Require contrastive_weight >= 0 and temperature > 0")
        del self.classifier
        self.label_queries = nn.Parameter(torch.empty(self.num_labels, 768))
        nn.init.normal_(self.label_queries, std=0.02)
        self.attention_projection = nn.Linear(768, 768)
        self.label_classifier = nn.Parameter(torch.empty(self.num_labels, 768))
        nn.init.normal_(self.label_classifier, std=0.02)
        self.label_bias = nn.Parameter(torch.zeros(self.num_labels))
        self.contrastive_weight, self.temperature = contrastive_weight, temperature
        # Only E6 needs the CLS projection; avoid unused random parameters in E5.
        if contrastive_weight:
            self.sentence_projection = nn.Linear(768, 768)

    def refine_contexts(self, contexts):
        return contexts

    def forward(self, input_ids, attention_mask=None, labels=None, label_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state
        scores = torch.einsum("bsh,lh->bls", torch.tanh(self.attention_projection(hidden)),
                              self.label_queries) / math.sqrt(hidden.shape[-1])
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask[:, None, :].bool(), torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        contexts = torch.einsum("bls,bsh->blh", weights, hidden)
        contexts = self.refine_contexts(contexts)
        logits = (contexts * self.label_classifier).sum(-1) + self.label_bias
        loss = None
        if labels is not None:
            labels = labels.to(logits)
            mask = torch.ones_like(labels, dtype=torch.bool) if label_mask is None else label_mask.bool()
            if mask.shape != labels.shape or not mask.any():
                raise ValueError("label_mask must match labels and select targets")
            loss = self.loss_fn(logits, labels)[mask].mean()
            if self.contrastive_weight:
                sentence = F.normalize(self.sentence_projection(hidden[:, 0]), dim=-1)
                queries = F.normalize(self.label_queries, dim=-1)
                similarities = sentence @ queries.T / self.temperature
                positives = labels * mask
                valid = positives.sum(-1) > 0
                if valid.any():
                    log_probs = F.log_softmax(similarities[valid].masked_fill(~mask[valid],
                                              torch.finfo(similarities.dtype).min), dim=-1)
                    targets = positives[valid] / positives[valid].sum(-1, keepdim=True)
                    contrast = -(targets * log_probs).sum(-1).mean()
                    loss = loss + self.contrastive_weight * contrast
        return SequenceClassifierOutput(loss=loss, logits=logits,
                                        hidden_states=outputs.hidden_states, attentions=outputs.attentions)
