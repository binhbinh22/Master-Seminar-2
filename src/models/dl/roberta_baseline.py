"""Exp 1: the main RoBERTa baseline for 19-label classification."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput


class RobertaBaseline(nn.Module):
    """`roberta-base` CLS representation followed by a 19-output linear head."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 19,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        self.num_labels = num_labels
        self.threshold = threshold
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        if hidden_size != 768:
            raise ValueError(
                f"Exp 1 requires a 768-dimensional CLS vector, got {hidden_size}"
            )
        self.classifier = nn.Linear(768, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        label_mask: Optional[Tensor] = None,
    ) -> SequenceClassifierOutput:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)

        loss = None
        if labels is not None:
            element_loss = self.loss_fn(
                logits,
                labels.to(device=logits.device, dtype=logits.dtype),
            )
            if label_mask is None:
                loss = element_loss.mean()
            else:
                if label_mask.shape != logits.shape:
                    raise ValueError("label_mask must have the same shape as logits")
                selected_loss = element_loss[
                    label_mask.to(device=logits.device, dtype=torch.bool)
                ]
                if selected_loss.numel() == 0:
                    raise ValueError("label_mask must select at least one target")
                loss = selected_loss.mean()

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def predict(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Return a binary label matrix using Exp 1's fixed threshold."""
        logits = self(input_ids=input_ids, attention_mask=attention_mask).logits
        return (torch.sigmoid(logits) >= self.threshold).to(dtype=torch.int64)
