"""Metrics for masked multi-label classifiers with logits outputs."""

from __future__ import annotations

import statistics
from collections.abc import Sequence

import torch


SUMMARY_METRICS = (
    "micro_precision", "micro_recall", "micro_f1", "macro_precision",
    "macro_recall", "macro_f1", "micro_average_precision",
    "macro_average_precision", "hamming_loss",
)


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    """Average precision with tied scores grouped at the same threshold."""
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    cumulative = labels[order].to(torch.float64).cumsum(0)
    ends = torch.cat((sorted_scores[1:] != sorted_scores[:-1],
                      torch.ones(1, dtype=torch.bool)))
    positions = ends.nonzero(as_tuple=True)[0]
    tp = cumulative[positions]
    previous = torch.cat((tp.new_zeros(1), tp[:-1]))
    return float(((tp - previous) / positives * tp / (positions + 1)).sum())


def masked_multilabel_metrics(
    true_positive: torch.Tensor,
    false_positive: torch.Tensor,
    false_negative: torch.Tensor,
    true_negative: torch.Tensor,
    label_names: Sequence[str],
    per_label: bool,
    average_precisions: list[float | None] | None = None,
    micro_ap: float | None = None,
) -> dict:
    """Summarize confusion counts for targets selected by a label mask."""
    tp, fp, fn, tn = (x.to(torch.float64) for x in
                      (true_positive, false_positive, false_negative, true_negative))
    support = tp + fn
    predicted = tp + fp
    precision = tp / predicted.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
    total = (tp + fp + fn + tn).sum()
    defined_aps = [ap for ap in average_precisions or [] if ap is not None]
    result = {
        "micro_precision": float(total_tp / (total_tp + total_fp).clamp_min(1)),
        "micro_recall": float(total_tp / (total_tp + total_fn).clamp_min(1)),
        "micro_f1": float(2 * total_tp / (2 * total_tp + total_fp + total_fn).clamp_min(1)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "micro_average_precision": micro_ap,
        "macro_average_precision": statistics.mean(defined_aps) if defined_aps else None,
        "hamming_loss": float((fp.sum() + fn.sum()) / total.clamp_min(1)),
        "evaluated_targets": int(total),
        "positive_targets": int(support.sum()),
    }
    if per_label:
        result["per_label"] = {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "average_precision": average_precisions[i] if average_precisions else None,
                "support": int(support[i]),
                "evaluated_targets": int(tp[i] + fp[i] + fn[i] + tn[i]),
            }
            for i, name in enumerate(label_names)
        }
    return result


@torch.no_grad()
def evaluate_masked_multilabel(model, loader, device, threshold, label_names, per_label) -> dict:
    """Evaluate batches with input_ids, attention_mask, labels and label_mask."""
    model.eval()
    tp = torch.zeros(len(label_names), dtype=torch.int64)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)
    tn = torch.zeros_like(tp)
    all_scores, all_labels, all_masks = [], [], []
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        scores = torch.sigmoid(logits).float().cpu()
        predicted = scores >= threshold
        actual = batch["labels"].bool()
        mask = batch["label_mask"].bool()
        all_scores.append(scores)
        all_labels.append(actual)
        all_masks.append(mask)
        tp += (predicted & actual & mask).sum(dim=0)
        fp += (predicted & ~actual & mask).sum(dim=0)
        fn += (~predicted & actual & mask).sum(dim=0)
        tn += (~predicted & ~actual & mask).sum(dim=0)
    if not all_scores:
        raise ValueError("Evaluation split has no sentences")
    scores, labels, masks = map(torch.cat, (all_scores, all_labels, all_masks))
    aps = [average_precision(scores[masks[:, i], i], labels[masks[:, i], i])
           for i in range(len(label_names))]
    micro_ap = average_precision(scores[masks], labels[masks])
    return masked_multilabel_metrics(tp, fp, fn, tn, label_names, per_label, aps, micro_ap)
