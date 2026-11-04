"""Multi-label metrics with the exact same field names as the DL side's
`src.evaluation.multilabel.masked_multilabel_metrics`, so a `metrics.json` from this
module and one from the RoBERTa/CodeBERT pipeline are diffable key-for-key. ML never
needs the label mask DL uses for cross-language joint training -- every row here has
every label observed.
"""
import numpy as np
from sklearn.metrics import average_precision_score


def compute_metrics(y_true, y_score, y_pred, label_names, per_label=True):
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    y_score = np.asarray(y_score, dtype=float)

    tp = (y_pred & y_true).sum(axis=0)
    fp = (y_pred & ~y_true).sum(axis=0)
    fn = (~y_pred & y_true).sum(axis=0)
    tn = (~y_pred & ~y_true).sum(axis=0)
    support = tp + fn
    predicted = tp + fp
    precision = tp / np.maximum(predicted, 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)

    total_tp, total_fp, total_fn, total_tn = tp.sum(), fp.sum(), fn.sum(), tn.sum()
    total = total_tp + total_fp + total_fn + total_tn

    aps = [
        float(average_precision_score(y_true[:, j], y_score[:, j])) if y_true[:, j].sum() > 0 else None
        for j in range(len(label_names))
    ]
    defined_aps = [ap for ap in aps if ap is not None]
    micro_ap = float(average_precision_score(y_true.ravel(), y_score.ravel())) if y_true.any() else None

    result = {
        "micro_precision": float(total_tp / max(total_tp + total_fp, 1)),
        "micro_recall": float(total_tp / max(total_tp + total_fn, 1)),
        "micro_f1": float(2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "micro_average_precision": micro_ap,
        "macro_average_precision": float(np.mean(defined_aps)) if defined_aps else None,
        "hamming_loss": float((fp.sum() + fn.sum()) / max(total, 1)),
        "evaluated_targets": int(total),
        "positive_targets": int(support.sum()),
        "predicted_positive_targets": int(predicted.sum()),
        "true_positive_targets": int(tp.sum()),
        "false_positive_targets": int(fp.sum()),
        "false_negative_targets": int(fn.sum()),
    }
    if per_label:
        result["per_label"] = {
            label_names[j]: {
                "precision": float(precision[j]),
                "recall": float(recall[j]),
                "f1": float(f1[j]),
                "average_precision": aps[j],
                "support": int(support[j]),
                "predicted_positive": int(predicted[j]),
                "true_positive": int(tp[j]),
                "evaluated_targets": int(tp[j] + fp[j] + fn[j] + tn[j]),
            }
            for j in range(len(label_names))
        }
    return result
