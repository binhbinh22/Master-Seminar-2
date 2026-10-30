"""Per-label threshold selection using training OOF predictions only."""

import torch

from src.training.progress import progress


@torch.no_grad()
def predict_probabilities(model, loader, device, *, show_progress=True):
    model.eval()
    scores = []
    for batch in progress(loader, enabled=show_progress, desc="OOF predict", unit="batch"):
        logits = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device)).logits
        scores.append(torch.sigmoid(logits).float().cpu())
    return torch.cat(scores)


def tune_thresholds(scores, labels, grid):
    if scores.shape != labels.shape or scores.ndim != 2:
        raise ValueError("Scores and labels must be matching matrices")
    if not grid or any(not 0 < value < 1 for value in grid):
        raise ValueError("Threshold grid must contain values in (0, 1)")
    # Stable tie-break: prefer a threshold closer to the baseline 0.5.
    candidates = sorted(set(grid), key=lambda value: (abs(value - 0.5), value))
    selected = []
    for column in range(labels.shape[1]):
        actual = labels[:, column].bool()
        if not actual.any() or actual.all():
            selected.append(0.5)
            continue
        best_score, best_threshold = -1.0, 0.5
        for threshold in candidates:
            predicted = scores[:, column] >= threshold
            tp = (predicted & actual).sum().item()
            denominator = predicted.sum().item() + actual.sum().item()
            f1 = 2 * tp / max(denominator, 1)
            if f1 > best_score:
                best_score, best_threshold = f1, threshold
        selected.append(best_threshold)
    return torch.tensor(selected, dtype=torch.float32)
