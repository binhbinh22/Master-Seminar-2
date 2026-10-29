"""Plot training histories and classification metrics for any experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_training_plots(
    output_dir: str | Path,
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    validation_metric: str | None = "micro_f1",
    cross_validation: Mapping[str, Mapping[str, float | None]] | None = None,
    test_metrics: Mapping[str, Any] | None = None,
    score_metrics: Sequence[str] = (
        "micro_f1", "macro_f1", "micro_average_precision", "macro_average_precision"
    ),
    label_metrics: Sequence[str] = ("f1", "average_precision"),
) -> list[str]:
    """Save available charts and return their paths.

    Each history is a sequence of records with ``epoch`` and ``train_loss``.
    Validation records, when present, live under ``validation``. CV summaries
    map metric names to ``mean`` and ``std``. Test metrics may include a
    ``per_label`` mapping. Missing sections are skipped, so the same function
    can plot experiments without cross-validation or a test split.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(fig, filename: str) -> None:
        path = directory / filename
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    if histories:
        has_validation = validation_metric is not None and any(
            (record.get("validation") or {}).get(validation_metric) is not None
            for history in histories.values() for record in history
        )
        fig, axes = plt.subplots(1, 2 if has_validation else 1,
                                 figsize=(12, 4.5) if has_validation else (7, 4.5))
        loss_ax = axes[0] if has_validation else axes
        validation_ax = axes[1] if has_validation else None
        for name, history in histories.items():
            if not history:
                continue
            loss_ax.plot([record["epoch"] for record in history],
                         [record["train_loss"] for record in history],
                         marker="o", label=name)
            if validation_ax is not None:
                points = [(record["epoch"], (record.get("validation") or {}).get(validation_metric))
                          for record in history]
                points = [(epoch, value) for epoch, value in points if value is not None]
                if points:
                    validation_ax.plot([epoch for epoch, _ in points],
                                       [value for _, value in points], marker="o", label=name)
        loss_ax.set(title="Training loss", xlabel="Epoch", ylabel="Loss")
        if validation_ax is not None:
            validation_ax.set(title=f"Validation {validation_metric.replace('_', ' ')}",
                              xlabel="Epoch", ylabel="Score")
        for axis in (loss_ax, validation_ax):
            if axis is not None:
                axis.grid(alpha=0.25)
                if axis.lines:
                    axis.legend(fontsize=8)
        save(fig, "training_curves.png")

    if cross_validation:
        keys = [key for key in score_metrics
                if key in cross_validation and cross_validation[key].get("mean") is not None]
        if keys:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(range(len(keys)), [cross_validation[key]["mean"] for key in keys],
                   yerr=[cross_validation[key].get("std") or 0 for key in keys], capsize=5)
            ax.set_xticks(range(len(keys)), [key.replace("_", " ") for key in keys],
                          rotation=20, ha="right")
            ax.set(title="Cross-validation: mean ± sample SD", ylabel="Score")
            ax.grid(axis="y", alpha=0.25)
            save(fig, "cross_validation.png")

    if test_metrics:
        keys = [key for key in score_metrics
                if key in test_metrics and test_metrics[key] is not None]
        if keys:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(range(len(keys)), [test_metrics[key] for key in keys])
            ax.set_xticks(range(len(keys)), [key.replace("_", " ") for key in keys],
                          rotation=20, ha="right")
            ax.set(title="Test metrics", ylabel="Score")
            ax.grid(axis="y", alpha=0.25)
            save(fig, "test_summary.png")

        per_label = test_metrics.get("per_label")
        if per_label:
            names = list(per_label)
            metrics = [key for key in label_metrics
                       if any(per_label[name].get(key) is not None for name in names)]
            if metrics:
                fig, ax = plt.subplots(figsize=(11, max(6, len(names) * 0.4)))
                height = 0.8 / len(metrics)
                for index, key in enumerate(metrics):
                    offset = (index - (len(metrics) - 1) / 2) * height
                    values = [per_label[name].get(key) for name in names]
                    ax.barh([i + offset for i in range(len(names))],
                            [value if value is not None else float("nan") for value in values],
                            height=height, label=key.replace("_", " "))
                ax.set_yticks(range(len(names)), names)
                ax.invert_yaxis()
                ax.set(title="Test performance by label", xlabel="Score")
                ax.grid(axis="x", alpha=0.25)
                ax.legend()
                save(fig, "test_per_label.png")
    return paths
