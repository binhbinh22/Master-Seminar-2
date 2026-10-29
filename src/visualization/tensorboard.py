"""TensorBoard logging shared by experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.utils.tensorboard import SummaryWriter


def log_metrics(writer: SummaryWriter, prefix: str, metrics: Mapping[str, Any], step: int) -> None:
    """Log defined scalar metrics, including nested per-label scores."""
    for name, value in metrics.items():
        if name == "per_label":
            for label, values in value.items():
                for metric, score in values.items():
                    if score is not None:
                        writer.add_scalar(f"{prefix}/per_label/{label}/{metric}", score, step)
        elif value is not None:
            writer.add_scalar(f"{prefix}/{name}", value, step)
