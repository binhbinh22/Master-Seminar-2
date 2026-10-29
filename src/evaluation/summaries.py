"""Summarize numeric metrics across repeated runs or folds."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence


def summarize_metric_runs(
    runs: Sequence[Mapping[str, float | None]], keys: Sequence[str]
) -> dict[str, dict[str, float | int | None]]:
    """Return mean, sample SD and count of defined values for each metric."""
    summary = {}
    for key in keys:
        values = [run[key] for run in runs if run.get(key) is not None]
        summary[key] = {
            "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "defined_folds": len(values),
        }
    return summary
