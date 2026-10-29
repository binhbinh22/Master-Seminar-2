"""Reusable split construction for cross-validation."""

from __future__ import annotations

import random
from collections.abc import Hashable, Sequence


def make_balanced_folds(
    indices: Sequence[int], strata: Sequence[Hashable], count: int, seed: int
) -> list[list[int]]:
    """Assign each index to one fold, balancing counts within each stratum."""
    if len(indices) != len(strata):
        raise ValueError("indices and strata must have the same length")
    if len(set(indices)) != len(indices):
        raise ValueError("indices must be unique")
    if not 2 <= count <= len(indices):
        raise ValueError(f"folds must be between 2 and {len(indices)}")
    groups: dict[Hashable, list[int]] = {}
    for index, stratum in zip(indices, strata):
        groups.setdefault(stratum, []).append(index)
    folds: list[list[int]] = [[] for _ in range(count)]
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
        for position, index in enumerate(group):
            folds[position % count].append(index)
    if any(not fold for fold in folds):
        raise ValueError("Every fold must contain at least one sample")
    return folds
