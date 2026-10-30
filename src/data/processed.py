"""Load the shared sentence-level splits without changing their row order."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from src.data.nlbse2023 import NLBSEBundle


@dataclass(frozen=True)
class ProcessedBundle(NLBSEBundle):
    folds: list[dict]
    train_size: int
    data_dir: str


def load_processed(data_dir: str | Path, language: str) -> ProcessedBundle:
    """Validate splits, label order, groups and CV coverage before training."""
    directory = Path(data_dir) / language
    labels = json.loads((directory / "label_cols.json").read_text())
    if not labels or len(set(labels)) != len(labels) or not all(isinstance(x, str) for x in labels):
        raise ValueError("label_cols.json must contain unique label names")
    splits = []
    for split in ("train", "test"):
        with (directory / f"{split}.csv").open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            expected = ["comment_sentence_id", "class", "group_id", "text"] + labels
            if reader.fieldnames != expected:
                raise ValueError(f"{split}.csv columns must be {expected}")
            rows = list(reader)
        if not rows or any(not r[k] for r in rows for k in expected):
            raise ValueError(f"{split}.csv contains empty data")
        if any(r[label] not in ("0", "1") for r in rows for label in labels):
            raise ValueError(f"{split}.csv has non-binary labels")
        splits.append(rows)
    train, test = splits
    rows = train + test
    ids = [r["comment_sentence_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sentence IDs within or across splits")
    group_text, text_group = {}, {}
    for row in rows:
        group, text = row["group_id"], row["text"]
        if text != " ".join(text.split()):
            raise ValueError("Expected normalized, non-empty text")
        if group_text.setdefault(group, text) != text or text_group.setdefault(text, group) != group:
            raise ValueError("group_id must correspond to exactly one normalized text")
    for key in ("group_id", "text"):
        if {r[key] for r in train} & {r[key] for r in test}:
            raise ValueError(f"Train/test overlap in {key}")
    folds = json.loads((directory / "folds.json").read_text())
    if len(folds) < 2:
        raise ValueError("At least two CV folds are required")
    seen = []
    for number, fold in enumerate(folds):
        tr, va = fold["train_indices"], fold["val_indices"]
        indices = tr + va
        if (fold["fold"] != number or not tr or not va
                or any(type(i) is not int for i in indices)
                or sorted(indices) != list(range(len(train)))):
            raise ValueError(f"Invalid indices or coverage in fold {number}")
        if {train[i]["group_id"] for i in tr} & {train[i]["group_id"] for i in va}:
            raise ValueError(f"Group leakage in fold {number}")
        seen.extend(va)
    if Counter(seen) != Counter(range(len(train))):
        raise ValueError("Each train row must be validation exactly once")
    targets = torch.tensor([[int(r[label]) for label in labels] for r in rows], dtype=torch.float32)
    train_mask = torch.zeros_like(targets, dtype=torch.bool)
    train_mask[:len(train)] = True
    return ProcessedBundle(
        texts=tuple(r["text"] for r in rows), languages=(language,) * len(rows),
        sentence_ids=tuple(ids), class_names=tuple(r["class"] for r in rows),
        label_names=tuple(labels), labels=targets, train_mask=train_mask,
        test_mask=~train_mask, folds=folds, train_size=len(train), data_dir=str(directory),
    )
