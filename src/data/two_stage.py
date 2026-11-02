"""Validated datasets for screening CV on all rows and final 80/10/10 training."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from src.data.nlbse2023 import NLBSEBundle

PROTOCOLS = {"kfold": "kfold_all_data", "final": "train_val_test_80_10_10"}
FILES = {"kfold": ("all.csv", "folds.json", "label_cols.json"),
         "final": ("train.csv", "val.csv", "test.csv", "label_cols.json")}
EVALUATION_NOTE = ("Exploratory evaluation: stage-1 method selection used all rows, "
                   "including the rows assigned to stage-2 validation/test. "
                   "Stage-2 test is not independent of method selection.")


def file_fingerprint(directory, names):
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((Path(directory) / name).read_bytes())
    return digest.hexdigest()


def population_fingerprint(rows, labels):
    ordered = sorted(rows, key=lambda row: str(row["comment_sentence_id"]))
    return hashlib.sha256(json.dumps({"labels": labels, "rows": ordered},
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_rows(path, labels):
    columns = ["comment_sentence_id", "class", "group_id", "text"] + list(labels)
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != columns:
            raise ValueError(f"{path}: columns must be {columns}")
        rows = list(reader)
    if not rows or any(not row.get(key) for row in rows for key in columns):
        raise ValueError(f"{path}: empty or missing values")
    if any(row[label] not in ("0", "1") for row in rows for label in labels):
        raise ValueError(f"{path}: non-binary labels")
    return rows


@dataclass(frozen=True)
class StageBundle(NLBSEBundle):
    validation_mask: torch.Tensor
    folds: list[dict]
    train_size: int
    validation_indices: list[int]
    test_indices: list[int]
    data_dir: str
    protocol: str
    population_fingerprint: str
    data_fingerprint: str


def load_stage_data(data_dir, language, stage):
    if stage not in PROTOCOLS:
        raise ValueError("stage must be kfold or final")
    directory = Path(data_dir) / language
    labels = json.loads((directory / "label_cols.json").read_text())
    if not isinstance(labels, list) or not labels or not all(isinstance(x, str) and x for x in labels) or len(set(labels)) != len(labels):
        raise ValueError("Expected unique label names")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["protocol"] != PROTOCOLS[stage] or manifest["language"] != language:
        raise ValueError("Dataset manifest does not match the requested stage/language")
    split_names = ("all",) if stage == "kfold" else ("train", "val", "test")
    splits = [read_rows(directory / f"{name}.csv", labels) for name in split_names]
    rows = [row for split in splits for row in split]
    ids = [row["comment_sentence_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sentence IDs within/across splits")
    group_text, text_group = {}, {}
    for row in rows:
        group, text = row["group_id"], row["text"]
        if not text or text != " ".join(text.split()):
            raise ValueError("Expected normalized non-empty text")
        if group_text.setdefault(group, text) != text or text_group.setdefault(text, group) != group:
            raise ValueError("group_id must correspond to exactly one normalized text")
    for i, first in enumerate(splits):
        for second in splits[i+1:]:
            if {r["group_id"] for r in first} & {r["group_id"] for r in second}:
                raise ValueError("Group leakage across train/validation/test")
    fingerprint = population_fingerprint(rows, labels)
    if fingerprint != manifest["population_fingerprint"]:
        raise ValueError("Population fingerprint differs from the data manifest")
    for name, split in zip(split_names, splits):
        if len(split) != manifest["counts"][name]:
            raise ValueError("Split counts differ from the data manifest")
    folds = json.loads((directory / "folds.json").read_text()) if stage == "kfold" else []
    if stage == "kfold":
        if len(folds) < 2 or len(folds) != manifest["fold_count"]:
            raise ValueError("Invalid fold count")
        seen = []
        for number, fold in enumerate(folds):
            tr, va = fold["train_indices"], fold["val_indices"]
            if (fold["fold"] != number or not tr or not va
                    or any(type(i) is not int for i in tr + va)
                    or sorted(tr + va) != list(range(len(rows)))):
                raise ValueError(f"Invalid fold indices/coverage in fold {number}")
            if {rows[i]["group_id"] for i in tr} & {rows[i]["group_id"] for i in va}:
                raise ValueError(f"Group leakage in fold {number}")
            seen.extend(va)
        if Counter(seen) != Counter(range(len(rows))):
            raise ValueError("Each row must appear in validation exactly once")
    targets = torch.tensor([[int(row[label]) for label in labels] for row in rows], dtype=torch.float32)
    train_size = len(splits[0])
    val_indices = list(range(train_size, train_size + len(splits[1]))) if stage == "final" else []
    test_indices = list(range(train_size + len(val_indices), len(rows))) if stage == "final" else []
    masks = [torch.zeros_like(targets, dtype=torch.bool) for _ in range(3)]
    for mask, indices in zip(masks, (range(train_size), val_indices, test_indices)):
        mask[list(indices)] = True
    return StageBundle(texts=tuple(r["text"] for r in rows), languages=(language,) * len(rows),
                       sentence_ids=tuple(ids), class_names=tuple(r["class"] for r in rows),
                       label_names=tuple(labels), labels=targets, train_mask=masks[0],
                       validation_mask=masks[1], test_mask=masks[2], folds=folds,
                       train_size=train_size, validation_indices=val_indices, test_indices=test_indices,
                       data_dir=str(directory), protocol=PROTOCOLS[stage], population_fingerprint=fingerprint,
                       data_fingerprint=file_fingerprint(directory, FILES[stage]))
