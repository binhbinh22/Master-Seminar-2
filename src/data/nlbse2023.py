"""Download and load the NLBSE'23 code-comment classification dataset.

The official CSVs contain one row per (sentence, category). This module pivots
them into 6,738 sentence records with 19 language-scoped targets. The official
partition is category-specific, so train/test membership is represented by a
mask for every target instead of a single split value for every sentence.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
from urllib.request import urlopen

import torch
import yaml
from torch import Tensor
from torch.utils.data import Dataset


Split = Literal["train", "test"]


@dataclass(frozen=True)
class NLBSEBundle:
    """Sentence-level data and category-specific official split masks."""

    texts: tuple[str, ...]
    languages: tuple[str, ...]
    sentence_ids: tuple[str, ...]
    class_names: tuple[str, ...]
    label_names: tuple[str, ...]
    labels: Tensor
    train_mask: Tensor
    test_mask: Tensor

    def __len__(self) -> int:
        return len(self.texts)


def read_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open(encoding="utf-8") as file:
        if Path(config_path).suffix in (".yaml", ".yml"):
            return yaml.safe_load(file)
        return json.load(file)


def download_dataset(config: Mapping[str, Any], force: bool = False) -> list[Path]:
    """Download the three official CSV files declared in the dataset config."""
    data_dir = Path(config["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for language, language_config in config["languages"].items():
        destination = data_dir / language_config["file"]
        if destination.exists() and not force:
            downloaded.append(destination)
            continue

        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urlopen(language_config["url"]) as response:
                temporary.write_bytes(response.read())
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Could not download the {language} dataset") from None
        downloaded.append(destination)

    return downloaded


def _label_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    template = config["label_namespace"]
    return tuple(
        template.format(language=language, category=category)
        for language, language_config in config["languages"].items()
        for category in language_config["categories"]
    )


def load_dataset(config: Mapping[str, Any], validate_size: bool = True) -> NLBSEBundle:
    """Load and pivot the official CSV files into sentence-level tensors."""
    label_names = _label_names(config)
    if len(label_names) != config["num_labels"] or len(set(label_names)) != len(label_names):
        raise ValueError("Dataset config must define exactly 19 unique namespaced labels")

    label_to_index = {name: index for index, name in enumerate(label_names)}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    data_dir = Path(config["data_dir"])
    namespace = config["label_namespace"]
    train_value = int(config["split"]["train_value"])
    test_value = int(config["split"]["test_value"])
    if train_value == test_value:
        raise ValueError("Train and test partition values must be different")

    for language, language_config in config["languages"].items():
        csv_path = data_dir / language_config["file"]
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing {csv_path}. Run this module with --download first."
            )

        expected_categories = set(language_config["categories"])
        observed_categories: set[str] = set()
        with csv_path.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                category = row[config["category_column"]]
                observed_categories.add(category)
                if category not in expected_categories:
                    raise ValueError(f"Unexpected category {category!r} in {csv_path}")

                sentence_id = row["comment_sentence_id"]
                key = (language, sentence_id)
                record = records.setdefault(
                    key,
                    {
                        "text": row[config["text_column"]],
                        "class_name": row["class"],
                        "targets": {},
                    },
                )
                if record["text"] != row[config["text_column"]]:
                    raise ValueError(f"Sentence {key} has inconsistent text")
                if category in record["targets"]:
                    raise ValueError(f"Duplicate sentence/category pair: {key}, {category}")

                target = int(row[config["target_column"]])
                partition = int(row[config["partition_column"]])
                if target not in (0, 1) or partition not in (train_value, test_value):
                    raise ValueError(f"Invalid target or partition for sentence {key}")
                record["targets"][category] = (target, partition)

        if observed_categories != expected_categories:
            missing = sorted(expected_categories - observed_categories)
            raise ValueError(f"Missing categories in {csv_path}: {missing}")

    size = len(records)
    expected_size = int(config["expected_samples"])
    if validate_size and size != expected_size:
        raise ValueError(f"Expected {expected_size} sentences, found {size}")

    labels = torch.zeros((size, len(label_names)), dtype=torch.float32)
    train_mask = torch.zeros_like(labels, dtype=torch.bool)
    test_mask = torch.zeros_like(labels, dtype=torch.bool)
    texts: list[str] = []
    languages: list[str] = []
    sentence_ids: list[str] = []
    class_names: list[str] = []

    for row_index, ((language, sentence_id), record) in enumerate(records.items()):
        texts.append(record["text"])
        languages.append(language)
        sentence_ids.append(sentence_id)
        class_names.append(record["class_name"])
        expected = config["languages"][language]["categories"]
        if set(record["targets"]) != set(expected):
            raise ValueError(f"Sentence {(language, sentence_id)} has incomplete targets")

        for category, (target, partition) in record["targets"].items():
            name = namespace.format(language=language, category=category)
            column = label_to_index[name]
            labels[row_index, column] = target
            selected_mask = train_mask if partition == train_value else test_mask
            selected_mask[row_index, column] = True

    if torch.any(train_mask & test_mask):
        raise ValueError("A sentence/category pair cannot be in both partitions")

    return NLBSEBundle(
        texts=tuple(texts),
        languages=tuple(languages),
        sentence_ids=tuple(sentence_ids),
        class_names=tuple(class_names),
        label_names=label_names,
        labels=labels,
        train_mask=train_mask,
        test_mask=test_mask,
    )


class TokenizedNLBSEDataset(Dataset[dict[str, Tensor]]):
    """Tokenized view that includes the mask needed for the selected split."""

    def __init__(
        self,
        bundle: NLBSEBundle,
        tokenizer: Any,
        split: Split,
        max_length: int = 128,
        indices: Sequence[int] | None = None,
    ) -> None:
        self.bundle = bundle
        self.tokenizer = tokenizer
        self.label_mask = bundle.train_mask if split == "train" else bundle.test_mask
        eligible = self.label_mask.any(dim=1)
        self.indices = (
            eligible.nonzero(as_tuple=True)[0].tolist()
            if indices is None
            else list(indices)
        )
        if any(not bool(eligible[index]) for index in self.indices):
            raise ValueError("Selected indices must contain targets for the split")
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        index = self.indices[item]
        encoded = self.tokenizer(
            self.bundle.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": self.bundle.labels[index],
            "label_mask": self.label_mask[index],
        }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/datasets/nlbse2023_code_comments.yaml",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = read_config(args.config)
    if args.download:
        download_dataset(config, force=args.force)
    bundle = load_dataset(config)
    print(f"Loaded {len(bundle):,} sentences and {len(bundle.label_names)} labels")


if __name__ == "__main__":
    main()
