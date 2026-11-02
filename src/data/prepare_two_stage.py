"""Create grouped all-data CV and 80/10/10 datasets from the raw long CSVs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from src.data.two_stage import PROTOCOLS, EVALUATION_NOTE, load_stage_data, population_fingerprint

LANGUAGES = ("python", "java", "pharo")
ROOT = Path(__file__).resolve().parents[2]


def raw_to_rows(path, labels):
    records = {}
    required = {"comment_sentence_id", "class", "comment_sentence", "category", "instance_type", "partition"}
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing raw columns in {path}")
        for source in reader:
            sid, category = source["comment_sentence_id"], source["category"]
            if (not sid or not source["class"].strip() or not source["comment_sentence"].strip()
                    or category not in labels or source["instance_type"] not in ("0", "1")
                    or source["partition"] not in ("0", "1")):
                raise ValueError(f"Invalid raw row in {path}")
            record = records.setdefault(sid, {"comment_sentence_id": sid, "class": source["class"],
                                              "text": source["comment_sentence"]})
            if record["class"] != source["class"] or record["text"] != source["comment_sentence"] or category in record:
                raise ValueError(f"Inconsistent metadata or duplicate ID/category: {sid}")
            record[category] = source["instance_type"]
    if not records:
        raise ValueError("Raw data is empty")
    groups = {}
    rows = []
    for sid in sorted(records, key=lambda x: (0, int(x)) if x.isdigit() else (1, x)):
        record = records[sid]
        if not all(label in record for label in labels):
            raise ValueError(f"Incomplete targets for {sid}")
        text = " ".join(record["text"].split())
        group_id = groups.setdefault(text, str(len(groups)))
        rows.append({"comment_sentence_id": sid, "class": record["class"], "group_id": group_id,
                     "text": text, **{label: record[label] for label in labels}})
    return rows


def grouped_parts(rows, labels, count, seed):
    """Greedy group assignment balancing positive-label counts and sample sizes."""
    groups = sorted({row["group_id"] for row in rows})
    if len(groups) < count:
        raise ValueError(f"Need at least {count} distinct text groups")
    group_index = {group: i for i, group in enumerate(groups)}
    inverse = np.array([group_index[row["group_id"]] for row in rows])
    features = np.zeros((len(groups), len(labels) + 1))
    np.add.at(features, inverse, np.array([[int(row[label]) for label in labels] + [1] for row in rows]))
    target = features.sum(axis=0) / count
    scale = np.maximum(target, 1.)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    order = order[np.argsort(-features[order, -1], kind="stable")]
    totals = np.zeros((count, len(labels) + 1))
    assignment = np.empty(len(groups), dtype=int)
    for position, group in enumerate(order):
        cost = (((totals + features[group] - target) / scale)**2 - ((totals-target) / scale)**2).sum(axis=1)
        # Seed each part with one group so rare-label fixtures cannot create empty splits.
        if position < count:
            available = np.flatnonzero(totals[:, -1] == 0)
        else:
            available = np.arange(count)
        candidates = available[cost[available] <= cost[available].min() + 1e-12]
        part = int(rng.choice(candidates))
        assignment[group] = part
        totals[part] += features[group]
    part_by_row = assignment[inverse]
    return [np.flatnonzero(part_by_row == part).tolist() for part in range(count)]


def write_csv(path, rows, labels):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["comment_sentence_id", "class", "group_id", "text"] + labels)
        writer.writeheader()
        writer.writerows(rows)


def prepare(raw_dir, output_dir, seed=42, folds=5, force=False, config_path=None):
    if type(seed) is not int or not 0 <= seed < 2**32 or type(folds) is not int or folds < 2:
        raise ValueError("Invalid seed or fold count")
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    if not (raw_dir / "python.csv").exists() and (raw_dir / "nlbse2023").is_dir():
        raw_dir = raw_dir / "nlbse2023"
    config = yaml.safe_load(Path(config_path or ROOT / "configs/datasets/nlbse2023_code_comments.yaml").read_text())
    planned = []
    for language in LANGUAGES:
        path = raw_dir / f"{language}.csv"
        labels = sorted(config["languages"][language]["categories"])
        rows = raw_to_rows(path, labels)
        cv_parts = grouped_parts(rows, labels, folds, seed)
        full_parts = grouped_parts(rows, labels, 10, seed)
        all_indices = set(range(len(rows)))
        cv_folds = [{"fold": k, "train_indices": sorted(all_indices - set(part)), "val_indices": part}
                    for k, part in enumerate(cv_parts)]
        train_indices = sorted(i for part in full_parts[:8] for i in part)
        selections = {"kfold": {"all": rows}, "final": {
            "train": [rows[i] for i in train_indices],
            "val": [rows[i] for i in full_parts[8]], "test": [rows[i] for i in full_parts[9]]}}
        fingerprint = population_fingerprint(rows, labels)
        for stage, splits in selections.items():
            folder = output_dir / ("k_fold_data" if stage == "kfold" else "full_training_data") / language
            if folder.exists() and any(folder.iterdir()) and not force:
                raise ValueError(f"Refusing to overwrite {folder}; use --force to regenerate")
            manifest = {"protocol": PROTOCOLS[stage], "language": language, "seed": seed,
                        "population_fingerprint": fingerprint, "source_raw": str(path.resolve()),
                        "source_raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "counts": {name: len(part) for name, part in splits.items()},
                        "label_positive_counts": {name: {label: sum(int(row[label]) for row in part) for label in labels}
                                                  for name, part in splits.items()},
                        "grouping": "exact_normalized_text", "evaluation_note": EVALUATION_NOTE,
                        "fold_count": folds if stage == "kfold" else 0,
                        "target_ratios": None if stage == "kfold" else {"train": .8, "val": .1, "test": .1}}
            planned.append((folder, stage, language, labels, splits, manifest, cv_folds))
    for folder, stage, language, labels, splits, manifest, cv_folds in planned:
        folder.mkdir(parents=True, exist_ok=True)
        for name, part in splits.items():
            write_csv(folder / f"{name}.csv", part, labels)
        (folder / "label_cols.json").write_text(json.dumps(labels, indent=2) + "\n")
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        if stage == "kfold":
            (folder / "folds.json").write_text(json.dumps(cv_folds) + "\n")
        load_stage_data(folder.parent, language, stage)
    return {language: {stage: manifest["counts"] for _, stage, lang, _, _, manifest, _ in planned if lang == language}
            for language in LANGUAGES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw_dir, args.output_dir, args.seed, args.folds, args.force), indent=2))


if __name__ == "__main__":
    main()
