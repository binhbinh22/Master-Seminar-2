"""Chuyển dữ liệu NLBSE'23 từ long one-vs-rest sang wide multi-label.

Dùng cho java.csv / pharo.csv (tương tự python_preprocessed.csv của Python).

Output ``data/{language}_preprocessed.csv`` gồm:
    comment_sentence_id, class, comment_sentence, <label cols...>,
    label_count, partition_official

``partition_official`` là partition theo đa số phiếu giữa các category của câu
(0 = train, 1 = test; hòa phiếu -> train). Cột này chỉ để đối chiếu, KHÔNG dùng
để chia train/test (partition gốc nằm trên từng cặp (câu, category)).

Usage:
    python src/data/make_wide.py --language java pharo
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REQUIRED = {"comment_sentence_id", "class", "comment_sentence", "partition", "instance_type", "category"}


def normalize_whitespace(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def make_wide(raw: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"Thiếu cột: {sorted(missing)}")

    labels = sorted(raw["category"].unique())
    y = raw.pivot_table(
        index="comment_sentence_id", columns="category", values="instance_type", aggfunc="max"
    )[labels].astype(int)

    meta = (
        raw.sort_values("comment_sentence_id")
        .drop_duplicates("comment_sentence_id")
        .set_index("comment_sentence_id")[["class", "comment_sentence"]]
    )
    meta["comment_sentence"] = meta["comment_sentence"].map(normalize_whitespace)

    votes = raw.groupby("comment_sentence_id")["partition"].mean()
    wide = meta.join(y)
    wide["label_count"] = y.sum(axis=1)
    wide["partition_official"] = (votes.reindex(wide.index) > 0.5).astype(int)
    return wide.reset_index(), labels


def validate(raw: pd.DataFrame, wide: pd.DataFrame, labels: list[str]) -> None:
    assert len(wide) == raw["comment_sentence_id"].nunique(), "Số dòng không khớp số câu unique"
    assert wide["comment_sentence_id"].is_unique, "comment_sentence_id bị trùng"
    assert wide.isnull().sum().sum() == 0, "Có missing value"
    src_pos = raw.groupby("category")["instance_type"].sum()
    for label in labels:
        assert wide[label].sum() == src_pos[label], f"Tổng positive của {label} không khớp"
    assert (wide["label_count"] == wide[labels].sum(axis=1)).all()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", nargs="+", default=["java", "pharo"], choices=["java", "pharo", "python"])
    args = parser.parse_args()
    for language in args.language:
        raw = pd.read_csv(DATA_DIR / f"{language}.csv")
        wide, labels = make_wide(raw)
        validate(raw, wide, labels)
        out = DATA_DIR / f"{language}_preprocessed.csv"
        wide.to_csv(out, index=False)
        print(f"{language}: {len(wide)} dòng, {len(labels)} nhãn {labels} -> {out}")


if __name__ == "__main__":
    main()
