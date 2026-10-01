"""Data loading for the NLBSE'23 code comment classification dataset (Python subset).

The raw CSV (`data/python.csv`) is stored in "exploded" form: each comment_sentence_id
appears once per category, with `instance_type` (0/1) as the true binary indicator for
that category. `category` alone is NOT the label -- it only names which category the row
represents. This module pivots the exploded rows back into one row per comment sentence
with a multi-hot label vector, which is the correct multi-label classification setup.

It also derives a `group_id` from normalized text (lowercased, whitespace-collapsed).
112 comment sentences in this dataset are near-duplicates of another sentence elsewhere
in the file (same wording copied across different classes/docstrings) -- without
grouping, a plain stratified k-fold can place the same sentence in both the train and
test side of a fold, which leaks information and inflates F1. `group_id` lets the
splitter in `splitting.py` keep every duplicate on the same side of every fold.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[3] / 'data' / 'python.csv'


def _normalize_text(text: str) -> str:
    return ' '.join(text.lower().split())


def load_multilabel(path=DEFAULT_PATH):
    df = pd.read_csv(path, dtype=str, encoding='utf-8', on_bad_lines='skip')
    df['instance_type'] = df['instance_type'].astype(int)

    categories = sorted(df['category'].unique())

    pivot = df.pivot_table(
        index=['comment_sentence_id', 'class', 'comment_sentence'],
        columns='category',
        values='instance_type',
        aggfunc='max',
        fill_value=0,
    ).reset_index()

    pivot = pivot.sort_values('comment_sentence_id').reset_index(drop=True)
    pivot['group_id'] = pivot['comment_sentence'].fillna('').astype(str).map(_normalize_text)

    texts = pivot['comment_sentence'].fillna('').astype(str).values
    Y = pivot[categories].values.astype(int)
    groups = pivot['group_id'].values

    return texts, Y, categories, groups, pivot


if __name__ == '__main__':
    texts, Y, categories, groups, pivot = load_multilabel()
    n_dup_groups = pivot['group_id'].duplicated(keep=False).sum()
    print('n samples:', len(texts))
    print('categories:', categories)
    print('label cardinality (avg labels/sample):', Y.sum(axis=1).mean())
    print('label distribution:', dict(zip(categories, Y.sum(axis=0).tolist())))
    print('unique groups:', len(set(groups)), f'({n_dup_groups} rows share a group with a duplicate)')
