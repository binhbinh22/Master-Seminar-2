"""Correct data loading for the NLBSE'23 code comment classification dataset (Python subset).

The raw CSV is stored in "exploded" form: each comment_sentence_id appears once per
category, with `instance_type` (0/1) as the true binary indicator for that category.
`category` alone is NOT the label -- it only names which category the row represents.
This module pivots the exploded rows back into one row per comment sentence with a
multi-hot label vector, which is the correct multi-label classification setup.
"""
import pandas as pd
import numpy as np


def load_multilabel(path='python.csv'):
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

    texts = pivot['comment_sentence'].fillna('').astype(str).values
    Y = pivot[categories].values.astype(int)

    return texts, Y, categories, pivot


if __name__ == '__main__':
    texts, Y, categories, pivot = load_multilabel()
    print('n samples:', len(texts))
    print('categories:', categories)
    print('label cardinality (avg labels/sample):', Y.sum(axis=1).mean())
    print('label distribution:', dict(zip(categories, Y.sum(axis=0).tolist())))
