"""Export the corrected, pivoted multi-label dataset to a single CSV so other team
members can train their own models (e.g. a different DL architecture) without having
to re-discover the exploded-row / instance_type bug in the raw python.csv.

Also assigns a fixed 5-fold column using the same MultilabelStratifiedKFold split used
by ml_pipeline.py, so everyone's results are comparable on identical train/test splits.
"""
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from src.models.ml.data_utils import load_multilabel


def export(path_in='python.csv', path_out='data/python_multilabel_processed.csv', folds=5, seed=42):
    texts, Y, categories, pivot = load_multilabel(path_in)

    out = pivot[['comment_sentence_id', 'class', 'comment_sentence'] + categories].copy()

    mskf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out['fold'] = -1
    for fold_idx, (_, val_idx) in enumerate(mskf.split(texts, Y)):
        out.iloc[val_idx, out.columns.get_loc('fold')] = fold_idx

    import os
    os.makedirs('data', exist_ok=True)
    out.to_csv(path_out, index=False)

    print(f'Saved {len(out)} rows x {len(out.columns)} cols to {path_out}')
    print('Columns:', list(out.columns))
    print('Fold sizes:', out['fold'].value_counts().sort_index().to_dict())
    return out


if __name__ == '__main__':
    export()
