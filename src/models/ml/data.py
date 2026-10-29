"""Data loading for the NLBSE'23 code comment classification dataset.

Reads the team's shared, pre-split artifacts in `data/processed/<language>/` instead of
the raw per-language CSVs. That split (train/test + 5 CV folds) is produced once by
`notebooks/data_preprocessing.ipynb` and is group-aware: every near-duplicate sentence
(copy-pasted docstring text) is kept on one side of every train/test and train/val
boundary, so no model -- classical ML here or the team's RoBERTa baseline -- can leak a
test sentence into its own training data.

Both the ML and RoBERTa experiments MUST read from this same processed split. That's
what makes their held-out test metrics directly comparable: same sentences in train,
same sentences in test, same folds, for both methods.
"""
import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[3] / 'data' / 'processed'


def load_processed(language='python', processed_dir=PROCESSED_DIR):
    lang_dir = Path(processed_dir) / language

    train_df = pd.read_csv(lang_dir / 'train.csv')
    test_df = pd.read_csv(lang_dir / 'test.csv')
    label_names = json.loads((lang_dir / 'label_cols.json').read_text(encoding='utf-8'))
    folds = json.loads((lang_dir / 'folds.json').read_text(encoding='utf-8'))

    return {
        'language': language,
        'label_names': label_names,
        'folds': folds,
        'train_texts': train_df['text'].fillna('').astype(str).to_numpy(),
        'train_labels': train_df[label_names].to_numpy(dtype=int),
        'test_texts': test_df['text'].fillna('').astype(str).to_numpy(),
        'test_labels': test_df[label_names].to_numpy(dtype=int),
        'train_df': train_df,
        'test_df': test_df,
    }


if __name__ == '__main__':
    bundle = load_processed('python')
    print('language:', bundle['language'])
    print('labels:', bundle['label_names'])
    print('train:', bundle['train_texts'].shape, 'test:', bundle['test_texts'].shape)
    print('folds:', len(bundle['folds']))
