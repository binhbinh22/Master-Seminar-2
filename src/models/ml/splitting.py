"""Group-aware multilabel k-fold splitting.

`iterative-stratification`'s MultilabelStratifiedKFold has no notion of groups, so used
directly on this dataset it can put two rows with identical (duplicate) text on opposite
sides of a fold -- the model then sees a test sentence verbatim during training. See
`data.py` module docstring for why that matters (112 duplicate-text groups, ~14% of a
naive test split affected).

This module stratifies at the GROUP level: one representative label vector per group
(the first occurrence), split into folds with MultilabelStratifiedKFold, then every row
sharing that group_id is assigned to its group's fold. This keeps every duplicate on one
side of every train/test boundary at the cost of slightly coarser label stratification
for the ~14% of groups whose duplicate occurrences carry different label combinations.
"""
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def group_aware_multilabel_kfold(texts, Y, groups, n_splits=5, seed=42):
    unique_groups = np.unique(groups)
    group_to_row = {g: np.where(groups == g)[0][0] for g in unique_groups}
    representative_rows = np.array([group_to_row[g] for g in unique_groups])
    Y_group = Y[representative_rows]

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_dummy = np.zeros((len(unique_groups), 1))

    for _, val_group_idx in mskf.split(X_dummy, Y_group):
        val_groups = set(unique_groups[val_group_idx])
        val_mask = np.array([g in val_groups for g in groups])
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        yield train_idx, val_idx


if __name__ == '__main__':
    from src.models.ml.data import load_multilabel

    texts, Y, categories, groups, pivot = load_multilabel()
    for fold, (train_idx, val_idx) in enumerate(group_aware_multilabel_kfold(texts, Y, groups)):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = train_groups & val_groups
        print(f'fold {fold}: train={len(train_idx)} val={len(val_idx)} group_overlap={len(overlap)}')
