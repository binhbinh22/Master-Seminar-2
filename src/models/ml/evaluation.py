"""Cross-validated evaluation for the multi-label classical ML models.

Uses the group-aware splitter in splitting.py (not plain MultilabelStratifiedKFold) so
duplicate-text sentences never appear on both sides of a fold. See data.py and
splitting.py docstrings for why that matters.
"""
import numpy as np
from sklearn.metrics import (
    f1_score, jaccard_score, hamming_loss, precision_score, recall_score,
    precision_recall_fscore_support,
)

from src.models.ml.features import build_vectorizer
from src.models.ml.splitting import group_aware_multilabel_kfold


def evaluate_model(build_clf_fn, texts, Y, groups, folds=5, seed=42):
    metrics = {'micro_f1': [], 'macro_f1': [], 'jaccard': [], 'hamming': [],
               'precision_micro': [], 'recall_micro': []}

    for train_idx, test_idx in group_aware_multilabel_kfold(texts, Y, groups, n_splits=folds, seed=seed):
        vec = build_vectorizer()
        X_train = vec.fit_transform(texts[train_idx])
        X_test = vec.transform(texts[test_idx])
        y_train, y_test = Y[train_idx], Y[test_idx]

        clf = build_clf_fn()
        clf.fit(X_train, y_train)
        y_pred = np.asarray(clf.predict(X_test))

        metrics['micro_f1'].append(f1_score(y_test, y_pred, average='micro', zero_division=0))
        metrics['macro_f1'].append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        metrics['jaccard'].append(jaccard_score(y_test, y_pred, average='samples', zero_division=0))
        metrics['hamming'].append(hamming_loss(y_test, y_pred))
        metrics['precision_micro'].append(precision_score(y_test, y_pred, average='micro', zero_division=0))
        metrics['recall_micro'].append(recall_score(y_test, y_pred, average='micro', zero_division=0))

    return {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in metrics.items()}


def per_label_metrics(build_clf_fn, texts, Y, groups, categories, folds=5, seed=42):
    n_labels = len(categories)
    p_acc = np.zeros((folds, n_labels))
    r_acc = np.zeros((folds, n_labels))
    f_acc = np.zeros((folds, n_labels))
    support_total = np.zeros(n_labels)

    splits = list(group_aware_multilabel_kfold(texts, Y, groups, n_splits=folds, seed=seed))
    for i, (train_idx, test_idx) in enumerate(splits):
        vec = build_vectorizer()
        X_train = vec.fit_transform(texts[train_idx])
        X_test = vec.transform(texts[test_idx])
        y_train, y_test = Y[train_idx], Y[test_idx]

        clf = build_clf_fn()
        clf.fit(X_train, y_train)
        y_pred = np.asarray(clf.predict(X_test))

        p, r, f, sup = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
        p_acc[i], r_acc[i], f_acc[i] = p, r, f
        support_total += sup

    out = {}
    for j, lbl in enumerate(categories):
        out[lbl] = {
            'precision': float(p_acc[:, j].mean()),
            'recall': float(r_acc[:, j].mean()),
            'f1': float(f_acc[:, j].mean()),
            'support_total': int(support_total[j]),
        }
    return out
