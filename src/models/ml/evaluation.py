"""Cross-validation + held-out test evaluation for the multi-label classical ML models.

Both use the pre-computed, group-aware split from `data/processed/` (see data.py) --
this module never derives its own train/val/test split.
"""
import numpy as np
from sklearn.metrics import (
    f1_score, jaccard_score, hamming_loss, precision_score, recall_score,
    precision_recall_fscore_support,
)

from src.models.ml.features import build_vectorizer


def _predict(build_clf_fn, X_train, y_train, X_test):
    clf = build_clf_fn()
    clf.fit(X_train, y_train)
    return np.asarray(clf.predict(X_test))


def cross_validate(build_clf_fn, train_texts, train_labels, folds, feature_kwargs=None):
    """5-fold CV on the training split only, using the team's pre-computed fold indices.

    Used for model selection/comparison -- never touches the held-out test set.
    """
    feature_kwargs = feature_kwargs or {}
    metrics = {'micro_f1': [], 'macro_f1': [], 'jaccard': [], 'hamming': [],
               'precision_micro': [], 'recall_micro': []}

    for fold in folds:
        train_idx = np.array(fold['train_indices'])
        val_idx = np.array(fold['val_indices'])

        vec = build_vectorizer(**feature_kwargs)
        X_train = vec.fit_transform(train_texts[train_idx])
        X_val = vec.transform(train_texts[val_idx])
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]

        y_pred = _predict(build_clf_fn, X_train, y_train, X_val)

        metrics['micro_f1'].append(f1_score(y_val, y_pred, average='micro', zero_division=0))
        metrics['macro_f1'].append(f1_score(y_val, y_pred, average='macro', zero_division=0))
        metrics['jaccard'].append(jaccard_score(y_val, y_pred, average='samples', zero_division=0))
        metrics['hamming'].append(hamming_loss(y_val, y_pred))
        metrics['precision_micro'].append(precision_score(y_val, y_pred, average='micro', zero_division=0))
        metrics['recall_micro'].append(recall_score(y_val, y_pred, average='micro', zero_division=0))

    return {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in metrics.items()}


def evaluate_on_test(build_clf_fn, train_texts, train_labels, test_texts, test_labels, label_names,
                      feature_kwargs=None):
    """Fit on the FULL training split, report metrics on the held-out test split.

    This is the one number that should be compared against the RoBERTa baseline's test
    metrics -- both methods are fit on the same train.csv and scored on the same
    test.csv.
    """
    vec = build_vectorizer(**(feature_kwargs or {}))
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    y_pred = _predict(build_clf_fn, X_train, train_labels, X_test)

    overall = {
        'micro_f1': float(f1_score(test_labels, y_pred, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(test_labels, y_pred, average='macro', zero_division=0)),
        'jaccard': float(jaccard_score(test_labels, y_pred, average='samples', zero_division=0)),
        'hamming': float(hamming_loss(test_labels, y_pred)),
        'precision_micro': float(precision_score(test_labels, y_pred, average='micro', zero_division=0)),
        'recall_micro': float(recall_score(test_labels, y_pred, average='micro', zero_division=0)),
    }

    p, r, f, sup = precision_recall_fscore_support(test_labels, y_pred, zero_division=0)
    per_label = {
        label_names[j]: {
            'precision': float(p[j]), 'recall': float(r[j]), 'f1': float(f[j]), 'support': int(sup[j]),
        }
        for j in range(len(label_names))
    }

    return overall, per_label
