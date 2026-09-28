"""Per-label precision/recall/F1 for the best classical ML model (OVR Logistic Regression),
averaged across the same MultilabelStratifiedKFold splits used in ml_pipeline.py.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from data_utils import load_multilabel
from ml_pipeline import build_vectorizer


def run(data_path='python.csv', folds=5):
    texts, Y, categories, _ = load_multilabel(data_path)
    mskf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    n_labels = len(categories)
    p_acc = np.zeros((folds, n_labels))
    r_acc = np.zeros((folds, n_labels))
    f_acc = np.zeros((folds, n_labels))
    support_total = np.zeros(n_labels)

    for i, (train_idx, test_idx) in enumerate(mskf.split(texts, Y)):
        vec = build_vectorizer()
        X_train = vec.fit_transform(texts[train_idx])
        X_test = vec.transform(texts[test_idx])
        y_train, y_test = Y[train_idx], Y[test_idx]

        clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight='balanced', C=5.0))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        p, r, f, sup = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
        p_acc[i] = p
        r_acc[i] = r
        f_acc[i] = f
        support_total += sup

    out = {}
    for j, lbl in enumerate(categories):
        out[lbl] = {
            'precision': float(p_acc[:, j].mean()),
            'recall': float(r_acc[:, j].mean()),
            'f1': float(f_acc[:, j].mean()),
            'support_total': int(support_total[j]),
        }

    with open('results/per_label_metrics.json', 'w') as f:
        json.dump(out, f, indent=2)

    print(f"{'label':<20}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for lbl, m in out.items():
        print(f"{lbl:<20}{m['precision']:>10.4f}{m['recall']:>10.4f}{m['f1']:>10.4f}{m['support_total']:>10d}")

    return out


if __name__ == '__main__':
    run()
