import json
import os
import numpy as np
import pandas as pd
import re
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, f1_score

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

re_split = re.compile(r"[;|,\\/]+")


def load_and_process(path):
    df = pd.read_csv(path, dtype=str, encoding='utf-8', on_bad_lines='skip')
    text_col = next((c for c in ['comment_sentence', 'text', 'comment', 'sentence'] if c in df.columns), df.columns[2])
    label_col = next((c for c in ['labels', 'label', 'category', 'topics', 'class'] if c in df.columns), None)
    if label_col is None:
        raise ValueError('No label column found in CSV')
    texts = df[text_col].fillna('').astype(str).values
    raw_labels = df[label_col].fillna('').astype(str).values
    labels_list = []
    for s in raw_labels:
        s = s.strip()
        if s == '':
            labels_list.append([])
        else:
            parts = [p.strip() for p in re_split.split(s) if p.strip()]
            labels_list.append(parts)
    return texts, labels_list


def build_vectorizer():
    word_tfidf = TfidfVectorizer(ngram_range=(1,3), analyzer='word', max_features=25000)
    char_tfidf = TfidfVectorizer(ngram_range=(3,6), analyzer='char', max_features=20000)
    return FeatureUnion([('word', word_tfidf), ('char', char_tfidf)])


def get_probas(clf, X):
    # return n_samples x n_labels probability estimates for positive class
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(X)
        # predict_proba may return list or array; ensure shape
        probs = np.asarray(probs)
    else:
        # fallback to decision_function + sigmoid
        df = clf.decision_function(X)
        probs = 1 / (1 + np.exp(-df))
    return probs


def tune_thresholds_cv(texts, Y, folds=5):
    mskf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    n_labels = Y.shape[1]
    thresholds_per_fold = []

    for train_idx, val_idx in mskf.split(texts, Y):
        X_train_texts, X_val_texts = texts[train_idx], texts[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]

        vec = build_vectorizer()
        X_train = vec.fit_transform(X_train_texts)
        X_val = vec.transform(X_val_texts)

        base = LogisticRegression(solver='saga', max_iter=1000, class_weight='balanced')
        chain = ClassifierChain(base, order='random', random_state=42)
        chain.fit(X_train, y_train)

        probas = get_probas(chain, X_val)
        # ensure shape
        probas = np.asarray(probas)
        if probas.ndim == 1:
            probas = probas.reshape(-1, n_labels)

        # for each label find threshold maximizing F1 on this fold
        best_thresh = []
        for j in range(n_labels):
            best_f1 = -1.0
            best_t = 0.5
            y_true = y_val[:, j]
            p = probas[:, j]
            for t in np.linspace(0.0, 1.0, 101):
                y_pred = (p >= t).astype(int)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            best_thresh.append(best_t)

        thresholds_per_fold.append(best_thresh)

    thresholds_per_fold = np.array(thresholds_per_fold)
    mean_thresholds = thresholds_per_fold.mean(axis=0)
    return mean_thresholds


def final_evaluate(texts, Y, thresholds):
    vec = build_vectorizer()
    X = vec.fit_transform(texts)
    base = LogisticRegression(solver='saga', max_iter=1000, class_weight='balanced')
    chain = ClassifierChain(base, order='random', random_state=42)
    chain.fit(X, Y)
    probas = get_probas(chain, X)
    probas = np.asarray(probas)
    if probas.ndim == 1:
        probas = probas.reshape(-1, Y.shape[1])

    Y_pred = (probas >= thresholds).astype(int)

    per_label = {}
    p, r, f1, sup = precision_recall_fscore_support(Y, Y_pred, zero_division=0)
    for i, lbl in enumerate(m_labels):
        per_label[lbl] = {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f1[i]), 'support': int(sup[i]), 'threshold': float(thresholds[i])}

    micro = f1_score(Y, Y_pred, average='micro', zero_division=0)
    macro = f1_score(Y, Y_pred, average='macro', zero_division=0)
    return per_label, micro, macro


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='python.csv')
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()

    texts, labels_list = load_and_process(args.data)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels_list)
    global m_labels
    m_labels = list(mlb.classes_)

    print('Tuning thresholds via CV...')
    thresholds = tune_thresholds_cv(texts, Y, folds=args.folds)
    print('Thresholds:', dict(zip(m_labels, thresholds.tolist())))

    per_label, micro, macro = final_evaluate(texts, Y, thresholds)

    out = {
        'generated': datetime.utcnow().isoformat() + 'Z',
        'n_samples': int(Y.shape[0]),
        'labels': m_labels,
        'thresholds': {m_labels[i]: float(thresholds[i]) for i in range(len(m_labels))},
        'micro_f1': float(micro),
        'macro_f1': float(macro),
        'per_label': per_label
    }

    os.makedirs('results', exist_ok=True)
    with open('results/detailed_metrics.json', 'w') as f:
        json.dump(out, f, indent=2)

    # write human-readable report
    lines = []
    lines.append('# Detailed Analysis')
    lines.append(f'Date: {out["generated"]}')
    lines.append(f'- samples: {out["n_samples"]}')
    lines.append(f'- micro-F1 (after thresholding): {out["micro_f1"]:.4f}')
    lines.append(f'- macro-F1: {out["macro_f1"]:.4f}\n')
    lines.append('## Per-label metrics:')
    for lbl, stats in out['per_label'].items():
        lines.append(f'- {lbl}: precision={stats["precision"]:.4f}, recall={stats["recall"]:.4f}, f1={stats["f1"]:.4f}, support={stats["support"]}, threshold={stats["threshold"]:.2f}')

    with open('results/detailed_report.md', 'w') as f:
        f.write('\n'.join(lines))

    print('Detailed analysis saved to results/detailed_metrics.json and results/detailed_report.md')
