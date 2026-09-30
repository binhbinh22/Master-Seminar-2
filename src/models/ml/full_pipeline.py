import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, jaccard_score, hamming_loss

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except Exception:
    raise


re_split = re.compile(r"[;|,\\/]+")


def load_and_process(path):
    df = pd.read_csv(path, dtype=str, encoding='utf-8', on_bad_lines='skip')
    # detect text and label columns
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


def evaluate_model(clf, X, Y, folds=5):
    mskf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    metrics = {'micro_f1': [], 'macro_f1': [], 'jaccard': [], 'hamming': []}

    for train_idx, test_idx in mskf.split(X, Y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        metrics['micro_f1'].append(f1_score(y_test, y_pred, average='micro', zero_division=0))
        metrics['macro_f1'].append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        metrics['jaccard'].append(jaccard_score(y_test, y_pred, average='samples', zero_division=0))
        metrics['hamming'].append(hamming_loss(y_test, y_pred))

    # aggregate
    agg = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in metrics.items()}
    return agg


def run_all(data_path='python.csv', folds=5):
    started = datetime.utcnow().isoformat() + 'Z'
    os.makedirs('results', exist_ok=True)

    texts, labels_list = load_and_process(data_path)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels_list)

    vec = build_vectorizer()
    X = vec.fit_transform(texts)

    models = {}

    # 1) One-vs-Rest Logistic Regression
    clf1 = OneVsRestClassifier(LogisticRegression(solver='saga', max_iter=1000, class_weight='balanced'))
    models['ovr_logreg'] = evaluate_model(clf1, X, Y, folds=folds)

    # 2) Classifier Chains (with LogisticRegression base)
    base = LogisticRegression(solver='saga', max_iter=1000, class_weight='balanced')
    chain = ClassifierChain(base, order='random', random_state=42)
    models['classifier_chain'] = evaluate_model(chain, X, Y, folds=folds)

    # 3) One-vs-Rest SGDClassifier (fast linear, hinge loss / SVM-like)
    sgd = OneVsRestClassifier(SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3))
    models['ovr_sgd'] = evaluate_model(sgd, X, Y, folds=folds)

    results = {
        'started': started,
        'finished': datetime.utcnow().isoformat() + 'Z',
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'labels': list(mlb.classes_),
        'models': models
    }

    with open('results/full_pipeline_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # produce a simple report
    lines = []
    lines.append('# Full ML Baseline Report')
    lines.append(f'Date: {results["finished"]}')
    lines.append('\n## Dataset')
    lines.append(f'- samples: {results["n_samples"]}')
    lines.append(f'- features: {results["n_features"]}')
    lines.append(f'- labels: {results["labels"]}')

    lines.append('\n## Models and Results')
    for name, stats in models.items():
        lines.append(f'\n### {name}')
        for metric, vals in stats.items():
            lines.append(f'- {metric}: mean={vals["mean"]:.4f}, std={vals["std"]:.4f}')

    report_path = 'results/report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print('Full pipeline finished. Results in results/full_pipeline_results.json and results/report.md')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='python.csv')
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()
    run_all(args.data, args.folds)
