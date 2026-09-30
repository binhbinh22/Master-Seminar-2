"""Classical ML baselines for multi-label code comment classification (Python subset).

Uses TF-IDF (word + char n-grams) features with several linear multi-label classifiers,
evaluated via MultilabelStratifiedKFold cross-validation.
"""
import json
import os
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, jaccard_score, hamming_loss, precision_score, recall_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from src.models.ml.data_utils import load_multilabel


def build_vectorizer():
    word_tfidf = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', min_df=2, max_features=20000)
    char_tfidf = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', min_df=2, max_features=20000)
    return FeatureUnion([('word', word_tfidf), ('char', char_tfidf)])


def build_models():
    return {
        'ovr_logreg': OneVsRestClassifier(
            LogisticRegression(max_iter=2000, class_weight='balanced', C=5.0)
        ),
        'classifier_chain_logreg': ClassifierChain(
            LogisticRegression(max_iter=2000, class_weight='balanced', C=5.0),
            order='random', random_state=42,
        ),
        'ovr_linear_svc': OneVsRestClassifier(
            LinearSVC(class_weight='balanced', C=1.0)
        ),
        'ovr_sgd': OneVsRestClassifier(
            SGDClassifier(loss='log_loss', max_iter=2000, tol=1e-3, class_weight='balanced', random_state=42)
        ),
    }


def evaluate_model(build_clf_fn, texts, Y, folds=5, seed=42):
    mskf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    metrics = {'micro_f1': [], 'macro_f1': [], 'jaccard': [], 'hamming': [],
               'precision_micro': [], 'recall_micro': []}

    for train_idx, test_idx in mskf.split(texts, Y):
        vec = build_vectorizer()
        X_train = vec.fit_transform(texts[train_idx])
        X_test = vec.transform(texts[test_idx])
        y_train, y_test = Y[train_idx], Y[test_idx]

        clf = build_clf_fn()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred = np.asarray(y_pred)

        metrics['micro_f1'].append(f1_score(y_test, y_pred, average='micro', zero_division=0))
        metrics['macro_f1'].append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        metrics['jaccard'].append(jaccard_score(y_test, y_pred, average='samples', zero_division=0))
        metrics['hamming'].append(hamming_loss(y_test, y_pred))
        metrics['precision_micro'].append(precision_score(y_test, y_pred, average='micro', zero_division=0))
        metrics['recall_micro'].append(recall_score(y_test, y_pred, average='micro', zero_division=0))

    return {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in metrics.items()}


def run(data_path='python.csv', folds=5):
    started = datetime.utcnow().isoformat() + 'Z'
    os.makedirs('results', exist_ok=True)

    texts, Y, categories, _ = load_multilabel(data_path)
    models = build_models()

    results = {}
    for name, clf in models.items():
        print(f'Evaluating {name}...')
        results[name] = evaluate_model(lambda clf=clf: clf, texts, Y, folds=folds)
        print(f"  micro-F1: {results[name]['micro_f1']['mean']:.4f}  "
              f"macro-F1: {results[name]['macro_f1']['mean']:.4f}  "
              f"jaccard: {results[name]['jaccard']['mean']:.4f}")

    out = {
        'started': started,
        'finished': datetime.utcnow().isoformat() + 'Z',
        'n_samples': int(len(texts)),
        'labels': categories,
        'folds': folds,
        'models': results,
    }
    with open('results/ml_pipeline_results.json', 'w') as f:
        json.dump(out, f, indent=2)

    lines = ['# Classical ML Baseline (corrected multi-label data)', f'Date: {out["finished"]}',
              '', '## Dataset', f'- samples (unique comments): {out["n_samples"]}',
              f'- labels: {categories}', f'- folds: {folds}', '', '## Results']
    for name, stats in results.items():
        lines.append(f'\n### {name}')
        for metric, vals in stats.items():
            lines.append(f'- {metric}: mean={vals["mean"]:.4f}, std={vals["std"]:.4f}')
    with open('results/ml_pipeline_report.md', 'w') as f:
        f.write('\n'.join(lines))

    print('\nSaved to results/ml_pipeline_results.json and results/ml_pipeline_report.md')
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='python.csv')
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()
    run(args.data, args.folds)
