import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix


def load_data(path):
    df = pd.read_csv(path, dtype=str, encoding='utf-8', on_bad_lines='skip')
    # text and target
    text_col = next((c for c in ['comment_sentence', 'text', 'comment', 'sentence'] if c in df.columns), df.columns[2])
    target_col = next((c for c in ['category', 'labels', 'label', 'class'] if c in df.columns), None)
    if target_col is None:
        raise ValueError('No target column found')
    texts = df[text_col].fillna('').astype(str).values
    y = df[target_col].fillna('').astype(str).values
    return texts, y, df


def build_vectorizer():
    word_tfidf = TfidfVectorizer(ngram_range=(1,3), analyzer='word', max_features=30000)
    char_tfidf = TfidfVectorizer(ngram_range=(3,6), analyzer='char', max_features=15000)
    return FeatureUnion([('word', word_tfidf), ('char', char_tfidf)])


def run_multiclass(path='python.csv', folds=5):
    started = datetime.utcnow().isoformat() + 'Z'
    os.makedirs('results', exist_ok=True)

    texts, y, df = load_data(path)
    classes, inv = np.unique(y, return_inverse=True)

    vec = build_vectorizer()
    X = vec.fit_transform(texts)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    metrics = {'accuracy': [], 'micro_f1': [], 'macro_f1': []}
    per_class_accum = {c: {'precision': [], 'recall': [], 'f1': [], 'support': 0} for c in classes}

    fold = 1
    for train_idx, test_idx in skf.split(X, inv):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = inv[train_idx], inv[test_idx]

        clf = LogisticRegression(max_iter=1000, solver='saga', multi_class='auto', class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
        macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

        metrics['accuracy'].append(acc)
        metrics['micro_f1'].append(micro)
        metrics['macro_f1'].append(macro)

        p, r, f1, sup = precision_recall_fscore_support(y_test, y_pred, labels=range(len(classes)), zero_division=0)
        for i, c in enumerate(classes):
            per_class_accum[c]['precision'].append(float(p[i]))
            per_class_accum[c]['recall'].append(float(r[i]))
            per_class_accum[c]['f1'].append(float(f1[i]))
            per_class_accum[c]['support'] += int(sup[i])

        fold += 1

    # aggregate
    agg = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in metrics.items()}
    per_class_final = {}
    for c, stats in per_class_accum.items():
        per_class_final[c] = {
            'precision_mean': float(np.mean(stats['precision'])) if stats['precision'] else 0.0,
            'recall_mean': float(np.mean(stats['recall'])) if stats['recall'] else 0.0,
            'f1_mean': float(np.mean(stats['f1'])) if stats['f1'] else 0.0,
            'support': int(stats['support'])
        }

    results = {
        'started': started,
        'finished': datetime.utcnow().isoformat() + 'Z',
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'n_classes': int(len(classes)),
        'classes': list(classes),
        'metrics': agg,
        'per_class': per_class_final
    }

    with open('results/multiclass_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # write report including steps performed
    lines = []
    lines.append('# Multiclass Pipeline Report')
    lines.append(f'Date: {results["finished"]}')
    lines.append('\n## What I did')
    lines.append('- Loaded CSV and detected `category` as the target column (single-label).')
    lines.append('- Built TF‑IDF features: word ngrams (1-3) and char ngrams (3-6).')
    lines.append('- Used `LogisticRegression` (multinomial) with `StratifiedKFold` for evaluation.')
    lines.append('- Computed accuracy, micro-F1, macro-F1, and per-class precision/recall/F1.')

    lines.append('\n## Results (averaged across folds)')
    for k, v in agg.items():
        lines.append(f'- {k}: mean={v["mean"]:.4f}, std={v["std"]:.4f}')

    lines.append('\n## Per-class metrics')
    for c, s in per_class_final.items():
        lines.append(f'- {c}: precision={s["precision_mean"]:.4f}, recall={s["recall_mean"]:.4f}, f1={s["f1_mean"]:.4f}, support={s["support"]}')

    lines.append('\n## Notes & Next steps')
    lines.append('- If you intended multi-label classification, the dataset appears single-label; confirm source labels.')
    lines.append('- Next: try ClassifierChains or LightGBM, tune thresholds, add handcrafted features.')

    with open('results/multiclass_report.md', 'w') as f:
        f.write('\n'.join(lines))

    print('Multiclass pipeline complete. Results in results/multiclass_results.json and results/multiclass_report.md')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='python.csv')
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()
    run_multiclass(args.data, args.folds)
