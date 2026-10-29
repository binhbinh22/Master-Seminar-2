import json
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, jaccard_score, hamming_loss, precision_score, recall_score

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except Exception:
    print("Missing dependency: install iterative-stratification (pip install iterative-stratification)")
    raise


def load_data(path):
    df = pd.read_csv(path, dtype=str, encoding='utf-8', on_bad_lines='skip')
    print('Columns detected:', list(df.columns))
    # choose text column
    text_col = None
    for candidate in ['comment_sentence', 'text', 'comment', 'sentence']:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        # fallback to third column
        text_col = df.columns[2]
    # choose label column
    label_col = None
    for candidate in ['labels', 'label', 'category', 'topics', 'class']:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError('No label column found')

    texts = df[text_col].fillna('').astype(str).values
    raw_labels = df[label_col].fillna('').astype(str).values

    # parse labels into lists
    labels_list = []
    for s in raw_labels:
        s = s.strip()
        if s == '':
            labels_list.append([])
        else:
            # split on common separators
            parts = [p.strip() for p in re_split.split(s) if p.strip()]
            labels_list.append(parts)

    return texts, labels_list


import re
re_split = re.compile(r"[;|,\\/]+")


def run(path='python.csv', k=5):
    texts, labels_list = load_data(path)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels_list)
    print('Number of labels:', len(mlb.classes_))

    tf = TfidfVectorizer(ngram_range=(1,3), analyzer='word', max_features=30000)
    X = tf.fit_transform(texts)

    mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    micro_f1s = []
    macro_f1s = []
    jaccards = []
    hammings = []

    fold = 1
    for train_idx, test_idx in mskf.split(X, Y):
        print(f'Fold {fold}')
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        clf = OneVsRestClassifier(LogisticRegression(solver='saga', max_iter=1000, class_weight='balanced'))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
        macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        jac = jaccard_score(y_test, y_pred, average='samples', zero_division=0)
        ham = hamming_loss(y_test, y_pred)

        print(f'  micro-F1: {micro:.4f}  macro-F1: {macro:.4f}  jaccard: {jac:.4f}  hamming: {ham:.4f}')

        micro_f1s.append(micro)
        macro_f1s.append(macro)
        jaccards.append(jac)
        hammings.append(ham)
        fold += 1

    results = {
        'micro_f1_mean': float(np.mean(micro_f1s)),
        'micro_f1_std': float(np.std(micro_f1s)),
        'macro_f1_mean': float(np.mean(macro_f1s)),
        'jaccard_mean': float(np.mean(jaccards)),
        'hamming_mean': float(np.mean(hammings)),
        'labels': list(mlb.classes_)
    }

    with open('baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\nAveraged results saved to baseline_results.json')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='python.csv')
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()
    run(args.data, args.folds)
