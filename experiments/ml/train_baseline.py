"""Train and evaluate the classical ML baselines (TF-IDF + linear multi-label models)
for the NLBSE'23 code-comment classification dataset.

Reads the shared, group-aware split in `data/processed/<language>/` -- the same split
the team's RoBERTa baseline (`experiments/dl/train_exp1.py`) uses -- so the two methods'
held-out test metrics are directly comparable.

Run from the repo root:
    python -m experiments.ml.train_baseline --config configs/exp_ml_baseline.yaml
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.models.ml.data import load_processed
from src.models.ml.evaluation import cross_validate, evaluate_on_test
from src.models.ml.models import build_models


def run(config_path: Path) -> dict:
    with config_path.open(encoding='utf-8') as file:
        config = yaml.safe_load(file)

    started = datetime.now(timezone.utc).isoformat()
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    language = config['data']['language']
    bundle = load_processed(language, processed_dir=config['data']['processed_dir'])
    label_names = bundle['label_names']
    folds = bundle['folds']
    feature_kwargs = config.get('features', {})

    all_models = build_models()
    requested = config['models']
    unknown = set(requested) - set(all_models)
    if unknown:
        raise ValueError(f'Unknown model(s) in config: {sorted(unknown)}')

    cv_results = {}
    test_results = {}
    per_label_results = {}

    for name in requested:
        clf = all_models[name]
        print(f'[{name}] cross-validating on train split ({len(folds)} folds)...')
        cv_results[name] = cross_validate(
            lambda clf=clf: clf, bundle['train_texts'], bundle['train_labels'], folds,
            feature_kwargs=feature_kwargs,
        )
        cv = cv_results[name]
        print(f"  CV   micro-F1: {cv['micro_f1']['mean']:.4f}  macro-F1: {cv['macro_f1']['mean']:.4f}  "
              f"jaccard: {cv['jaccard']['mean']:.4f}  hamming: {cv['hamming']['mean']:.4f}")

        print(f'[{name}] fitting on full train, evaluating on held-out test...')
        overall, per_label = evaluate_on_test(
            lambda clf=clf: clf,
            bundle['train_texts'], bundle['train_labels'],
            bundle['test_texts'], bundle['test_labels'],
            label_names,
            feature_kwargs=feature_kwargs,
        )
        test_results[name] = overall
        per_label_results[name] = per_label
        print(f"  TEST micro-F1: {overall['micro_f1']:.4f}  macro-F1: {overall['macro_f1']:.4f}  "
              f"jaccard: {overall['jaccard']:.4f}  hamming: {overall['hamming']:.4f}")

    result = {
        'experiment': config['experiment'],
        'language': language,
        'started': started,
        'finished': datetime.now(timezone.utc).isoformat(),
        'n_train': int(len(bundle['train_texts'])),
        'n_test': int(len(bundle['test_texts'])),
        'labels': label_names,
        'folds': len(folds),
        'data_source': 'data/processed (shared group-aware split, same as RoBERTa baseline)',
        'cv': cv_results,
        'test': test_results,
        'per_label_test': per_label_results,
    }

    metrics_path = output_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'\nSaved metrics to {metrics_path}')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/exp_ml_baseline.yaml'))
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
