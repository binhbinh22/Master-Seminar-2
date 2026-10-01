"""Train and evaluate the classical ML baselines (TF-IDF + linear multi-label models)
for the NLBSE'23 Python code-comment classification subset.

Run from the repo root:
    python -m experiments.ml.train_baseline --config configs/exp_ml_baseline.yaml
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.models.ml.data import load_multilabel
from src.models.ml.evaluation import evaluate_model, per_label_metrics
from src.models.ml.models import build_models


def run(config_path: Path) -> dict:
    with config_path.open(encoding='utf-8') as file:
        config = yaml.safe_load(file)

    started = datetime.now(timezone.utc).isoformat()
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    texts, Y, categories, groups, _ = load_multilabel(config['data']['path'])

    eval_cfg = config['evaluation']
    folds = eval_cfg['folds']
    seed = eval_cfg['seed']

    all_models = build_models()
    requested = config['models']
    unknown = set(requested) - set(all_models)
    if unknown:
        raise ValueError(f'Unknown model(s) in config: {sorted(unknown)}')

    model_results = {}
    for name in requested:
        print(f'Evaluating {name}...')
        clf = all_models[name]
        model_results[name] = evaluate_model(lambda clf=clf: clf, texts, Y, groups, folds=folds, seed=seed)
        m = model_results[name]
        print(f"  micro-F1: {m['micro_f1']['mean']:.4f}  macro-F1: {m['macro_f1']['mean']:.4f}  "
              f"jaccard: {m['jaccard']['mean']:.4f}  hamming: {m['hamming']['mean']:.4f}")

    result = {
        'experiment': config['experiment'],
        'started': started,
        'finished': datetime.now(timezone.utc).isoformat(),
        'n_samples': int(len(texts)),
        'n_unique_groups': int(len(set(groups))),
        'labels': categories,
        'folds': folds,
        'seed': seed,
        'group_aware_split': eval_cfg.get('group_aware_split', True),
        'models': model_results,
    }

    if eval_cfg.get('report_per_label'):
        label_model_name = eval_cfg['per_label_model']
        print(f'Computing per-label metrics for {label_model_name}...')
        clf = all_models[label_model_name]
        result['per_label'] = per_label_metrics(
            lambda clf=clf: clf, texts, Y, groups, categories, folds=folds, seed=seed
        )

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
