"""RoBERTa fine-tuning baseline for multi-label code comment classification (Python subset).

Fine-tunes roberta-base with a sigmoid multi-label head, evaluated via the same
MultilabelStratifiedKFold splits used for the classical ML baselines so results are
directly comparable.
"""
import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, jaccard_score, hamming_loss, precision_score, recall_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from data_utils import load_multilabel

MODEL_NAME = 'roberta-base'
MAX_LEN = 64


def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        enc = tokenizer(list(texts), truncation=True, padding='max_length', max_length=MAX_LEN)
        self.input_ids = enc['input_ids']
        self.attention_mask = enc['attention_mask']
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx]),
            'attention_mask': torch.tensor(self.attention_mask[idx]),
            'labels': torch.tensor(self.labels[idx]),
        }


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        'micro_f1': f1_score(labels, preds, average='micro', zero_division=0),
        'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
        'jaccard': jaccard_score(labels, preds, average='samples', zero_division=0),
        'hamming': hamming_loss(labels, preds),
    }


def run_fold(texts_train, y_train, texts_val, y_val, n_labels, epochs=8, seed=42):
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=n_labels, problem_type='multi_label_classification'
    )

    train_ds = CommentDataset(texts_train, y_train, tokenizer)
    val_ds = CommentDataset(texts_val, y_val, tokenizer)

    args = TrainingArguments(
        output_dir='./roberta_ckpt_tmp',
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='micro_f1',
        greater_is_better=True,
        logging_steps=50,
        seed=seed,
        report_to=[],
        use_mps_device=(get_device() == 'mps'),
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    metrics = trainer.evaluate()

    preds = trainer.predict(val_ds)
    probs = 1 / (1 + np.exp(-preds.predictions))
    y_pred = (probs >= 0.5).astype(int)
    p_micro = precision_score(y_val, y_pred, average='micro', zero_division=0)
    r_micro = recall_score(y_val, y_pred, average='micro', zero_division=0)

    import shutil
    shutil.rmtree('./roberta_ckpt_tmp', ignore_errors=True)

    return {
        'micro_f1': metrics['eval_micro_f1'],
        'macro_f1': metrics['eval_macro_f1'],
        'jaccard': metrics['eval_jaccard'],
        'hamming': metrics['eval_hamming'],
        'precision_micro': p_micro,
        'recall_micro': r_micro,
    }


def run(data_path='python.csv', folds=5, epochs=8):
    started = datetime.utcnow().isoformat() + 'Z'
    os.makedirs('results', exist_ok=True)

    texts, Y, categories, _ = load_multilabel(data_path)
    n_labels = len(categories)

    mskf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    fold_metrics = []
    for i, (train_idx, val_idx) in enumerate(mskf.split(texts, Y), 1):
        print(f'--- Fold {i}/{folds} ---')
        m = run_fold(texts[train_idx], Y[train_idx], texts[val_idx], Y[val_idx], n_labels, epochs=epochs)
        print(f"  micro-F1: {m['micro_f1']:.4f}  macro-F1: {m['macro_f1']:.4f}  jaccard: {m['jaccard']:.4f}")
        fold_metrics.append(m)

    agg = {}
    for k in fold_metrics[0]:
        vals = [m[k] for m in fold_metrics]
        agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    out = {
        'started': started,
        'finished': datetime.utcnow().isoformat() + 'Z',
        'model': MODEL_NAME,
        'n_samples': int(len(texts)),
        'labels': categories,
        'folds': folds,
        'epochs': epochs,
        'fold_metrics': fold_metrics,
        'aggregated': agg,
    }
    with open('results/roberta_results.json', 'w') as f:
        json.dump(out, f, indent=2)

    lines = ['# RoBERTa Fine-tuned Baseline', f'Date: {out["finished"]}', '',
              '## Dataset', f'- samples: {out["n_samples"]}', f'- labels: {categories}',
              f'- folds: {folds}', f'- epochs: {epochs}', '', '## Aggregated Results']
    for metric, vals in agg.items():
        lines.append(f'- {metric}: mean={vals["mean"]:.4f}, std={vals["std"]:.4f}')
    with open('results/roberta_report.md', 'w') as f:
        f.write('\n'.join(lines))

    print('\nSaved to results/roberta_results.json and results/roberta_report.md')
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='python.csv')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=8)
    args = parser.parse_args()
    run(args.data, args.folds, args.epochs)
