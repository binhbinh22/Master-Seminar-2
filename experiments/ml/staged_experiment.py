"""Run one classical-ML method on one language, for one stage of the team's shared
two-stage protocol (see docs/two_stage_training.md). Mirrors the output contract used by
experiments/dl/staged_experiment.py so ML and RoBERTa/CodeBERT runs are diffable:

<output>/<method>/<language>/
    config.yaml, data_manifest.json, label_cols.json, metrics.json, predictions.csv,
    training.log, plots/, artifacts/ (final stage only)

Stage "kfold": 5-fold CV over 100% of the language's data, for comparing ML methods
against each other (and later, against the DL k-fold leaderboard) on equal footing.
Produces out-of-fold predictions covering every sentence exactly once.

Stage "final": fit once on the 80% train split, report validation (diagnostic only --
classical linear models have no per-epoch checkpoint to select) and a single held-out
test score. Same EVALUATION_NOTE caveat as the DL side: stage-1 screening already saw
every sentence, so the stage-2 test score is exploratory, not independent of method
selection.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.base import clone

from src.evaluation.multilabel import SUMMARY_METRICS
from src.evaluation.summaries import summarize_metric_runs
from src.models.ml.data import load
from src.models.ml.features import build_vectorizer
from src.models.ml.metrics import compute_metrics
from src.models.ml.models import build_models
from src.models.ml.scoring import predict_scores
from src.visualization.training_plots import save_training_plots

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_NOTE = ("Exploratory evaluation: stage-1 method selection used all rows, "
                   "including the rows assigned to stage-2 validation/test. "
                   "Stage-2 test is not independent of method selection.")


def resolve_output_dir(stage, timestamp, method, language, output_dir=None, base_dir=None):
    if output_dir is not None:
        output = Path(output_dir)
    else:
        base = Path(base_dir) if base_dir is not None else ROOT / "results" / "training" / stage / timestamp
        output = base / method / language
    if (output / "metrics.json").exists():
        raise ValueError(f"Refusing to overwrite completed results at {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def make_logger(output_dir):
    logger = logging.getLogger(f"ml.{output_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    file_handler = logging.FileHandler(output_dir / "training.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def write_predictions_csv(path, rows, label_names):
    import csv
    fieldnames = ["comment_sentence_id", "class", "split"]
    for label in label_names:
        fieldnames += [f"{label}_true", f"{label}_score", f"{label}_pred"]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prediction_rows(sentence_ids, class_names, split_labels, indices, y_true, y_score, y_pred, label_names):
    rows = []
    for row, index in enumerate(indices):
        entry = {"comment_sentence_id": sentence_ids[index], "class": class_names[index],
                 "split": split_labels[row] if isinstance(split_labels, list) else split_labels}
        for j, label in enumerate(label_names):
            entry[f"{label}_true"] = int(y_true[row, j])
            entry[f"{label}_score"] = float(y_score[row, j])
            entry[f"{label}_pred"] = int(y_pred[row, j])
        rows.append(entry)
    return rows


def model_config(clf):
    params = {k: v for k, v in clf.get_params().items() if not k.startswith("estimator__")}
    base = clf.get_params().get("estimator")
    return {"wrapper": type(clf).__name__, "wrapper_params": {k: v for k, v in params.items() if k != "estimator"},
            "base_estimator": type(base).__name__ if base is not None else None,
            "base_params": base.get_params() if base is not None else None}


def run_staged(method, language, stage, *, output_dir=None, output_base=None, seed=42, data_dir=None):
    if stage not in ("kfold", "final"):
        raise ValueError("stage must be kfold or final")
    started = time.monotonic()
    data = load(language, stage, data_dir=data_dir)
    bundle = data["bundle"]
    texts, labels, label_names = data["texts"], data["labels"], data["label_names"]
    sentence_ids, class_names = data["sentence_ids"], data["class_names"]
    all_models = build_models(seed=seed)
    if method not in all_models:
        raise ValueError(f"Unknown method: {method}")
    clf_template = all_models[method]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = resolve_output_dir(stage, timestamp, method, language, output_dir=output_dir, base_dir=output_base)
    logger = make_logger(output)

    vectorizer_params = {"word_ngram_range": [1, 2], "char_ngram_range": [3, 5], "max_features_per_vectorizer": 20000}
    config = {"method": method, "model_type": "classical_ml", "language": language, "stage": stage, "seed": seed,
              "features": {"type": "tfidf_word_char", **vectorizer_params},
              "model": model_config(clf_template), "output_dir": str(output)}
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "label_cols.json").write_text(json.dumps(label_names, indent=2) + "\n")
    (output / "data_manifest.json").write_text((Path(bundle.data_dir) / "manifest.json").read_text())

    result = {"stage": stage, "protocol": bundle.protocol, "method": method, "model_type": "classical_ml",
              "language": language, "labels": label_names, "seed": seed, "device": "cpu",
              "data_dir": bundle.data_dir, "dataset_samples": len(texts),
              "data_fingerprint": bundle.data_fingerprint, "population_fingerprint": bundle.population_fingerprint,
              "evaluation_note": EVALUATION_NOTE, "test_independent_of_method_selection": False,
              "split_unit": "normalized_text_group"}

    if stage == "kfold":
        folds = bundle.folds
        (output / "folds.json").write_text(json.dumps(folds) + "\n")
        n, n_labels = len(texts), len(label_names)
        oof_score = np.full((n, n_labels), np.nan)
        oof_pred = np.zeros((n, n_labels), dtype=int)
        split_labels = [""] * n
        fold_records = []
        for fold in folds:
            number = fold["fold"]
            train_idx = np.array(fold["train_indices"])
            val_idx = np.array(fold["val_indices"])
            logger.info("%s %s: screening fold %d/%d", method, language, number + 1, len(folds))
            fit_start = time.monotonic()
            vec = build_vectorizer()
            X_train = vec.fit_transform(texts[train_idx])
            clf = clone(clf_template)
            clf.fit(X_train, labels[train_idx])
            fit_seconds = time.monotonic() - fit_start

            predict_start = time.monotonic()
            X_val = vec.transform(texts[val_idx])
            y_score = predict_scores(clf, X_val)
            y_pred = np.asarray(clf.predict(X_val))
            predict_seconds = time.monotonic() - predict_start

            metrics = compute_metrics(labels[val_idx], y_score, y_pred, label_names, per_label=True)
            fold_records.append({**metrics, "fold": number, "train_size": len(train_idx),
                                 "val_size": len(val_idx), "fit_seconds": fit_seconds,
                                 "predict_seconds": predict_seconds})
            oof_score[val_idx] = y_score
            oof_pred[val_idx] = y_pred
            for i in val_idx:
                split_labels[i] = f"fold_{number}"
            logger.info("  micro-F1=%.4f macro-F1=%.4f hamming=%.4f", metrics["micro_f1"],
                       metrics["macro_f1"], metrics["hamming_loss"])

        result["folds"] = fold_records
        result["cross_validation"] = summarize_metric_runs(fold_records, SUMMARY_METRICS)
        result["cv_population_sentences"] = n
        result["test_sentences"] = 0
        result["selection_metric"] = "cv_micro_f1_mean"
        oof_metrics = compute_metrics(labels, oof_score, oof_pred, label_names, per_label=True)
        result["out_of_fold"] = oof_metrics
        rows = prediction_rows(sentence_ids, class_names, split_labels, range(n), labels, oof_score, oof_pred, label_names)
        write_predictions_csv(output / "predictions.csv", rows, label_names)
        test_metrics_for_plot = None
    else:
        train_size = bundle.train_size
        train_idx = np.arange(train_size)
        val_idx = np.array(bundle.validation_indices)
        test_idx = np.array(bundle.test_indices)

        fit_start = time.monotonic()
        vec = build_vectorizer()
        X_train = vec.fit_transform(texts[train_idx])
        clf = clone(clf_template)
        clf.fit(X_train, labels[train_idx])
        fit_seconds = time.monotonic() - fit_start
        logger.info("%s %s: fit on %d train sentences in %.2fs", method, language, train_size, fit_seconds)

        predict_start = time.monotonic()
        X_val = vec.transform(texts[val_idx])
        val_score = predict_scores(clf, X_val)
        val_pred = np.asarray(clf.predict(X_val))
        X_test = vec.transform(texts[test_idx])
        test_score = predict_scores(clf, X_test)
        test_pred = np.asarray(clf.predict(X_test))
        predict_seconds = time.monotonic() - predict_start

        validation_metrics = compute_metrics(labels[val_idx], val_score, val_pred, label_names, per_label=True)
        test_metrics = compute_metrics(labels[test_idx], test_score, test_pred, label_names, per_label=True)
        logger.info("  validation micro-F1=%.4f  test micro-F1=%.4f", validation_metrics["micro_f1"],
                   test_metrics["micro_f1"])

        result.update(train_sentences=train_size, validation_sentences=len(val_idx), test_sentences=len(test_idx),
                      validation=validation_metrics, test=test_metrics,
                      validation_note="Diagnostic only -- not used to select hyperparameters or a checkpoint; "
                                       "classical linear models here have no per-epoch state to select.",
                      fit_seconds=fit_seconds, predict_seconds=predict_seconds)

        rows = prediction_rows(sentence_ids, class_names, "validation", val_idx, labels[val_idx], val_score, val_pred, label_names)
        rows += prediction_rows(sentence_ids, class_names, "test", test_idx, labels[test_idx], test_score, test_pred, label_names)
        write_predictions_csv(output / "predictions.csv", rows, label_names)

        artifacts = output / "artifacts"
        artifacts.mkdir(exist_ok=True)
        joblib.dump(vec, artifacts / "vectorizer.joblib")
        joblib.dump(clf, artifacts / "model.joblib")
        test_metrics_for_plot = test_metrics

    result.update(duration_seconds=time.monotonic() - started, complete=True)
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    plots = save_training_plots(output / "plots", {}, cross_validation=result.get("cross_validation"),
                                test_metrics=test_metrics_for_plot)
    result["plots"] = plots
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("Done in %.1fs. Output: %s", result["duration_seconds"], output)
    return result
