"""Shared E1–E12 runner using the saved processed splits and folds."""

from __future__ import annotations

import argparse
import json
import hashlib
import time
import statistics
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from src.data.nlbse2023 import TokenizedNLBSEDataset
from src.data.processed import load_processed
from src.evaluation.multilabel import SUMMARY_METRICS, evaluate_masked_multilabel
from src.evaluation.summaries import summarize_metric_runs
from src.models.dl.zoo import build_model
from src.models.dl.advanced_variants import describe_label_graph
from src.evaluation.thresholds import predict_probabilities, tune_thresholds
from src.training.masked_classifier import fit_masked_classifier
from src.training.reproducibility import set_seed
from src.training.devices import resolve_device
from src.visualization.tensorboard import log_metrics
from src.visualization.training_plots import save_training_plots


def make_loader(bundle, tokenizer, max_length, split, indices, batch_size, workers, shuffle=False):
    dataset = TokenizedNLBSEDataset(bundle, tokenizer, split, max_length, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)


def train_model(config, bundle, tokenizer, device, train_indices, val_indices=None,
                epochs=None, checkpoint_path=None, seed=None, log_dir=None, max_run_epochs=None):
    """Build a model and datasets, then use the shared training loop."""
    training, model_config = config["training"], config["model"]
    set_seed(training["seed"] if seed is None else seed)
    length = config["tokenizer"]["max_length"]
    train_loader = make_loader(bundle, tokenizer, length, "train", train_indices,
                               training["batch_size"], training["num_workers"], True)
    val_loader = None if val_indices is None else make_loader(
        bundle, tokenizer, length, "train", val_indices,
        training["eval_batch_size"], training["num_workers"])
    model = build_model(config, bundle, train_indices)
    model = model.to(device)
    fit = fit_masked_classifier(
        model, train_loader, val_loader, device=device, training=training,
        threshold=model_config["threshold"], label_names=bundle.label_names,
        report_per_label=config["evaluation"]["report_per_label"],
        epochs=epochs, checkpoint_path=checkpoint_path, log_dir=log_dir,
        max_run_epochs=max_run_epochs,
    )
    if model.loss_fn.pos_weight is not None:
        fit["pos_weight"] = model.loss_fn.pos_weight.detach().cpu().tolist()
    if hasattr(model, "graph"):
        fit["label_graph"] = describe_label_graph(
            model.graph.adjacency, bundle.label_names,
            language=bundle.languages[train_indices[0]], train_size=len(train_indices),
            threshold=model_config["graph_threshold"],
        )
    if hasattr(model, "head_version"):
        fit["head_version"] = int(model.head_version)
    return model, fit


def run(config_path: Path, *, language: str | None = None, data_dir=None, output_dir=None,
        device=None) -> dict:
    started = time.monotonic()
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    device = resolve_device(device if device is not None else config["training"].get("device", "auto"))
    config["training"]["device"] = str(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    else:
        print("Device: cpu", flush=True)
    language = language or config["dataset"]["language"]
    if language not in ("python", "java", "pharo"):
        raise ValueError("language must be python, java or pharo")
    if data_dir is not None:
        config["dataset"]["data_dir"] = str(data_dir)
    config["dataset"]["language"] = language
    bundle = load_processed(config["dataset"]["data_dir"], language)
    model_config, training = config["model"], config["training"]
    model_config["num_labels"] = len(bundle.label_names)
    fold_indices = bundle.folds
    training["folds"] = len(fold_indices)
    if model_config["hidden_size"] != 768:
        raise ValueError("Experiments require a 768-dimensional RoBERTa/CodeBERT backbone")
    if not 0 < model_config["threshold"] < 1:
        raise ValueError("threshold must be between 0 and 1")
    for key in ("batch_size", "eval_batch_size", "gradient_accumulation_steps",
                "epochs", "early_stopping_patience"):
        if training[key] < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 <= training["warmup_ratio"] < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"], use_fast=True)
    output_dir = Path(output_dir) if output_dir is not None else Path(config["output_dir"]) / language
    config["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = output_dir / "tensorboard" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output_dir / "dataset_config.yaml").write_text(
        yaml.safe_dump(config["dataset"], sort_keys=False), encoding="utf-8")
    (output_dir / "label_cols.json").write_text(json.dumps(list(bundle.label_names), indent=2) + "\n")
    (output_dir / "folds.json").write_text(json.dumps(bundle.folds) + "\n")
    all_train_indices = list(range(bundle.train_size))
    tune = config["variant"] == "threshold_tuning"
    oof_scores = torch.empty((bundle.train_size, len(bundle.label_names))) if tune else None
    fingerprint = hashlib.sha256()
    for name in ("train.csv", "test.csv", "folds.json", "label_cols.json"):
        fingerprint.update(name.encode())
        fingerprint.update((Path(bundle.data_dir) / name).read_bytes())
    result = {"data_fingerprint": fingerprint.hexdigest(), "variant": config["variant"],
              "experiment": config["experiment"], "dataset_samples": len(bundle),
              "device": str(device),
              "language": language, "data_dir": bundle.data_dir,
              "seed": training["seed"], "threshold": model_config["threshold"],
              "labels": list(bundle.label_names), "fold_count": len(fold_indices),
              "split_unit": "normalized_text_group", "split_source": "processed",
              "tensorboard_log_dir": str(tensorboard_dir),
              "folds": []}
    for fold in fold_indices:
        fold_number = fold["fold"]
        train_indices, val_indices = fold["train_indices"], fold["val_indices"]
        print(f"fold {fold_number + 1}/{len(fold_indices)}: train={len(train_indices)}, validation={len(val_indices)}", flush=True)
        model, fit = train_model(config, bundle, tokenizer, device, train_indices,
                                 val_indices, seed=training["seed"] + fold_number,
                                 log_dir=tensorboard_dir / f"fold_{fold_number}")
        result["folds"].append({"fold": fold_number, "train_sentences": len(train_indices),
                                "validation_sentences": len(val_indices),
                                "train_indices": train_indices, "validation_indices": val_indices, **fit})
        if tune:
            loader = make_loader(bundle, tokenizer, config["tokenizer"]["max_length"],
                                 "train", val_indices, training["eval_batch_size"], training["num_workers"])
            oof_scores[val_indices] = predict_probabilities(
                model, loader, device, show_progress=training.get("progress", True))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result["cross_validation"] = summarize_metric_runs(
        [fold["validation"] for fold in result["folds"]], SUMMARY_METRICS,
    )
    with SummaryWriter(log_dir=str(tensorboard_dir / "summary")) as writer:
        for fold in result["folds"]:
            log_metrics(writer, "fold_validation", fold["validation"], fold["fold"])
        for name, values in result["cross_validation"].items():
            if values["mean"] is not None:
                writer.add_scalar(f"cross_validation_mean/{name}", values["mean"], 0)
                writer.add_scalar(f"cross_validation_std/{name}", values["std"], 0)
    threshold = model_config["threshold"]
    if tune:
        tuning = config["threshold_tuning"]
        threshold = tune_thresholds(oof_scores, bundle.labels[:bundle.train_size], tuning["grid"])
        torch.save({"scores": oof_scores, "labels": bundle.labels[:bundle.train_size]}, output_dir / "oof_predictions.pt")
        result["threshold_tuning"] = {"source": "out_of_fold_train", "objective": "per_label_f1",
                                      "grid": tuning["grid"], "thresholds": threshold.tolist()}
    thresholds = threshold.tolist() if isinstance(threshold, torch.Tensor) else [threshold] * len(bundle.label_names)
    (output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
    result["thresholds"] = thresholds
    final_epochs = max(1, round(statistics.median(fold["best_epoch"] for fold in result["folds"])))
    result["final_training_epochs"] = final_epochs
    print(f"training final checkpoint for {final_epochs} epochs", flush=True)
    model, final_fit = train_model(config, bundle, tokenizer, device, all_train_indices,
                                   epochs=final_epochs, checkpoint_path=output_dir / "best_model.pt",
                                   log_dir=tensorboard_dir / "final")
    result["final_training"] = final_fit
    if "label_graph" in final_fit:
        (output_dir / "label_graph.json").write_text(
            json.dumps(final_fit["label_graph"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result["train_sentences"] = len(all_train_indices)
    if config["evaluation"]["run_test_after_training"]:
        test_loader = make_loader(bundle, tokenizer, config["tokenizer"]["max_length"],
                                  "test", None, training["eval_batch_size"], training["num_workers"])
        result["test_sentences"] = len(test_loader.dataset)
        result["test"] = evaluate_masked_multilabel(
            model, test_loader, device, threshold,
            bundle.label_names, config["evaluation"]["report_per_label"],
            show_progress=training.get("progress", True),
        )
        with SummaryWriter(log_dir=str(tensorboard_dir / "final")) as writer:
            log_metrics(writer, "test", result["test"], final_epochs)
        print(f"test_micro_f1={result['test']['micro_f1']:.4f}", flush=True)
    result["duration_seconds"] = time.monotonic() - started
    # Persist metrics before plotting so completed training survives a plotting error.
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    histories = {f"Fold {fold['fold']}": fold["history"] for fold in result["folds"]}
    histories["Final model"] = result["final_training"]["history"]
    result["plots"] = save_training_plots(
        output_dir / "plots", histories, validation_metric="micro_f1",
        cross_validation=result["cross_validation"], test_metrics=result.get("test"),
    )
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def experiment_main(default_config):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(default_config))
    parser.add_argument("--language", choices=("python", "java", "pharo", "all"))
    parser.add_argument("--data-dir", type=Path, help="Root containing python/, java/, pharo/")
    parser.add_argument("--output-dir", type=Path, help="Exact output directory; with all, append language")
    parser.add_argument("--device", help="auto, cpu, cuda:0, cuda:1, ... (default: config or auto)")
    args = parser.parse_args()
    languages = ("python", "java", "pharo") if args.language == "all" else (args.language,)
    for language in languages:
        output = args.output_dir
        if output is not None and args.language == "all":
            output = output / language
        run(args.config, language=language, data_dir=args.data_dir, output_dir=output, device=args.device)
