"""Train one real E9 fold for one epoch, preserving the full 10-epoch LR schedule."""

import argparse
import gc
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from experiments.dl.roberta_experiment import train_model
from src.data.processed import load_processed
from src.training.devices import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp9_roberta_label_gnn.yaml"))
    parser.add_argument("--language", choices=("python", "java", "pharo", "all"), default="all")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/diagnostics") / f"e9_v2_first_epoch_{datetime.now():%Y%m%d-%H%M%S}")
    args = parser.parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config = yaml.safe_load(args.config.read_text())
    if config["variant"] != "label_gnn":
        parser.error("This diagnostic is for E9 only")
    languages = ("python", "java", "pharo") if args.language == "all" else (args.language,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], use_fast=True)
    summaries = []
    for language in languages:
        started = time.monotonic()
        bundle = load_processed(config["dataset"]["data_dir"], language)
        if not 0 <= args.fold < len(bundle.folds):
            parser.error("fold index is outside folds.json")
        fold = bundle.folds[args.fold]
        config["dataset"]["language"] = language
        config["model"]["num_labels"] = len(bundle.label_names)
        config["training"]["device"] = str(device)
        output = args.output_dir / language / f"fold_{args.fold}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"CHECK E9 v2 / {language} / fold {args.fold} / {device}; "
              f"run 1 epoch, scheduler planned for {config['training']['epochs']} epochs", flush=True)
        model, fit = train_model(
            config, bundle, tokenizer, device, fold["train_indices"], fold["val_indices"],
            seed=config["training"]["seed"] + args.fold,
            log_dir=output / "tensorboard", max_run_epochs=1,
        )
        validation = fit["validation"]
        fingerprint = hashlib.sha256()
        for name in ("train.csv", "test.csv", "folds.json", "label_cols.json"):
            fingerprint.update(name.encode())
            fingerprint.update((Path(bundle.data_dir) / name).read_bytes())
        result = {
            "diagnostic": "first_epoch_only_not_full_cv", "language": language,
            "device": str(device), "fold": args.fold, "threshold": config["model"]["threshold"],
            "data_fingerprint": fingerprint.hexdigest(), "head_version": int(model.head_version),
            "labels": list(bundle.label_names), "train_sentences": len(fold["train_indices"]),
            "validation_sentences": len(fold["val_indices"]),
            "train_indices": fold["train_indices"], "validation_indices": fold["val_indices"],
            "duration_seconds": time.monotonic() - started,
            "zero_f1": validation["micro_f1"] == 0, **fit,
        }
        (output / "first_epoch_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        summaries.append({key: result[key] for key in
                          ("language", "fold", "head_version", "threshold", "duration_seconds", "zero_f1")})
        summaries[-1].update({"validation_micro_f1": validation["micro_f1"],
                              "validation_macro_f1": validation["macro_f1"],
                              "predicted_positive_targets": validation["predicted_positive_targets"],
                              "true_positive_targets": validation["true_positive_targets"],
                              "metrics_path": str(output / "first_epoch_metrics.json")})
        (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print("RESULT " + json.dumps(summaries[-1]), flush=True)
        del model, bundle, fit
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    lines = ["# E9 v2: first-epoch diagnostic", "",
             "Real pretrained backbone and complete fold-train/validation; test is not evaluated.",
             "Only one fold per language, not a completed CV run or an estimate of final quality.", "",
             "| Language | Fold | Threshold | Micro F1 | Macro F1 | Predicted positives | True positives |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['language']} | {row['fold']} | {row['threshold']} | "
                     f"{row['validation_micro_f1']:.6f} | {row['validation_macro_f1']:.6f} | "
                     f"{row['predicted_positive_targets']} | {row['true_positive_targets']} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    if any(row["zero_f1"] for row in summaries):
        parser.exit(1, "At least one diagnostic fold still has zero F1; see summary.json.\n")


if __name__ == "__main__":
    main()
