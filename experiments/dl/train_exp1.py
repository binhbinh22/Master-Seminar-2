"""Train and evaluate Exp 1 using the official NLBSE'23 category partitions."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from src.data.nlbse2023 import (
    TokenizedNLBSEDataset,
    download_dataset,
    load_dataset,
    read_config,
)
from src.data.splits import make_balanced_folds
from src.evaluation.multilabel import SUMMARY_METRICS, evaluate_masked_multilabel
from src.evaluation.summaries import summarize_metric_runs
from src.models.dl.roberta_baseline import RobertaBaseline
from src.training.masked_classifier import fit_masked_classifier
from src.training.reproducibility import set_seed
from src.visualization.tensorboard import log_metrics
from src.visualization.training_plots import save_training_plots


def make_loader(bundle, tokenizer, max_length, split, indices, batch_size, workers, shuffle=False):
    dataset = TokenizedNLBSEDataset(bundle, tokenizer, split, max_length, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)


def train_model(config, bundle, tokenizer, device, train_indices, val_indices=None,
                epochs=None, checkpoint_path=None, seed=None, log_dir=None):
    """Build the Exp 1 model and datasets, then use the shared training loop."""
    training, model_config = config["training"], config["model"]
    set_seed(training["seed"] if seed is None else seed)
    length = config["tokenizer"]["max_length"]
    train_loader = make_loader(bundle, tokenizer, length, "train", train_indices,
                               training["batch_size"], training["num_workers"], True)
    val_loader = None if val_indices is None else make_loader(
        bundle, tokenizer, length, "train", val_indices,
        training["eval_batch_size"], training["num_workers"])
    model = RobertaBaseline(model_name=model_config["name"],
                            num_labels=model_config["num_labels"],
                            threshold=model_config["threshold"]).to(device)
    fit = fit_masked_classifier(
        model, train_loader, val_loader, device=device, training=training,
        threshold=model_config["threshold"], label_names=bundle.label_names,
        report_per_label=config["evaluation"]["report_per_label"],
        epochs=epochs, checkpoint_path=checkpoint_path, log_dir=log_dir,
    )
    return model, fit


def run(config_path: Path, download: bool = False) -> dict:
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    dataset_config = read_config(config["dataset_config"])
    if download:
        download_dataset(dataset_config)
    bundle = load_dataset(dataset_config)
    model_config, training = config["model"], config["training"]
    if model_config["num_labels"] != len(bundle.label_names):
        raise ValueError("Model and dataset label counts differ")
    if model_config["hidden_size"] != 768 or model_config["loss"] != "BCEWithLogitsLoss":
        raise ValueError("Exp 1 requires Linear(768, 19) and BCEWithLogitsLoss")
    if not 0 < model_config["threshold"] < 1:
        raise ValueError("threshold must be between 0 and 1")
    for key in ("batch_size", "eval_batch_size", "gradient_accumulation_steps",
                "epochs", "early_stopping_patience"):
        if training[key] < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 <= training["warmup_ratio"] < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    eligible = bundle.train_mask.any(dim=1).nonzero(as_tuple=True)[0].tolist()
    fold_indices = make_balanced_folds(
        eligible, [bundle.languages[i] for i in eligible],
        training["folds"], training["seed"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"], use_fast=True)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = output_dir / "tensorboard" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output_dir / "dataset_config.yaml").write_text(
        yaml.safe_dump(dataset_config, sort_keys=False), encoding="utf-8")
    all_train_indices = sorted(i for fold in fold_indices for i in fold)
    result = {"experiment": config["experiment"], "dataset_samples": len(bundle),
              "source_commit": dataset_config.get("source_commit"),
              "seed": training["seed"], "threshold": model_config["threshold"],
              "labels": list(bundle.label_names), "fold_count": len(fold_indices),
              "split_unit": "sentence", "split_source": "official_train_targets",
              "tensorboard_log_dir": str(tensorboard_dir),
              "folds": []}
    for fold_number, val_indices in enumerate(fold_indices, 1):
        val_set = set(val_indices)
        train_indices = [i for i in all_train_indices if i not in val_set]
        print(f"fold {fold_number}/{len(fold_indices)}: train={len(train_indices)}, validation={len(val_indices)}", flush=True)
        model, fit = train_model(config, bundle, tokenizer, device, train_indices,
                                 val_indices, seed=training["seed"] + fold_number,
                                 log_dir=tensorboard_dir / f"fold_{fold_number}")
        result["folds"].append({"fold": fold_number, "train_sentences": len(train_indices),
                                "validation_sentences": len(val_indices),
                                "validation_indices": val_indices, **fit})
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
    final_epochs = max(1, round(statistics.median(fold["best_epoch"] for fold in result["folds"])))
    result["final_training_epochs"] = final_epochs
    print(f"training final checkpoint for {final_epochs} epochs", flush=True)
    model, final_fit = train_model(config, bundle, tokenizer, device, all_train_indices,
                                   epochs=final_epochs, checkpoint_path=output_dir / "best_model.pt",
                                   log_dir=tensorboard_dir / "final")
    result["final_training"] = final_fit
    result["train_sentences"] = len(all_train_indices)
    if config["evaluation"]["run_test_after_training"]:
        test_loader = make_loader(bundle, tokenizer, config["tokenizer"]["max_length"],
                                  "test", None, training["eval_batch_size"], training["num_workers"])
        result["test_sentences"] = len(test_loader.dataset)
        result["test"] = evaluate_masked_multilabel(
            model, test_loader, device, model_config["threshold"],
            bundle.label_names, config["evaluation"]["report_per_label"],
        )
        with SummaryWriter(log_dir=str(tensorboard_dir / "final")) as writer:
            log_metrics(writer, "test", result["test"], final_epochs)
        print(f"test_micro_f1={result['test']['micro_f1']:.4f}", flush=True)
    histories = {f"Fold {fold['fold']}": fold["history"] for fold in result["folds"]}
    histories["Final model"] = result["final_training"]["history"]
    result["plots"] = save_training_plots(
        output_dir / "plots", histories, validation_metric="micro_f1",
        cross_validation=result["cross_validation"], test_metrics=result.get("test"),
    )
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp1_roberta_baseline.yaml"))
    parser.add_argument("--download", action="store_true", help="Download missing official CSV files")
    args = parser.parse_args()
    run(args.config, args.download)


if __name__ == "__main__":
    main()
