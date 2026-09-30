"""Train and evaluate Exp 1 using the official NLBSE'23 category partitions."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.nlbse2023 import (
    TokenizedNLBSEDataset,
    download_dataset,
    load_dataset,
    read_config,
)
from src.models.dl.roberta_baseline import RobertaBaseline


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_validation(bundle, fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Hold out whole sentences from the official training targets."""
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    indices = bundle.train_mask.any(dim=1).nonzero(as_tuple=True)[0].tolist()
    rng = random.Random(seed)
    rng.shuffle(indices)
    count = max(1, round(len(indices) * fraction))
    return indices[count:], indices[:count]


def metric_dict(
    true_positive: torch.Tensor,
    false_positive: torch.Tensor,
    false_negative: torch.Tensor,
    true_negative: torch.Tensor,
    label_names: tuple[str, ...],
    per_label: bool,
) -> dict:
    tp, fp, fn, tn = (x.to(torch.float64) for x in (true_positive, false_positive, false_negative, true_negative))
    support = tp + fn
    predicted = tp + fp
    precision = torch.where(predicted > 0, tp / predicted.clamp_min(1), 0)
    recall = torch.where(support > 0, tp / support.clamp_min(1), 0)
    f1 = torch.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn).clamp_min(1), 0)
    total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
    total = (tp + fp + fn + tn).sum()
    result = {
        "micro_precision": float(total_tp / (total_tp + total_fp).clamp_min(1)),
        "micro_recall": float(total_tp / (total_tp + total_fn).clamp_min(1)),
        "micro_f1": float(2 * total_tp / (2 * total_tp + total_fp + total_fn).clamp_min(1)),
        "macro_f1": float(f1.mean()),
        "hamming_loss": float((fp.sum() + fn.sum()) / total.clamp_min(1)),
        "evaluated_targets": int(total),
    }
    if per_label:
        result["per_label"] = {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "evaluated_targets": int(tp[i] + fp[i] + fn[i] + tn[i]),
            }
            for i, name in enumerate(label_names)
        }
    return result


@torch.no_grad()
def evaluate(model, loader, device, threshold, label_names, per_label) -> dict:
    model.eval()
    tp = torch.zeros(len(label_names), dtype=torch.int64)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)
    tn = torch.zeros_like(tp)
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        predicted = (torch.sigmoid(logits).cpu() >= threshold)
        actual = batch["labels"].bool()
        mask = batch["label_mask"].bool()
        tp += (predicted & actual & mask).sum(dim=0)
        fp += (predicted & ~actual & mask).sum(dim=0)
        fn += (~predicted & actual & mask).sum(dim=0)
        tn += (~predicted & ~actual & mask).sum(dim=0)
    return metric_dict(tp, fp, fn, tn, label_names, per_label)


def run(config_path: Path, download: bool = False) -> dict:
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    dataset_config = read_config(config["dataset_config"])
    if download:
        download_dataset(dataset_config)
    bundle = load_dataset(dataset_config)

    model_config = config["model"]
    training = config["training"]
    evaluation = config["evaluation"]
    if model_config["num_labels"] != len(bundle.label_names):
        raise ValueError("Model and dataset label counts differ")
    if model_config["hidden_size"] != 768 or model_config["loss"] != "BCEWithLogitsLoss":
        raise ValueError("Exp 1 requires Linear(768, 19) and BCEWithLogitsLoss")
    if training["batch_size"] < 1 or training["gradient_accumulation_steps"] < 1:
        raise ValueError("Batch size and gradient accumulation must be positive")
    if training["epochs"] < 1:
        raise ValueError("At least one training epoch is required")
    if training["early_stopping_patience"] < 1:
        raise ValueError("Early stopping patience must be positive")

    set_seed(training["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"], use_fast=True)
    train_indices, val_indices = split_train_validation(
        bundle, training["validation_fraction"], training["seed"]
    )
    common = {"bundle": bundle, "tokenizer": tokenizer, "max_length": config["tokenizer"]["max_length"]}
    train_data = TokenizedNLBSEDataset(**common, split="train", indices=train_indices)
    val_data = TokenizedNLBSEDataset(**common, split="train", indices=val_indices)
    test_data = TokenizedNLBSEDataset(**common, split="test")
    train_loader = DataLoader(
        train_data, batch_size=training["batch_size"], shuffle=True,
        num_workers=training["num_workers"],
    )
    eval_loader_args = {"batch_size": training["eval_batch_size"], "num_workers": training["num_workers"]}
    val_loader = DataLoader(val_data, **eval_loader_args)
    test_loader = DataLoader(test_data, **eval_loader_args)

    model = RobertaBaseline(
        model_name=model_config["name"],
        num_labels=model_config["num_labels"],
        threshold=model_config["threshold"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    steps_per_epoch = math.ceil(len(train_loader) / training["gradient_accumulation_steps"])
    total_steps = steps_per_epoch * training["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * training["warmup_ratio"]),
        num_training_steps=total_steps,
    )
    use_amp = training["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    tokenizer.save_pretrained(output_dir / "tokenizer")
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "dataset_config.yaml").write_text(
        yaml.safe_dump(dataset_config, sort_keys=False), encoding="utf-8"
    )
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, training["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for step, batch in enumerate(train_loader, 1):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                    label_mask=batch["label_mask"].to(device),
                ).loss
            loss_sum += float(loss.detach())
            accumulation = training["gradient_accumulation_steps"]
            # The last partial group gets its own correct normalization.
            group_size = min(accumulation, len(train_loader) - ((step - 1) // accumulation) * accumulation)
            scaler.scale(loss / group_size).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        metrics = evaluate(
            model, val_loader, device, model_config["threshold"], bundle.label_names,
            evaluation["report_per_label"],
        )
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_loader), "validation": metrics})
        print(f"epoch {epoch}: loss={loss_sum / len(train_loader):.4f}, val_micro_f1={metrics['micro_f1']:.4f}")
        if metrics["micro_f1"] > best_f1:
            best_f1 = metrics["micro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= training["early_stopping_patience"]:
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    result = {
        "experiment": config["experiment"],
        "dataset_samples": len(bundle),
        "source_commit": dataset_config.get("source_commit"),
        "seed": training["seed"],
        "threshold": model_config["threshold"],
        "labels": list(bundle.label_names),
        "train_sentences": len(train_data),
        "validation_sentences": len(val_data),
        "test_sentences": len(test_data),
        "best_epoch": best_epoch,
        "validation": history[best_epoch - 1]["validation"],
        "history": history,
    }
    if evaluation["run_test_after_training"]:
        result["test"] = evaluate(
            model, test_loader, device, model_config["threshold"], bundle.label_names,
            evaluation["report_per_label"],
        )
        print(f"test_micro_f1={result['test']['micro_f1']:.4f}")
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
