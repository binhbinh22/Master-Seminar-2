"""Reusable optimization loop for masked multi-label classifiers."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from src.evaluation.multilabel import evaluate_masked_multilabel
from src.visualization.tensorboard import log_metrics


def fit_masked_classifier(
    model,
    train_loader,
    validation_loader,
    *,
    device: torch.device,
    training: dict,
    threshold: float,
    label_names: tuple[str, ...],
    report_per_label: bool,
    validation_metric: str = "micro_f1",
    epochs: int | None = None,
    checkpoint_path: Path | None = None,
    log_dir: Path | None = None,
) -> dict:
    """Fit a model returning ``loss``/``logits`` from masked batch fields.

    Model batches must contain ``input_ids``, ``attention_mask``, ``labels`` and
    ``label_mask``. The selected validation metric is maximized. With no
    validation loader, the model trains for the requested number of epochs.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"])
    epochs = training["epochs"] if epochs is None else epochs
    steps = math.ceil(len(train_loader) / training["gradient_accumulation_steps"]) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=round(steps * training["warmup_ratio"]),
        num_training_steps=steps)
    use_amp = training["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    writer = SummaryWriter(log_dir=str(log_dir)) if log_dir is not None else None
    best_score, best_epoch, best_metrics, stale_epochs = -float("inf"), 0, None, 0
    history = []
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            accumulation = training["gradient_accumulation_steps"]
            for step, batch in enumerate(train_loader, 1):
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = model(input_ids=batch["input_ids"].to(device),
                                 attention_mask=batch["attention_mask"].to(device),
                                 labels=batch["labels"].to(device),
                                 label_mask=batch["label_mask"].to(device)).loss
                batch_loss = float(loss.detach())
                loss_sum += batch_loss
                if writer is not None:
                    writer.add_scalar("train/batch_loss", batch_loss,
                                      (epoch - 1) * len(train_loader) + step)
                group_size = min(accumulation,
                                 len(train_loader) - ((step - 1) // accumulation) * accumulation)
                scaler.scale(loss / group_size).backward()
                if step % accumulation == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    if writer is not None:
                        writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0],
                                          (epoch - 1) * len(train_loader) + step)
                    optimizer.zero_grad(set_to_none=True)
            record = {"epoch": epoch, "train_loss": loss_sum / len(train_loader)}
            if writer is not None:
                writer.add_scalar("train/epoch_loss", record["train_loss"], epoch)
            if validation_loader is not None:
                metrics = evaluate_masked_multilabel(
                    model, validation_loader, device, threshold, label_names, report_per_label)
                record["validation"] = metrics
                if writer is not None:
                    log_metrics(writer, "validation", metrics, epoch)
                score = metrics[validation_metric]
                print(f"epoch {epoch}: loss={record['train_loss']:.4f}, "
                      f"val_{validation_metric}={score:.4f}", flush=True)
                if score > best_score:
                    best_score, best_epoch, best_metrics, stale_epochs = score, epoch, metrics, 0
                else:
                    stale_epochs += 1
            else:
                print(f"final epoch {epoch}: loss={record['train_loss']:.4f}", flush=True)
            history.append(record)
            if writer is not None:
                writer.flush()
            if validation_loader is not None and stale_epochs >= training["early_stopping_patience"]:
                break
    finally:
        if writer is not None:
            writer.close()
    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)
    return {"best_epoch": best_epoch if validation_loader is not None else epochs,
            "validation": best_metrics, "history": history}
