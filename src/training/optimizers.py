"""AdamW groups for discriminative and layer-wise CodeBERT fine-tuning."""

import math

import torch


def positive_rate(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def build_optimizer(model, training):
    """Preserve the original single-group AdamW unless LR grouping is requested.

    For L layers, layer i gets encoder_lr * decay**(L-1-i); embeddings get
    encoder_lr * decay**L. The last encoder layer uses encoder_lr and the
    classification head uses head_lr. Weight decay matches the original runs,
    including biases and LayerNorm, to isolate the learning-rate changes.
    """
    base_lr = positive_rate(training["learning_rate"], "learning_rate")
    options = ("encoder_learning_rate", "head_learning_rate", "layerwise_lr_decay")
    if not any(key in training for key in options):
        return torch.optim.AdamW(model.parameters(), lr=base_lr,
                                 weight_decay=training["weight_decay"])
    encoder_lr = positive_rate(training.get("encoder_learning_rate", base_lr), "encoder_learning_rate")
    head_lr = positive_rate(training.get("head_learning_rate", base_lr), "head_learning_rate")
    decay = positive_rate(training.get("layerwise_lr_decay", 1.), "layerwise_lr_decay")
    if decay > 1:
        raise ValueError("layerwise_lr_decay must be in (0, 1]")
    backbone = getattr(model, "roberta", None)
    if backbone is None or not hasattr(backbone, "encoder") or not hasattr(backbone.encoder, "layer"):
        raise ValueError("Layer-wise learning rates require a RoBERTa/CodeBERT encoder")
    layer_count = len(backbone.encoder.layer)
    if not layer_count:
        raise ValueError("Expected at least one transformer layer")
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("roberta.encoder.layer."):
            layer = int(name.split(".")[3])
            group = f"encoder_layer_{layer}"
            rate = encoder_lr * decay ** (layer_count - 1 - layer)
        elif name.startswith("roberta.embeddings."):
            group, rate = "embeddings", encoder_lr * decay ** layer_count
        elif name.startswith("roberta."):
            group, rate = "encoder_other", encoder_lr
        else:
            group, rate = "head", head_lr
        if group not in groups:
            groups[group] = {"group_name": group, "params": [], "lr": rate,
                             "weight_decay": training["weight_decay"]}
        groups[group]["params"].append(parameter)
    if not groups:
        raise ValueError("Model has no trainable parameters")
    return torch.optim.AdamW(list(groups.values()), lr=base_lr)


def optimizer_summary(optimizer):
    """Capture actual initial group rates before the warmup scheduler changes them."""
    return {"name": "AdamW", "groups": [
        {"name": group.get("group_name", "all"), "initial_learning_rate": group["lr"],
         "weight_decay": group["weight_decay"],
         "parameter_count": sum(p.numel() for p in group["params"])}
        for group in optimizer.param_groups
    ]}
