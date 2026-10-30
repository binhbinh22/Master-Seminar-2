"""Explicit CPU/CUDA selection with early validation."""

import re

import torch


def resolve_device(requested="auto"):
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return torch.device("cpu")
    if not isinstance(requested, str) or not re.fullmatch(r"cuda(?::[0-9]+)?", requested):
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested {requested}, but CUDA is unavailable in PyTorch {torch.__version__}. "
            "Use a Python environment with CUDA-enabled PyTorch, or select --device cpu."
        )
    index = int(requested.split(":")[1]) if ":" in requested else 0
    count = torch.cuda.device_count()
    if index >= count:
        raise ValueError(f"Requested cuda:{index}, but only {count} CUDA device(s) are visible")
    return torch.device("cuda", index)
