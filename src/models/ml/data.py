"""Load the shared two-stage dataset for classical ML, reusing the same validated
loader the RoBERTa/CodeBERT pipeline uses (`src.data.two_stage`). This guarantees ML
and DL read identical rows, folds and fingerprints for a given language/stage -- the
precondition for any meaningful cross-method comparison.
"""
from pathlib import Path

import numpy as np

from src.data.two_stage import EVALUATION_NOTE, load_stage_data

ROOT = Path(__file__).resolve().parents[3]


def load(language, stage, data_dir=None, *, store=None):
    root = Path(data_dir) if data_dir is not None else ROOT / "data" / (
        "k_fold_data" if stage == "kfold" else "full_training_data"
    )
    bundle = load_stage_data(root, language, stage, store=store)
    return {
        "bundle": bundle,
        "texts": np.array(bundle.texts, dtype=object),
        "labels": bundle.labels.numpy().astype(int),
        "label_names": list(bundle.label_names),
        "sentence_ids": list(bundle.sentence_ids),
        "class_names": list(bundle.class_names),
    }


__all__ = ["load", "EVALUATION_NOTE"]
