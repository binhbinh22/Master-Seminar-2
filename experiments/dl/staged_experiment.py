"""Separate all-data screening CV from fixed train/validation/test training."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from experiments.dl.roberta_experiment import make_loader, train_model
from src.data.two_stage import EVALUATION_NOTE, load_stage_data
from src.data.storage import dataset_store
from src.evaluation.multilabel import SUMMARY_METRICS, evaluate_masked_multilabel
from src.evaluation.summaries import summarize_metric_runs
from src.evaluation.thresholds import predict_probabilities, tune_thresholds
from src.models.dl.zoo import MODEL_ZOO
from src.training.devices import resolve_device
from src.visualization.training_plots import save_training_plots

ROOT = Path(__file__).resolve().parents[2]


def configuration_fingerprint(config):
    training = {k: v for k, v in config["training"].items()
                if k not in ("seed", "device", "folds", "progress")}
    payload = {"variant": config["variant"], "model": {k: v for k, v in config["model"].items() if k != "num_labels"},
               "tokenizer": config["tokenizer"], "training": training,
               "threshold_tuning": config.get("threshold_tuning")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def verify_selection(path, model_key, bundle, config_fingerprint):
    selection = json.loads(Path(path).read_text())
    if selection.get("stage") != "kfold_selection" or selection.get("selection_metric") != "cv_micro_f1_mean":
        raise ValueError("Expected a stage-1 CV selection manifest")
    candidates = selection.get("selected", {}).get(bundle.languages[0], [])
    candidate = next((entry for entry in candidates if entry["model"] == model_key), None)
    if candidate is None:
        raise ValueError(f"{model_key} was not selected for {bundle.languages[0]}")
    if candidate["population_fingerprint"] != bundle.population_fingerprint:
        raise ValueError("Selection and final data do not describe the same raw population")
    if candidate["configuration_fingerprint"] != config_fingerprint:
        raise ValueError("Final configuration differs from the selected CV configuration")
    return candidate


def run_staged(config_path, *, language=None, data_dir=None, output_dir=None,
               device=None, seed=None, stage="kfold", selection=None, storage_backend="file"):
    started = time.monotonic()
    if stage not in ("kfold", "final"):
        raise ValueError("stage must be kfold or final")
    if selection is not None and stage != "final":
        raise ValueError("--selection is only valid for the final stage")
    config = copy.deepcopy(yaml.safe_load(Path(config_path).read_text()))
    if seed is not None:
        config["training"]["seed"] = seed
    training, model_config = config["training"], config["model"]
    if type(training["seed"]) is not int or not 0 <= training["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for key in ("batch_size", "eval_batch_size", "gradient_accumulation_steps", "epochs", "early_stopping_patience"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if model_config["hidden_size"] != 768 or not 0 < model_config["threshold"] < 1:
        raise ValueError("Require hidden_size 768 and threshold in (0, 1)")
    if not 0 <= training["warmup_ratio"] < 1 or config["tokenizer"]["max_length"] < 1:
        raise ValueError("Invalid warmup ratio or tokenizer max length")
    language = language or config["dataset"]["language"]
    if language not in ("python", "java", "pharo"):
        raise ValueError("Unknown language")
    root = Path(data_dir) if data_dir is not None else ROOT / "data" / ("k_fold_data" if stage == "kfold" else "full_training_data")
    store = dataset_store(storage_backend, root / language)
    bundle = load_stage_data(root, language, stage, store=store)
    config_fingerprint = configuration_fingerprint(config)
    model_key = next((key for key, spec in MODEL_ZOO.items() if spec.variant == config["variant"]), None)
    if model_key is None:
        raise ValueError("Unregistered experiment variant")
    selected = verify_selection(selection, model_key, bundle, config_fingerprint) if selection is not None else None
    device = resolve_device(device if device is not None else training.get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config["dataset"].update(data_dir=str(root), language=language)
    training["device"] = str(device)
    training["folds"] = len(bundle.folds)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = Path(output_dir) if output_dir is not None else ROOT / "results/training" / stage / stamp / model_key.lower().replace("e", "exp", 1) / language
    if (output / "metrics.json").exists() or (output / "best_model.pt").exists():
        raise ValueError(f"Refusing to overwrite completed staged results at {output}; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    config.update(stage=stage, output_dir=str(output), storage_backend=storage_backend)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "label_cols.json").write_text(json.dumps(list(bundle.label_names), indent=2) + "\n")
    (output / "data_manifest.json").write_bytes(store.read_bytes(Path(bundle.data_dir) / "manifest.json"))
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"], use_fast=True)
    tokenizer.save_pretrained(output / "tokenizer")
    tensorboard = output / "tensorboard" / stamp
    result = {"stage": stage, "protocol": bundle.protocol, "split_source": bundle.protocol, "storage_backend": storage_backend,
              "model": model_key, "experiment": config["experiment"], "variant": config["variant"],
              "pretrained_backbone": model_config["name"], "language": language,
              "labels": list(bundle.label_names), "seed": training["seed"], "device": str(device),
              "data_dir": bundle.data_dir, "dataset_samples": len(bundle),
              "data_fingerprint": bundle.data_fingerprint, "population_fingerprint": bundle.population_fingerprint,
              "configuration_fingerprint": config_fingerprint, "fold_count": len(bundle.folds),
              "threshold": model_config["threshold"], "evaluation_note": EVALUATION_NOTE,
              "test_independent_of_method_selection": False,
              "tensorboard_log_dir": str(tensorboard), "split_unit": "normalized_text_group"}
    histories = {}
    if stage == "kfold":
        result["folds"] = []
        (output / "folds.json").write_text(json.dumps(bundle.folds) + "\n")
        for fold in bundle.folds:
            number = fold["fold"]
            print(f"{model_key} {language}: screening fold {number+1}/{len(bundle.folds)}", flush=True)
            model, fit = train_model(config, bundle, tokenizer, device, fold["train_indices"],
                                     fold["val_indices"], seed=training["seed"]+number,
                                     log_dir=tensorboard / f"fold_{number}")
            result["folds"].append({**fold, **fit})
            histories[f"Fold {number}"] = fit["history"]
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        result["cross_validation"] = summarize_metric_runs([f["validation"] for f in result["folds"]], SUMMARY_METRICS)
        result["cv_population_sentences"] = len(bundle)
        result["test_sentences"] = 0
        result["selection_metric"] = "cv_micro_f1_mean"
        if config["variant"] == "threshold_tuning":
            result["threshold_tuning"] = {"status": "deferred_to_stage2_validation", "cv_threshold": model_config["threshold"]}
    else:
        train_indices = list(range(bundle.train_size))
        model, fit = train_model(config, bundle, tokenizer, device, train_indices,
                                 bundle.validation_indices, validation_split="validation",
                                 checkpoint_path=output / "best_model.pt", log_dir=tensorboard / "final")
        result.update(training=fit, validation=fit["validation"], train_sentences=bundle.train_size,
                      validation_sentences=len(bundle.validation_indices), test_sentences=len(bundle.test_indices),
                      selected_from=str(Path(selection).resolve()) if selection is not None else None)
        if selected is not None:
            result["screening_cv_micro_f1"] = selected["cv_micro_f1_mean"]
        histories["Train 80% / validation 10%"] = fit["history"]
        if "label_graph" in fit:
            (output / "label_graph.json").write_text(json.dumps(fit["label_graph"], indent=2) + "\n")
        threshold = model_config["threshold"]
        if config["variant"] == "threshold_tuning":
            val_loader = make_loader(bundle, tokenizer, config["tokenizer"]["max_length"], "validation",
                                     bundle.validation_indices, training["eval_batch_size"], training["num_workers"])
            scores = predict_probabilities(model, val_loader, device, show_progress=training.get("progress", True))
            threshold = tune_thresholds(scores, bundle.labels[bundle.validation_indices], config["threshold_tuning"]["grid"])
            result["threshold_tuning"] = {"source": "stage2_validation", "objective": "per_label_f1",
                                          "note": "Validation used for checkpoint and threshold selection; not an independent score."}
        thresholds = threshold.tolist() if isinstance(threshold, torch.Tensor) else [threshold] * len(bundle.label_names)
        result["thresholds"] = thresholds
        (output / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
        if config["evaluation"]["run_test_after_training"]:
            loader = make_loader(bundle, tokenizer, config["tokenizer"]["max_length"], "test", bundle.test_indices,
                                 training["eval_batch_size"], training["num_workers"])
            result["test"] = evaluate_masked_multilabel(model, loader, device, threshold, bundle.label_names,
                                                      config["evaluation"]["report_per_label"],
                                                      show_progress=training.get("progress", True))
        del model
    result.update(duration_seconds=time.monotonic()-started, complete=True)
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    result["plots"] = save_training_plots(output / "plots", histories,
                                          cross_validation=result.get("cross_validation"), test_metrics=result.get("test"))
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
