"""Single registry of E1–E12 architectures and their configs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    variant: str
    config: str


MODEL_ZOO = {
    "E1": ModelSpec("RoBERTa + BCE", "bce", "configs/exp1_roberta_baseline.yaml"),
    "E2": ModelSpec("RoBERTa + Weighted BCE", "weighted_bce", "configs/exp2_roberta_weighted_bce.yaml"),
    "E3": ModelSpec("RoBERTa + Focal Loss", "focal", "configs/exp3_roberta_focal.yaml"),
    "E4": ModelSpec("RoBERTa + Threshold Tuning", "threshold_tuning", "configs/exp4_roberta_threshold_tuning.yaml"),
    "E5": ModelSpec("RoBERTa + Label-Aware Attention", "label_attention", "configs/exp5_roberta_label_attention.yaml"),
    "E6": ModelSpec("RoBERTa + Label-Aware Attention + Contrastive Loss", "label_attention_contrastive",
                    "configs/exp6_roberta_label_attention_contrastive.yaml"),
    "E7": ModelSpec("RoBERTa + Asymmetric Loss", "asl", "configs/exp7_roberta_asl.yaml"),
    "E8": ModelSpec("RoBERTa + R-Drop", "rdrop", "configs/exp8_roberta_rdrop.yaml"),
    "E9": ModelSpec("RoBERTa + Label Correlation (per-language GNN)", "label_gnn", "configs/exp9_roberta_label_gnn.yaml"),
    "E10": ModelSpec("RoBERTa + Adversarial Training (FGM)", "fgm", "configs/exp10_roberta_fgm.yaml"),
    "E11": ModelSpec("CodeBERT + BCE", "codebert_bce", "configs/exp11_codebert_bce.yaml"),
    "E12": ModelSpec("RoBERTa + Label-Aware Attention + Label Correlation", "label_attention_gnn",
                     "configs/exp12_roberta_label_attention_gnn.yaml"),
}


def build_model(config, bundle, train_indices):
    # Keep listing the registry independent of torch/CUDA/model downloads.
    from src.models.dl.roberta_baseline import RobertaBaseline
    from src.models.dl.roberta_weighted_bce import RobertaWeightedBCE, compute_pos_weight
    from src.models.dl.roberta_variants import RobertaFocal, RobertaLabelAttention
    from src.models.dl.advanced_variants import (
        RobertaASL, RobertaRDrop, RobertaLabelGNN, RobertaAttentionGNN, RobertaFGM,
        compute_label_adjacency,
    )

    model = config["model"]
    kwargs = dict(model_name=model["name"], num_labels=len(bundle.label_names),
                  threshold=model["threshold"])
    variant = config["variant"]
    if variant in ("bce", "threshold_tuning", "codebert_bce"):
        return RobertaBaseline(**kwargs)
    if variant == "weighted_bce":
        weights = compute_pos_weight(bundle.labels[train_indices], bundle.train_mask[train_indices])
        return RobertaWeightedBCE(pos_weight=weights, **kwargs)
    if variant == "focal":
        return RobertaFocal(gamma=model["focal_gamma"], alpha=model.get("focal_alpha"), **kwargs)
    if variant in ("label_attention", "label_attention_contrastive"):
        return RobertaLabelAttention(
            contrastive_weight=model.get("contrastive_weight", 0.0) if variant == "label_attention_contrastive" else 0.0,
            temperature=model.get("temperature", 0.1), **kwargs)
    if variant == "asl":
        return RobertaASL(gamma_pos=model["asl_gamma_pos"], gamma_neg=model["asl_gamma_neg"],
                          clip=model["asl_clip"], detach_focal=model.get("asl_detach_focal", True), **kwargs)
    if variant == "rdrop":
        return RobertaRDrop(rdrop_alpha=model["rdrop_alpha"], **kwargs)
    if variant in ("label_gnn", "label_attention_gnn"):
        if variant == "label_gnn":
            if model.get("graph_head", "residual_classifier_v2") != "residual_classifier_v2":
                raise ValueError("E9 requires graph_head=residual_classifier_v2; retrain legacy heads")
            if config["dataset"].get("label_space", "per_language") != "per_language":
                raise ValueError("E9 uses the label space of one processed language")
            if (not train_indices or len(set(train_indices)) != len(train_indices)
                    or any(type(i) is not int or not 0 <= i < len(bundle) for i in train_indices)):
                raise ValueError("E9 requires non-empty, unique, valid training indices")
            languages = {bundle.languages[i] for i in train_indices}
            if len(languages) != 1:
                raise ValueError("E9 must train one language at a time (Python/Java/Pharo)")
            if not bundle.train_mask[train_indices].all():
                raise ValueError("E9 graph requires fully observed labels from train.csv, never test.csv")
        # Never estimate correlation from validation/test labels.
        adjacency = compute_label_adjacency(bundle.labels[train_indices], model["graph_threshold"])
        cls = RobertaLabelGNN if variant == "label_gnn" else RobertaAttentionGNN
        return cls(adjacency=adjacency, graph_hidden=model["graph_hidden"], **kwargs)
    if variant == "fgm":
        return RobertaFGM(epsilon=model["fgm_epsilon"], adversarial_weight=model["adversarial_weight"], **kwargs)
    raise ValueError(f"Unknown variant: {variant}")
