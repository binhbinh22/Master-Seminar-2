"""Train and evaluate Exp 2: RoBERTa + Weighted BCE on NLBSE'23."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.dl.train_exp1 import run as run_roberta_experiment
from src.models.dl.roberta_weighted_bce import RobertaWeightedBCE, compute_pos_weight


def make_model(config, bundle, train_indices):
    """Estimate weights from this model's official training targets only."""
    weights = compute_pos_weight(bundle.labels[train_indices],
                                 bundle.train_mask[train_indices])
    model_config = config["model"]
    return RobertaWeightedBCE(pos_weight=weights, model_name=model_config["name"],
                             num_labels=model_config["num_labels"],
                             threshold=model_config["threshold"])


def run(config_path: Path, download: bool = False) -> dict:
    return run_roberta_experiment(config_path, download, model_factory=make_model,
                                 required_loss="WeightedBCEWithLogitsLoss")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/exp2_roberta_weighted_bce.yaml"))
    parser.add_argument("--download", action="store_true",
                        help="Download missing official CSV files")
    args = parser.parse_args()
    run(args.config, args.download)


if __name__ == "__main__":
    main()
