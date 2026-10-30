"""E6: RoBERTa label_attention_contrastive, using shared processed data."""

from pathlib import Path

from experiments.dl.roberta_experiment import experiment_main, run as run_experiment

DEFAULT_CONFIG = "configs/exp6_roberta_label_attention_contrastive.yaml"

def run(config_path=Path(DEFAULT_CONFIG), **kwargs):
    return run_experiment(Path(config_path), **kwargs)

def main():
    experiment_main(DEFAULT_CONFIG)

if __name__ == "__main__":
    main()
