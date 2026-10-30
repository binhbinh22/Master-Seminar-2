"""E9: roberta_label_gnn, using shared processed data."""

from pathlib import Path

from experiments.dl.roberta_experiment import experiment_main, run as run_experiment

DEFAULT_CONFIG = "configs/exp9_roberta_label_gnn.yaml"

def run(config_path=Path(DEFAULT_CONFIG), **kwargs):
    return run_experiment(Path(config_path), **kwargs)

def main():
    experiment_main(DEFAULT_CONFIG)

if __name__ == "__main__":
    main()
