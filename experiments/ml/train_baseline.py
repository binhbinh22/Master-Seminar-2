"""CLI entrypoint for the classical ML baselines, matching the team's two-stage protocol.

    python -m experiments.ml.train_baseline --stage kfold --language python
    python -m experiments.ml.train_baseline --stage final --language all --methods ovr_logreg ovr_random_forest

Default --stage is kfold, default --methods is all six, default --language is all three.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from experiments.ml.staged_experiment import run_staged
from src.models.ml.models import MODEL_KEYS

LANGUAGES = ("python", "java", "pharo")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("kfold", "final"), default="kfold")
    parser.add_argument("--language", choices=(*LANGUAGES, "all"), default="all")
    parser.add_argument("--methods", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Base dir for results/training/<stage>/<timestamp> if unset")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--storage", choices=("file", "memory"), default="file", help="Staged dataset persistence backend")
    args = parser.parse_args()

    languages = LANGUAGES if args.language == "all" else (args.language,)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = args.output_dir or Path("results/training") / args.stage / timestamp

    for method in args.methods:
        for language in languages:
            print(f"=== {method} / {language} / {args.stage} ===")
            result = run_staged(method, language, args.stage, output_base=base_dir,
                                seed=args.seed, data_dir=args.data_dir, storage_backend=args.storage)
            if args.stage == "kfold":
                cv = result["cross_validation"]["micro_f1"]
                print(f"CV micro-F1: {cv['mean']:.4f} +/- {cv['std']:.4f}")
            else:
                print(f"Test micro-F1: {result['test']['micro_f1']:.4f}  "
                      f"macro-F1: {result['test']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
