"""List registered experiments or train one with the shared processed data."""

import argparse
from pathlib import Path

from src.models.dl.zoo import MODEL_ZOO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List architectures and default configs")
    train = subparsers.add_parser("train", help="Train one registered experiment")
    train.add_argument("--exp", type=str.upper, choices=MODEL_ZOO, required=True)
    train.add_argument("--language", choices=("python", "java", "pharo", "all"), default="python")
    train.add_argument("--device", default=None)
    train.add_argument("--config", type=Path)
    train.add_argument("--data-dir", type=Path)
    train.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.command == "list":
        for key, spec in MODEL_ZOO.items():
            print(f"{key:3} {spec.name}\n    {spec.config}")
        return
    import yaml
    from experiments.dl.roberta_experiment import run

    spec = MODEL_ZOO[args.exp]
    config = args.config or Path(spec.config)
    if yaml.safe_load(config.read_text())["variant"] != spec.variant:
        parser.error(f"Config variant must match {args.exp}: {spec.variant}")
    languages = ("python", "java", "pharo") if args.language == "all" else (args.language,)
    for language in languages:
        output = args.output_dir
        if output is not None and args.language == "all":
            output = output / language
        run(config, language=language, device=args.device, data_dir=args.data_dir, output_dir=output)


if __name__ == "__main__":
    main()
