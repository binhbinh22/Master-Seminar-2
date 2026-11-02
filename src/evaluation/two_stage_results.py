"""Select methods using stage-1 CV only, or report stage-2 validation/test scores."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from src.models.dl.zoo import MODEL_ZOO

NOTE = "Stage-1 CV used all data, including future stage-2 test rows; final test is exploratory, not independent of method selection."


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_results(roots, stage):
    paths = set()
    for root in map(Path, roots):
        if not root.exists():
            raise ValueError(f"Results do not exist: {root}")
        paths.update([root.resolve()] if root.is_file() else (p.resolve() for p in root.rglob("metrics.json")))
    results = []
    for path in sorted(paths):
        data = json.loads(path.read_text())
        if data.get("stage") == stage and data.get("complete") is True:
            results.append((path, data))
    if not results:
        raise ValueError(f"No completed {stage} stage results found")
    return results


def select_methods(roots, output_dir, top_k=4):
    if type(top_k) is not int or not 1 <= top_k <= len(MODEL_ZOO):
        raise ValueError("Invalid top-k")
    groups = defaultdict(lambda: defaultdict(dict))
    cohorts, configs = {}, {}
    for path, data in read_results(roots, "kfold"):
        language, model, seed = data["language"], data["model"], data["seed"]
        if model not in MODEL_ZOO or data["variant"] != MODEL_ZOO[model].variant:
            raise ValueError(f"Unregistered model/variant in {path}")
        if data.get("protocol") != "kfold_all_data":
            raise ValueError(f"Not an all-data screening protocol: {path}")
        cohort = (data["data_fingerprint"], data["population_fingerprint"], tuple(data["labels"]), data["fold_count"])
        if cohorts.setdefault(language, cohort) != cohort:
            raise ValueError(f"Mixed data/folds for {language}; compare cohorts separately")
        config_key = language, model
        if configs.setdefault(config_key, data["configuration_fingerprint"]) != data["configuration_fingerprint"]:
            raise ValueError(f"Mixed hyperparameters for {model}/{language}; select one configuration per experiment")
        if seed in groups[language][model]:
            raise ValueError(f"Duplicate run for {model}/{language}/seed {seed}; pass explicit result roots")
        score = data["cross_validation"]["micro_f1"]["mean"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"Invalid CV score in {path}")
        if data["fold_count"] < 2 or len(data.get("folds", [])) != data["fold_count"]:
            raise ValueError(f"Incomplete folds in {path}")
        groups[language][model][seed] = path, data
    rows, selected = [], {}
    for language, models in sorted(groups.items()):
        if len(models) < top_k:
            raise ValueError(f"Only {len(models)} methods for {language}, fewer than top-k={top_k}")
        seed_sets = {tuple(sorted(runs)) for runs in models.values()}
        if len(seed_sets) != 1:
            raise ValueError(f"Methods for {language} must have the same completed base seeds")
        candidates = []
        for model, runs in models.items():
            ordered = [runs[seed] for seed in sorted(runs)]
            values = [data["cross_validation"]["micro_f1"]["mean"] for _, data in ordered]
            data = ordered[0][1]
            candidates.append({"model": model, "variant": data["variant"], "language": language,
                               "cv_micro_f1_mean": statistics.mean(values),
                               "cv_micro_f1_seed_std": statistics.stdev(values) if len(values) > 1 else None,
                               "mean_fold_std": statistics.mean(d["cross_validation"]["micro_f1"]["std"] for _, d in ordered),
                               "seeds": sorted(runs), "population_fingerprint": data["population_fingerprint"],
                               "data_fingerprint": data["data_fingerprint"],
                               "configuration_fingerprint": data["configuration_fingerprint"],
                               "config_path": str(ordered[0][0].parent / "config.yaml"),
                               "metrics_paths": [str(path) for path, _ in ordered]})
        candidates.sort(key=lambda row: (-row["cv_micro_f1_mean"], int(row["model"][1:])))
        selected[language] = candidates[:top_k]
        for rank, candidate in enumerate(candidates, 1):
            rows.append({"language": language, "rank": rank, "model": candidate["model"],
                         "cv_micro_f1_mean": candidate["cv_micro_f1_mean"],
                         "cv_micro_f1_seed_std": candidate["cv_micro_f1_seed_std"],
                         "mean_fold_std": candidate["mean_fold_std"],
                         "seeds": ",".join(map(str, candidate["seeds"])), "selected": rank <= top_k})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"stage": "kfold_selection", "selection_metric": "cv_micro_f1_mean", "top_k_per_language": top_k,
                "evaluation_note": NOTE, "selected": selected}
    (output / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_csv(output / "leaderboard.csv", rows)
    lines = ["# Stage 1: method selection", "", NOTE, "", "Ranked only by mean CV micro F1 across the same base seeds.",
             "Fold SD and between-seed SD are distinct, neither is a confidence interval.",
             "E4 uses threshold 0.5 during screening; threshold tuning is deferred to stage 2.", "",
             "| Language | Rank | Model | CV micro F1 | Mean fold SD | Seeds | Selected |", "|---|---:|---|---:|---:|---|---|"]
    lines.extend(f"| {r['language']} | {r['rank']} | {r['model']} | {r['cv_micro_f1_mean']:.4f} | {r['mean_fold_std']:.4f} | {r['seeds']} | {r['selected']} |" for r in rows)
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return manifest


def report_final(roots, output_dir):
    rows = []
    for path, data in read_results(roots, "final"):
        val, test = data["validation"], data.get("test", {})
        rows.append({"language": data["language"], "model": data["model"], "seed": data["seed"],
                     "data_fingerprint": data["data_fingerprint"],
                     "validation_micro_f1": val["micro_f1"], "validation_macro_f1": val["macro_f1"],
                     "test_micro_f1": test.get("micro_f1"), "test_macro_f1": test.get("macro_f1"),
                     "test_hamming_loss": test.get("hamming_loss"),
                     "best_epoch": data["training"]["best_epoch"], "metrics_path": str(path)})
    rows.sort(key=lambda row: (row["language"], row["data_fingerprint"], -row["validation_micro_f1"]))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "leaderboard.csv", rows)
    fmt = lambda value: "—" if value is None else f"{value:.4f}"
    lines = ["# Stage 2: final training results", "", NOTE, "",
             "Rows are ordered by validation micro F1 within each language/data fingerprint. Test is reporting only.",
             "The saved checkpoint is selected on validation at the baseline threshold; E4 subsequently tunes thresholds on validation.", "",
             "| Language | Data fingerprint | Model | Seed | Best epoch | Val micro F1 | Test micro F1 | Test macro F1 |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['language']} | {row['data_fingerprint'][:12]} | {row['model']} | {row['seed']} | {row['best_epoch']} | {fmt(row['validation_micro_f1'])} | {fmt(row['test_micro_f1'])} | {fmt(row['test_macro_f1'])} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return rows


def selected_candidate(path, language, model):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("stage") != "kfold_selection":
        raise ValueError("Expected a stage-1 selection manifest")
    entries = manifest.get("selected", {}).get(language, [])
    if not entries:
        raise ValueError(f"No selected candidates for {language}")
    if model is None:
        return ",".join(entry["model"][1:] for entry in entries)
    entry = next((entry for entry in entries if entry["model"] == model), None)
    if entry is None:
        raise ValueError(f"{model} is not selected for {language}")
    return entry["config_path"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("select", "report"):
        command = sub.add_parser(name)
        command.add_argument("--results", nargs="+", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        if name == "select":
            command.add_argument("--top-k", type=int, default=4)
    candidate = sub.add_parser("candidate")
    candidate.add_argument("--selection", type=Path, required=True)
    candidate.add_argument("--language", choices=("python", "java", "pharo"), required=True)
    candidate.add_argument("--model", type=str.upper)
    args = parser.parse_args()
    try:
        if args.command == "select":
            select_methods(args.results, args.output_dir, args.top_k)
            print(f"Selection: {args.output_dir / 'selection.json'}")
        elif args.command == "report":
            report_final(args.results, args.output_dir)
            print(f"Report: {args.output_dir / 'report.md'}")
        else:
            print(selected_candidate(args.selection, args.language, args.model))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
