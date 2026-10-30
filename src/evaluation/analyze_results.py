"""Build a model-zoo report from completed runs, without loading checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from src.models.dl.zoo import MODEL_ZOO


METRICS = ("micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall",
           "macro_f1", "micro_average_precision", "macro_average_precision", "hamming_loss")


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def formatted(value):
    return f"{value:.4f}" if numeric(value) else "—"


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_results(roots):
    files = set()
    for root in map(Path, roots):
        if not root.exists():
            raise ValueError(f"Result path does not exist: {root}")
        files.update([root.resolve()] if root.is_file() else (p.resolve() for p in root.rglob("metrics.json")))
    records, skipped = [], []
    variants = {spec.variant: key for key, spec in MODEL_ZOO.items()}
    for path in sorted(files):
        try:
            data = json.loads(path.read_text())
            if (not isinstance(data, dict) or not isinstance(data.get("cross_validation"), dict)
                    or not isinstance(data.get("labels"), list) or not data["labels"]
                    or data.get("language") not in ("python", "java", "pharo")
                    or not isinstance(data.get("experiment"), str)):
                raise ValueError("Not a completed per-language experiment result")
            row = {
                "run": f"R{len(records) + 1:03}",
                "model": variants.get(data.get("variant"), data["experiment"]),
                "experiment": data["experiment"], "language": data["language"],
                "data_fingerprint": data.get("data_fingerprint") or "unverified",
                "split_source": data.get("split_source", "unknown"),
                "seed": data.get("seed"), "device": data.get("device", "unknown"),
                "fold_count": data.get("fold_count"),
                "final_epochs": data.get("final_training_epochs"),
                "duration_seconds": data.get("duration_seconds"),
                "metrics_path": str(path), "checkpoint": str(path.parent / "best_model.pt"),
                "checkpoint_exists": (path.parent / "best_model.pt").is_file(),
                "thresholds_path": str(path.parent / "thresholds.json"),
            }
            for metric in METRICS:
                cv = data["cross_validation"].get(metric) or {}
                row[f"cv_{metric}_mean"] = cv.get("mean")
                row[f"cv_{metric}_std"] = cv.get("std")
                row[f"test_{metric}"] = (data.get("test") or {}).get(metric)
            records.append((path, data, row))
        except (ValueError, TypeError, AttributeError) as error:
            skipped.append(f"{path}: {error}")
    return records, skipped


def cohort_key(record):
    _, data, row = record
    # Unverified historical results get independent cohorts: do not assume their
    # splits match just because language and sample count happen to match.
    fingerprint = row["data_fingerprint"]
    if fingerprint == "unverified":
        fingerprint = row["metrics_path"]
    return (row["language"], fingerprint, tuple(data["labels"]), row["split_source"])


def comparison_plots(records, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [f"{row['model']} {row['run']}" for _, _, row in records]
    positions = np.arange(len(records))
    fig, axes = plt.subplots(1, 2, figsize=(max(11, len(records) * 1.5), 5))
    for index, (source, title) in enumerate((("cv", "CV at baseline threshold (mean ± fold SD)"),
                                           ("test", "Held-out test at saved thresholds"))):
        for offset, metric in enumerate(("micro_f1", "macro_f1")):
            values, errors = [], []
            for _, _, row in records:
                value = row[f"cv_{metric}_mean"] if source == "cv" else row[f"test_{metric}"]
                values.append(value if numeric(value) else float("nan"))
                error = row.get(f"cv_{metric}_std")
                errors.append(error if numeric(error) else 0.0)
            axes[index].bar(positions + (offset - .5) * .35, values, .35,
                            yerr=errors if source == "cv" else None, capsize=3, label=metric)
        axes[index].set_xticks(positions, names, rotation=35, ha="right")
        axes[index].set(title=title, ylabel="F1", ylim=(0, 1.05))
        axes[index].grid(axis="y", alpha=.2)
        axes[index].legend()
    fig.tight_layout()
    path = directory / "comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths = [path]

    labels = records[0][1]["labels"]
    values = [[(data.get("test") or {}).get("per_label", {}).get(label, {}).get("f1")
               for label in labels] for _, data, _ in records]
    matrix = np.array([[v if numeric(v) else np.nan for v in row] for row in values])
    if np.isfinite(matrix).any():
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), max(3, len(records) * .6)))
        heat = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_title("Test F1 by label")
        for i in range(len(names)):
            for j in range(len(labels)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                            color="white" if matrix[i, j] > .65 else "black")
        fig.colorbar(heat, ax=ax)
        fig.tight_layout()
        path = directory / "per_label_f1.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def analyze_results(roots, output_dir):
    records, skipped = collect_results(roots)
    if not records:
        detail = "\n".join(skipped)
        raise ValueError(f"No completed per-language metrics.json results found.\n{detail}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "leaderboard.csv", [row for _, _, row in records])
    per_label = []
    for _, data, row in records:
        for label in data["labels"]:
            metrics = (data.get("test") or {}).get("per_label", {}).get(label, {})
            per_label.append({"run": row["run"], "model": row["model"], "language": row["language"],
                              "label": label, **{key: metrics.get(key) for key in
                              ("precision", "recall", "f1", "average_precision", "support")}})
    write_csv(output / "per_label.csv", per_label)
    lines = ["# Model Zoo results", "", "## Registered models", "",
             "| Model | Architecture | Completed runs |", "|---|---|---:|"]
    for key, spec in MODEL_ZOO.items():
        count = sum(row["model"] == key for _, _, row in records)
        lines.append(f"| {key} | {spec.name} | {count} |")
    lines += ["", "Rows below are sorted by CV micro F1; test metrics are held-out reporting only.",
              "CV ± SD describes variation across folds, not a confidence interval.",
              "E4 CV scores use the baseline threshold; test scores use OOF-tuned thresholds.",
              "Runs are compared only within the same language, data fingerprint, label order and split protocol.", ""]
    cohorts = defaultdict(list)
    for record in records:
        cohorts[cohort_key(record)].append(record)
    for key, group in sorted(cohorts.items()):
        language = key[0]
        group.sort(key=lambda record: -(record[2]["cv_micro_f1_mean"]
                                        if numeric(record[2]["cv_micro_f1_mean"]) else -math.inf))
        tag = hashlib.sha256(repr(key).encode()).hexdigest()[:12]
        directory = output / f"{language}-{tag}"
        directory.mkdir(exist_ok=True)
        lines += [f"## {language} / {tag}", "",
                  f"Data fingerprint: `{group[0][2]['data_fingerprint']}`. Split: `{key[3]}`.", ""]
        if group[0][2]["data_fingerprint"] == "unverified":
            lines += ["Historical run without a data fingerprint; kept separate from other runs.", ""]
        lines += ["| Run | Model | CV micro F1 ± SD | Test micro F1 | Test macro F1 | Test macro AP | Hamming loss ↓ |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for _, _, row in group:
            lines.append(f"| {row['run']} | {row['model']} | {formatted(row['cv_micro_f1_mean'])} ± "
                         f"{formatted(row['cv_micro_f1_std'])} | {formatted(row['test_micro_f1'])} | "
                         f"{formatted(row['test_macro_f1'])} | {formatted(row['test_macro_average_precision'])} | "
                         f"{formatted(row['test_hamming_loss'])} |")
        lines.append("")
        for image in comparison_plots(group, directory):
            lines += [f"![{image.stem}]({image.relative_to(output).as_posix()})", ""]
        for path, data, row in group:
            relative = Path(os.path.relpath(path, output)).as_posix()
            lines += [f"- {row['run']}: [{data['experiment']}]({relative}); seed {row['seed']}; "
                      f"device `{row['device']}`; epochs {row['final_epochs']}; "
                      f"checkpoint {'present' if row['checkpoint_exists'] else 'missing'}."]
        lines.append("")
    if skipped:
        lines += ["## Skipped files", ""] + [f"- {reason}" for reason in skipped] + [""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"runs": len(records), "cohorts": len(cohorts), "skipped": skipped, "report": str(output / "report.md")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, default=[Path("results")],
                        help="One or more run roots or metrics.json files")
    parser.add_argument("--output-dir", type=Path, default=Path("results/model_zoo"))
    args = parser.parse_args()
    try:
        summary = analyze_results(args.results, args.output_dir)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(f"Analyzed {summary['runs']} runs in {summary['cohorts']} comparable groups.")
    print(f"Report: {summary['report']}")
    for reason in summary["skipped"]:
        print(f"Skipped: {reason}")


if __name__ == "__main__":
    main()
