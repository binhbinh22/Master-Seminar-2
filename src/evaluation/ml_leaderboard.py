"""Aggregate all classical-ML metrics.json files into two separate leaderboards --
one for stage-1 CV (screening, 100% data) and one for stage-2 held-out test -- so you
don't have to open each metrics.json by hand. Generic over method/language, unlike
src.evaluation.two_stage_results which is tied to the DL MODEL_ZOO registry.

`dummy_most_frequent` is excluded from both leaderboards: it always scores 0 (every
label here has <50% prevalence, so "most frequent" is always the negative class), so it
has no comparative value once its job -- confirming the real methods aren't just
guessing -- is done. It still runs and is visible in its own metrics.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXCLUDED_METHODS = {"dummy_most_frequent"}


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def kfold_leaderboard(roots):
    rows = []
    for root in map(Path, roots):
        for path in sorted(root.rglob("metrics.json")):
            data = json.loads(path.read_text())
            if data.get("stage") != "kfold" or not data.get("complete") or data["method"] in EXCLUDED_METHODS:
                continue
            cv = data["cross_validation"]
            rows.append({
                "language": data["language"], "method": data["method"],
                "cv_micro_f1_mean": round(cv["micro_f1"]["mean"], 4),
                "cv_micro_f1_std": round(cv["micro_f1"]["std"], 4),
                "cv_macro_f1_mean": round(cv["macro_f1"]["mean"], 4),
                "cv_hamming_mean": round(cv["hamming_loss"]["mean"], 4),
                "out_of_fold_micro_f1": round(data["out_of_fold"]["micro_f1"], 4),
                "metrics_path": str(path),
            })
    rows.sort(key=lambda r: (r["language"], -r["cv_micro_f1_mean"]))
    return rows


def final_leaderboard(roots):
    rows = []
    for root in map(Path, roots):
        for path in sorted(root.rglob("metrics.json")):
            data = json.loads(path.read_text())
            if data.get("stage") != "final" or not data.get("complete") or data["method"] in EXCLUDED_METHODS:
                continue
            val, test = data["validation"], data.get("test", {})
            rows.append({
                "language": data["language"], "method": data["method"],
                "validation_micro_f1": round(val["micro_f1"], 4),
                "validation_macro_f1": round(val["macro_f1"], 4),
                "test_micro_f1": round(test.get("micro_f1"), 4) if test.get("micro_f1") is not None else None,
                "test_macro_f1": round(test.get("macro_f1"), 4) if test.get("macro_f1") is not None else None,
                "test_hamming_loss": round(test.get("hamming_loss"), 4) if test.get("hamming_loss") is not None else None,
                "metrics_path": str(path),
            })
    rows.sort(key=lambda r: (r["language"], -(r["test_micro_f1"] or 0)))
    return rows


def markdown_table(rows, columns, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]) if row[c] is not None else "—" for c in columns) + " |")
    return "\n".join(lines)


def write_markdown(path, cv_rows, test_rows):
    cv_columns = ("language", "method", "cv_micro_f1_mean", "cv_micro_f1_std", "cv_macro_f1_mean", "cv_hamming_mean")
    cv_headers = ("Language", "Method", "CV micro-F1 (mean)", "CV micro-F1 (std)", "CV macro-F1", "CV Hamming loss")
    test_columns = ("language", "method", "validation_micro_f1", "test_micro_f1", "test_macro_f1", "test_hamming_loss")
    test_headers = ("Language", "Method", "Validation micro-F1", "Test micro-F1", "Test macro-F1", "Test Hamming loss")
    lines = [
        "# Classical ML leaderboard", "",
        "Sinh tự động bởi `python -m src.evaluation.ml_leaderboard`. "
        "Xếp hạng CV theo `cv_micro_f1_mean`; xếp hạng test theo `test_micro_f1`.", "",
        "## Giai đoạn 1 — CV (100% dữ liệu, sàng lọc phương pháp)", "",
        markdown_table(cv_rows, cv_columns, cv_headers), "",
        "## Giai đoạn 2 — Validation / Test (80/10/10, held-out)", "",
        markdown_table(test_rows, test_columns, test_headers), "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", default=["results/training"])
    parser.add_argument("--output-dir", default="results/training/ml_leaderboard")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cv_rows = kfold_leaderboard(args.results)
    test_rows = final_leaderboard(args.results)
    write_csv(output / "kfold_leaderboard.csv", cv_rows)
    write_csv(output / "final_leaderboard.csv", test_rows)
    write_markdown(output / "README.md", cv_rows, test_rows)

    print(f"{len(cv_rows)} kfold runs -> {output / 'kfold_leaderboard.csv'}")
    print(f"{len(test_rows)} final runs -> {output / 'final_leaderboard.csv'}")
    print(f"Markdown report -> {output / 'README.md'}")


if __name__ == "__main__":
    main()
