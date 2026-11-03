"""Export verified stage-1 CV results to one JPG and a detailed CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.evaluation.two_stage_results import read_results, select_methods


NAMES = {
    "E1": "RoBERTa · BCE", "E2": "RoBERTa · Weighted BCE",
    "E3": "RoBERTa · Focal", "E4": "RoBERTa · Threshold tuning*",
    "E5": "RoBERTa · Label attention", "E6": "RoBERTa · Attention + Contrastive",
    "E7": "RoBERTa · ASL", "E8": "RoBERTa · R-Drop",
    "E9": "RoBERTa · Label GNN", "E10": "RoBERTa · FGM",
    "E11": "CodeBERT · BCE", "E12": "RoBERTa · Attention + GNN",
    "E13": "CodeBERT · Head LR", "E14": "CodeBERT · LLRD",
    "E15": "CodeBERT · R-Drop", "E16": "CodeBERT · LLRD + R-Drop",
    "E17": "CodeBERT · Focal",
}


def export(roots, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    # Checks population/folds, model configurations, seed coverage and completion.
    selection = select_methods(roots, output, top_k=4)
    records = read_results(roots, "kfold")
    if any(len(entry["seeds"]) != 1 for entries in selection["selected"].values() for entry in entries):
        raise ValueError("This figure expects one common base seed; use separate exports for multiple seeds")
    languages = [lang for lang in ("python", "java", "pharo") if lang in selection["selected"]]
    rows, groups = [], {}
    for language in languages:
        group = [(path, data) for path, data in records if data["language"] == language]
        group.sort(key=lambda item: (-item[1]["cross_validation"]["micro_f1"]["mean"], int(item[1]["model"][1:])))
        groups[language] = group
        for rank, (path, data) in enumerate(group, 1):
            if len(data["folds"]) != data["fold_count"] or any(not f.get("validation") for f in data["folds"]):
                raise ValueError(f"Missing fold validation results: {path}")
            row = {"language": language, "rank": rank, "model": data["model"],
                   "method": NAMES[data["model"]], "seed": data["seed"],
                   "fold_count": data["fold_count"], "samples": data["dataset_samples"],
                   "top_4": rank <= 4, "duration_minutes": data["duration_seconds"] / 60,
                   "data_fingerprint": data["data_fingerprint"], "metrics_path": str(path)}
            for metric, summary in data["cross_validation"].items():
                row[f"{metric}_mean"] = summary["mean"]
                row[f"{metric}_std"] = summary["std"]
            rows.append(row)
    with (output / "stage1_metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(20, 3 + 6.5 * len(languages)), facecolor="#f1f5f9")
    fig.text(.045, .975, "KẾT QUẢ GIAI ĐOẠN 1 · K-FOLD", fontsize=27,
             weight="bold", color="#0f172a", va="top")
    seeds = sorted({data["seed"] for _, data in records})
    folds = sorted({data["fold_count"] for _, data in records})
    fig.text(.045, .949, f"{len(records)} lượt hoàn tất  |  {len(NAMES)} phương pháp được hỗ trợ  |  "
             f"Số fold: {', '.join(map(str, folds))}  |  Seed: {', '.join(map(str, seeds))}",
             fontsize=13, color="#475569")
    fig.text(.045, .930, "Xếp hạng riêng từng ngôn ngữ theo CV micro F1. Tô xanh: top 4. "
             "F1 và AP hiển thị theo %; ± là SD giữa các fold.", fontsize=12, color="#475569")

    top, bottom = .910, .075
    height = (top - bottom) / len(languages)
    for index, language in enumerate(languages):
        group = groups[language]
        first = group[0][1]
        winner = first["cross_validation"]["micro_f1"]
        y_top = top - index * height
        fig.text(.045, y_top, language.upper(), fontsize=19, weight="bold", color="#0f172a", va="top")
        fig.text(.175, y_top - .002,
                 f"{first['dataset_samples']:,} câu · {len(first['labels'])} nhãn · {len(group)} phương pháp",
                 fontsize=12, color="#64748b", va="top")
        fig.text(.955, y_top - .002,
                 f"Dẫn đầu: {first['model']}  |  Micro F1 {100*winner['mean']:.2f} ± {100*winner['std']:.2f}",
                 fontsize=13, weight="bold", color="#047857", va="top", ha="right")
        ax = fig.add_axes([.045, y_top - height + .018, .91, height - .047])
        ax.axis("off")
        contents = []
        for rank, (_, data) in enumerate(group, 1):
            cv = data["cross_validation"]
            def f1(metric):
                return f"{100*cv[metric]['mean']:.2f} ± {100*cv[metric]['std']:.2f}"
            ap = cv["micro_average_precision"]["mean"]
            contents.append([str(rank), data["model"], NAMES[data["model"]],
                             f1("micro_f1"), f1("macro_f1"),
                             "—" if ap is None else f"{100*ap:.2f}",
                             f"{cv['hamming_loss']['mean']:.4f}", f"{data['duration_seconds']/60:.1f}"])
        table = ax.table(cellText=contents,
                         colLabels=["Hạng", "Mã", "Phương pháp", "Micro F1 (%) ↑", "Macro F1 (%) ↑",
                                    "Micro AP (%) ↑", "Hamming ↓", "Phút / 5 fold"],
                         colWidths=[.045, .045, .30, .15, .15, .115, .095, .10],
                         cellLoc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#e2e8f0")
            cell.set_linewidth(.45)
            if row == 0:
                cell.set_facecolor("#142d4e")
                cell.set_text_props(color="white", weight="bold", fontsize=10.5)
            else:
                cell.set_facecolor("#dcfce7" if row <= 4 else ("#ffffff" if row % 2 else "#f8fafc"))
                cell.set_text_props(color="#166534" if row <= 4 else "#334155",
                                    weight="bold" if row == 1 else "normal")
                if col == 2:
                    cell.get_text().set_ha("left")
                    cell.PAD = .025
    fig.text(.045, .050, "* E4 chấm CV tại threshold 0.5; tuning threshold hoãn sang giai đoạn 2. "
             "SD giữa fold không phải khoảng tin cậy.", fontsize=11, color="#475569")
    fig.text(.045, .033, "CV dùng 100% dữ liệu và validation chọn epoch. Test giai đoạn 2 không độc lập "
             "với lựa chọn phương pháp. Thời gian là từng lượt chạy, không phải benchmark tốc độ.",
             fontsize=10.5, color="#475569")
    fig.text(.045, .016, "Nguồn: metrics.json của các lượt kfold hoàn tất trên kfold_gpu0 và kfold_gpu1. "
             "CSV đi kèm chứa đầy đủ 9 metrics, SD và đường dẫn nguồn.", fontsize=10.5, color="#64748b")
    path = output / "stage1_results.jpg"
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), pil_kwargs={"quality": 95})
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage1_summary"))
    args = parser.parse_args()
    print(export(args.results, args.output_dir))


if __name__ == "__main__":
    main()
