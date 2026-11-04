"""Validated data and restrained plotting utilities shared by the EDA notebooks.

All summaries are descriptive. Original category-wise partitions, grouped CV,
and the fixed stage-2 split are kept separate; this module never trains models.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np
import pandas as pd
import yaml

LANGUAGES = ("Python", "Java", "Pharo")
LANGUAGE_COLORS = {"Python": "#4477AA", "Java": "#228877", "Pharo": "#AA7744"}
SPLIT_COLORS = {"train": "#4477AA", "val": "#888888", "test": "#AA7744"}
LABEL_COLORS = ("#4477AA", "#228877", "#AA7744", "#777799", "#666666", "#88AABB", "#998866")
SHIFT_CMAP = LinearSegmentedColormap.from_list("signed_prevalence", ["#AA7744", "#FAFAFA", "#4477AA"])


def find_repo() -> Path:
    for directory in (Path.cwd(), *Path.cwd().parents):
        if (directory / "configs/datasets/nlbse2023_code_comments.yaml").is_file():
            return directory
    raise FileNotFoundError("Run from the repository root or its notebooks directory")


def configure_style() -> str:
    """Color encodes category or magnitude, never subjective model quality."""
    plt.rcdefaults()
    try:
        font_manager.findfont("Times New Roman", fallback_to_default=False)
        font = "Times New Roman"
    except ValueError:
        font = "DejaVu Serif"
        print("Times New Roman is unavailable; figures use DejaVu Serif.")
    mpl.rcParams.update({
        "font.family": font, "font.size": 11, "axes.titlesize": 13,
        "axes.titleweight": "normal", "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#555555", "axes.linewidth": .65,
        "axes.axisbelow": True, "grid.color": "#DEDEDE", "grid.linewidth": .5,
        "legend.frameon": False, "legend.fontsize": 10,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "figure.dpi": 110,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    pd.set_option("display.max_columns", 16)
    pd.set_option("display.max_colwidth", 65)
    return font


def wrap(value, width=25):
    return textwrap.fill(str(value), width=width, break_long_words=True, break_on_hyphens=False)


def finish(ax, title, xlabel="", ylabel="", grid="x"):
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if grid:
        ax.grid(axis=grid)
    ax.tick_params(length=3, width=.6)


@dataclass
class Corpus:
    language: str
    raw: pd.DataFrame
    frame: pd.DataFrame
    labels: list[str]
    source: Path

    @property
    def y(self):
        return self.frame[self.labels]

    @property
    def n(self):
        return len(self.frame)


def normalize_text(values):
    return values.astype(str).str.split().str.join(" ")


def load_corpora(root: Path) -> dict[str, Corpus]:
    config = yaml.safe_load((root / "configs/datasets/nlbse2023_code_comments.yaml").read_text())
    result = {}
    required = {"comment_sentence_id", "class", "comment_sentence", "category", "instance_type", "partition"}
    for language in LANGUAGES:
        source = root / "data/raw/nlbse2023" / f"{language.lower()}.csv"
        raw = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        assert required <= set(raw), f"Missing raw columns: {source}"
        assert not raw[list(required)].eq("").any().any(), f"Empty raw values: {source}"
        assert raw.instance_type.isin(["0", "1"]).all() and raw.partition.isin(["0", "1"]).all()
        assert not raw.duplicated(["comment_sentence_id", "category"]).any()
        labels = sorted(config["languages"][language.lower()]["categories"])
        assert set(raw.category) == set(labels)
        consistency = raw.groupby("comment_sentence_id").agg(
            texts=("comment_sentence", "nunique"), classes=("class", "nunique"), categories=("category", "nunique"))
        assert consistency.texts.eq(1).all() and consistency.classes.eq(1).all()
        assert consistency.categories.eq(len(labels)).all()
        raw["instance_type"] = raw.instance_type.astype("int64")
        raw["partition"] = raw.partition.astype("int64")
        targets = raw.pivot(index="comment_sentence_id", columns="category", values="instance_type")[labels].astype("int64")
        meta = raw.drop_duplicates("comment_sentence_id").set_index("comment_sentence_id")[["class", "comment_sentence"]]
        frame = meta.rename(columns={"comment_sentence": "text"}).join(targets).sort_index()
        frame["normalized_text"] = normalize_text(frame.text)
        assert frame.normalized_text.ne("").all()
        frame["group_id"] = pd.factorize(frame.normalized_text, sort=False)[0]
        frame["word_count"] = frame.text.str.findall(r"\b\w+\b").str.len()
        frame["char_count"] = frame.text.str.len()
        assert frame[labels].isin([0, 1]).all().all()
        assert frame[labels].sum(axis=1).ge(1).all()
        result[language] = Corpus(language, raw, frame, labels, source)
    assert sum(c.n for c in result.values()) == 6738
    return result


def label_summary(corpus):
    support = corpus.y.sum().astype("int64")
    distinct = corpus.frame.groupby("group_id")[corpus.labels].max().sum().astype("int64")
    grouped = corpus.frame.groupby("group_id")[corpus.labels].sum()
    return pd.DataFrame({
        "positive_sentences": support, "negative_sentences": corpus.n - support,
        "prevalence_pct": support / corpus.n * 100,
        "negative_per_positive": (corpus.n - support) / support,
        "positive_text_groups": distinct,
        "largest_group_positive_share_pct": grouped.max() / support * 100,
    }).rename_axis("label")


def group_summary(corpus):
    sizes = corpus.frame.groupby("group_id").size()
    conflicts = corpus.frame.groupby("group_id")[corpus.labels].nunique().gt(1).any(axis=1)
    return {
        "sentences": corpus.n, "labels": len(corpus.labels), "classes": corpus.frame["class"].nunique(),
        "text_groups": len(sizes), "repeated_groups": int(sizes.gt(1).sum()),
        "repeated_group_members": int(sizes[sizes.gt(1)].sum()),
        "excess_copies": int((sizes - 1).sum()), "largest_group": int(sizes.max()),
        "groups_with_label_disagreement": int(conflicts.sum()),
        "cardinality": corpus.y.sum(axis=1).mean(), "target_density_pct": corpus.y.to_numpy().mean() * 100,
    }


def association(corpus, kind="jaccard"):
    y = corpus.y.to_numpy(dtype=np.int64)
    counts = y.T @ y
    support = y.sum(0)
    assert np.array_equal(counts.diagonal(), support) and (counts >= 0).all()
    if kind == "counts":
        values = counts
    elif kind == "jaccard":
        denominator = support[:, None] + support[None, :] - counts
        values = np.divide(counts, denominator, out=np.zeros_like(counts, dtype=float), where=denominator > 0)
    elif kind == "conditional":
        values = counts / support[:, None]
    else:
        raise ValueError(kind)
    return pd.DataFrame(values, index=corpus.labels, columns=corpus.labels)


def plot_prevalence(corpus):
    summary = label_summary(corpus).sort_values("prevalence_pct")
    fig, ax = plt.subplots(figsize=(8.4, 1.6 + .48 * len(summary)), layout="constrained")
    bars = ax.barh([wrap(s) for s in summary.index], summary.prevalence_pct,
                   color=LANGUAGE_COLORS[corpus.language], height=.62)
    ax.bar_label(bars, labels=[f"{p:.1f}%  (n={n:,})" for p, n in zip(summary.prevalence_pct, summary.positive_sentences)], padding=5, fontsize=10)
    ax.set_xlim(0, max(15, summary.prevalence_pct.max() * 1.44))
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    finish(ax, f"{corpus.language}: positive-label prevalence | N = {corpus.n:,}", "Share of sentences")
    return fig


def plot_association(corpus, kind="jaccard", vmax=None):
    values = association(corpus, kind)
    masked = np.ma.array(values.to_numpy(dtype=float), mask=np.eye(len(values), dtype=bool))
    if vmax is None:
        vmax = max(.01, float(masked.max()))
    fig, ax = plt.subplots(figsize=(8.2, 6.6), layout="constrained")
    cmap = plt.colormaps["Blues"].copy(); cmap.set_bad("#F2F2F2")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax)
    labels = [wrap(s, 18) for s in values.index]
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    for i in range(len(values)):
        for j in range(len(values)):
            value = values.iloc[i, j]
            text = "—" if i == j else f"{int(value):,}" if kind == "counts" else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if i != j and value > vmax * .6 else "#333333")
    title = {"jaccard": "Jaccard similarity", "conditional": "Conditional co-occurrence", "counts": "Joint positive counts"}[kind]
    finish(ax, f"{corpus.language}: {title} | diagonal omitted", grid=None)
    if kind == "conditional":
        ax.set_ylabel("Conditioning label"); ax.set_xlabel("Column label (observed annotation)")
    fig.colorbar(im, ax=ax, fraction=.04, pad=.03, label=title)
    return fig


def plot_ecdf(ax, values, label, color, linestyle="-"):
    values = np.sort(np.asarray(values))
    ax.step(np.r_[0, values], np.r_[0, np.arange(1, len(values) + 1) / len(values) * 100],
            where="post", label=label, color=color, linestyle=linestyle, linewidth=1.7)
    ax.set_ylim(0, 101)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_xlim(left=0)


def official_rates(corpus):
    return corpus.raw.groupby(["category", "partition"]).instance_type.agg(["sum", "size", "mean"])


def plot_official_rates(corpus):
    rates = official_rates(corpus)
    labels = list(label_summary(corpus).sort_values("prevalence_pct").index)
    a = rates.xs(0, level="partition").reindex(labels)
    b = rates.xs(1, level="partition").reindex(labels)
    fig, ax = plt.subplots(figsize=(8.4, 1.7 + .48 * len(labels)), layout="constrained")
    positions = np.arange(len(labels))
    for i, (x, y) in enumerate(zip(a["mean"] * 100, b["mean"] * 100)):
        ax.plot([x, y], [i, i], color="#BBBBBB", linewidth=1)
    ax.scatter(a["mean"] * 100, positions, color=SPLIT_COLORS["train"], marker="o", label="Original train", s=34, zorder=3)
    ax.scatter(b["mean"] * 100, positions, color=SPLIT_COLORS["test"], marker="s", label="Original test", s=30, zorder=3)
    ax.set_yticks(positions, [wrap(s) for s in labels]); ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(PercentFormatter(100)); ax.legend(loc="lower right")
    finish(ax, f"{corpus.language}: original category-specific partitions", "Positive-label prevalence")
    return fig


def _validate_prepared(frame, corpus):
    frame["comment_sentence_id"] = frame.comment_sentence_id.astype(str)
    assert not frame.comment_sentence_id.duplicated().any()
    assert frame.comment_sentence_id.isin(corpus.frame.index).all()
    reference = corpus.frame.reindex(frame.comment_sentence_id)
    assert np.array_equal(frame[corpus.labels].to_numpy(dtype=np.int64), reference[corpus.labels].to_numpy(dtype=np.int64))
    assert np.array_equal(frame.text.to_numpy(), reference.normalized_text.to_numpy())
    assert np.array_equal(frame["class"].to_numpy(), reference["class"].to_numpy())
    return frame


def current_splits(root, corpus):
    directory = root / "data/full_training_data" / corpus.language.lower()
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["protocol"] == "train_val_test_80_10_10"
    assert manifest["source_raw_sha256"] == hashlib.sha256(corpus.source.read_bytes()).hexdigest()
    result = {name: _validate_prepared(pd.read_csv(directory / f"{name}.csv", dtype={"comment_sentence_id": str}, keep_default_na=False), corpus)
              for name in ("train", "val", "test")}
    assert sum(len(frame) for frame in result.values()) == corpus.n
    assert len(set().union(*(set(f.comment_sentence_id) for f in result.values()))) == corpus.n
    for i, first in enumerate(result):
        for second in list(result)[i + 1:]:
            assert not set(result[first].text) & set(result[second].text), "Exact-text leakage"
    return result


def current_folds(root, corpus):
    directory = root / "data/k_fold_data" / corpus.language.lower()
    frame = _validate_prepared(pd.read_csv(directory / "all.csv", dtype={"comment_sentence_id": str}, keep_default_na=False), corpus)
    assert len(frame) == corpus.n
    folds = json.loads((directory / "folds.json").read_text())
    assert len(folds) == 5
    seen = []
    rows, prevalence = [], []
    for fold in folds:
        train, val = fold["train_indices"], fold["val_indices"]
        assert sorted(train + val) == list(range(corpus.n))
        assert not set(frame.iloc[train].text) & set(frame.iloc[val].text)
        seen.extend(val)
        rows.append({"fold": fold["fold"] + 1, "train_n": len(train), "validation_n": len(val)})
        prevalence.append(frame.iloc[val][corpus.labels].mean().to_numpy() * 100)
    assert sorted(seen) == list(range(corpus.n))
    return pd.DataFrame(rows).set_index("fold"), pd.DataFrame(prevalence, index=range(1, 6), columns=corpus.labels)


def plot_fold_shift(corpus, prevalence, limit=None):
    shift = prevalence.T.subtract(corpus.y.mean() * 100, axis=0)
    limit = max(.1, float(np.abs(shift.to_numpy()).max())) if limit is None else limit
    fig, ax = plt.subplots(figsize=(7.8, 2 + .55 * len(corpus.labels)), layout="constrained")
    im = ax.imshow(shift, cmap=SHIFT_CMAP, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.set_xticks(range(5), [f"Fold {i}" for i in range(1, 6)])
    ax.set_yticks(range(len(corpus.labels)), [wrap(s) for s in corpus.labels])
    for i in range(len(corpus.labels)):
        for j in range(5):
            value = shift.iloc[i, j]
            ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(value) > limit * .65 else "#333333")
    finish(ax, f"{corpus.language}: actual CV validation-label shift", grid=None)
    fig.colorbar(im, ax=ax, fraction=.04, pad=.03, label="Prevalence minus full population (percentage points)")
    return fig


def plot_split_shift(corpus, splits):
    base = corpus.y.mean() * 100
    fig, ax = plt.subplots(figsize=(8.5, 1.7 + .52 * len(corpus.labels)), layout="constrained")
    y = np.arange(len(corpus.labels))
    for i, name in enumerate(("train", "val", "test")):
        shift = splits[name][corpus.labels].mean() * 100 - base
        ax.scatter(shift, y + (i - 1) * .18, marker=("o", "D", "s")[i], color=SPLIT_COLORS[name], s=30,
                   label=f"{name.capitalize()} (N={len(splits[name]):,})")
    ax.axvline(0, color="#777777", linewidth=.8, linestyle="--")
    ax.set_yticks(y, [wrap(s) for s in corpus.labels]); ax.legend(loc="best", fontsize=9)
    finish(ax, f"{corpus.language}: fixed stage-2 label shift", "Prevalence minus full population (percentage points)")
    return fig
