# Code Comment Classification — Analysis Report (Python subset)

Date: 2026-09-24

## 1. Task
NLBSE'23 Code Comment Classification (multi-label). Full competition dataset spans
Java + Python + Pharo (6,738 sentences total); this repo currently works with the
Python subset only (2,555 unique comment sentences, 5 categories).

## 2. Data bug found and fixed
`python.csv` stores each comment sentence "exploded" into 5 rows (one per category),
with `instance_type` (0/1) as the true label and `category` only naming which slot a
row represents. All four original pipelines (`run_ml_baseline.py`, `full_pipeline.py`,
`detailed_analysis.py`, `multiclass_pipeline.py`) used the `category` column directly as
the label for every row, ignoring `instance_type`. This duplicated every comment 5x with
mostly-contradictory single-label targets, which is why those pipelines scored near-zero
F1 (~0.01-0.33 degenerate).

Fix: [data_utils.py](../data_utils.py) pivots the exploded rows back to one row per
`comment_sentence_id` with a multi-hot label vector built from `instance_type`. This
produces the correct multi-label setup: 2,555 samples, avg. 1.12 labels/sample.

## 3. Method
- Features: TF-IDF, word (1-2 grams) + char_wb (3-5 grams), combined via `FeatureUnion`.
- Models: One-vs-Rest Logistic Regression, Classifier Chain (Logistic Regression base),
  One-vs-Rest Linear SVM, One-vs-Rest SGD (log-loss).
- Validation: 5-fold `MultilabelStratifiedKFold` (iterative-stratification), vectorizer
  refit per fold to avoid train/test leakage.
- Metrics: micro-F1, macro-F1, sample-averaged Jaccard, Hamming loss, micro precision/recall.

See [ml_pipeline.py](../ml_pipeline.py).

## 4. Results (5-fold CV, mean)

| Model | micro-F1 | macro-F1 | Jaccard | Hamming loss |
|---|---|---|---|---|
| **OVR Logistic Regression** | **0.6282** | **0.5928** | 0.5843 | — |
| OVR SGD (log-loss) | 0.6266 | 0.5932 | 0.5876 | — |
| OVR Linear SVM | 0.6236 | 0.5854 | 0.5801 | — |
| Classifier Chain (LR) | 0.6098 | 0.5667 | 0.5977 (best) | — |

Full numbers: [ml_pipeline_results.json](ml_pipeline_results.json).

### Per-label breakdown (OVR Logistic Regression)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Usage | 0.7251 | 0.7000 | 0.7118 | 800 |
| Parameters | 0.7035 | 0.7090 | 0.7050 | 794 |
| Expand | 0.5620 | 0.6032 | 0.5815 | 504 |
| Summary | 0.5551 | 0.6079 | 0.5793 | 454 |
| DevelopmentNotes | 0.3639 | 0.4138 | 0.3862 | 312 |

Full numbers: [per_label_metrics.json](per_label_metrics.json).

## 5. Analysis
- Performance tracks label frequency: `Usage` and `Parameters` (highest support, ~800
  samples) reach F1 ~0.70-0.71; `DevelopmentNotes` (lowest support, 312 samples) lags at
  F1 ~0.39. More training examples per label directly improves that label's F1.
- Linear models (LR, SVM, SGD) perform almost identically (~0.61-0.63 micro-F1),
  suggesting the TF-IDF feature space is close to linearly separable for this task and
  the bottleneck is data volume/label imbalance rather than model capacity.
- Classifier Chain trades micro/macro-F1 for the best Jaccard score — modeling label
  co-occurrence (a comment tagged `Usage` is likelier to also carry `Parameters`) helps
  get exact-set overlap right even though it very slightly hurts per-label F1 vs. plain
  OVR.
- `DevelopmentNotes` is both the rarest label and the most semantically diffuse
  ("miscellaneous notes for developers"), explaining why it is the hardest category by a
  clear margin.

## 6. RoBERTa baseline — not run
A `roberta-base` fine-tuning baseline was scaffolded in
[roberta_pipeline.py](../roberta_pipeline.py) (multi-label sigmoid head, same CV splits,
MPS acceleration on Apple Silicon) but could not run: the machine has only ~360MB of
free disk space, not enough to download the ~500MB model. Per user's decision, this
comparison was skipped for now; `roberta_pipeline.py` is ready to run once disk space is
freed (`python3 roberta_pipeline.py --folds 5 --epochs 6`).

## 7. Known limitations of this analysis
- Python-only subset (2,555 samples), not the full 6,738-sample Java+Python+Pharo
  dataset the task description references — would need `java.csv`/`pharo.csv` from the
  NLBSE'23 repo to match that number exactly, and each language has a different label
  set.
- No transformer/deep-learning baseline yet (see §6).
- Fixed 0.5 decision threshold for OVR/SGD; no per-label threshold tuning was done in
  this corrected pipeline (unlike the old, buggy `detailed_analysis.py`, which tuned
  thresholds on broken labels and is now superseded).
