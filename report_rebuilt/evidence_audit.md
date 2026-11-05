# Evidence audit

Audit date: 5 November 2026 (Asia/Saigon). Source: `main`, commit `8a84e2cc2294f42a09914ab0185114b2abc4dc9b`. The previously checked-out `feature/data-analysis` contained legacy work; this report uses the current `origin/main` snapshot. Original manuscripts, source, datasets, and results were not edited.

## Evidence classes

- **A: implemented/verified.** Source was substantively inspected, artifacts parsed, and numerical checks run as stated below. Static inspection is not a claim of full training execution.
- **B: documented/unverified.** A statement or capability lacks sufficient implementation/execution evidence.
- **C: proposed.** Future work only; no completed experiment is implied.

`source_inventory.csv` records paths, SHA-256 and sizes for candidate evidence. Inventory membership means discovery/hash inspection, not manual review of every file. `audit_facts.json` records all 99 parsed staged runs, paths, seeds, cohort fingerprints, dataset statistics, configuration changes, prediction search results, and summary discrepancies. Table/figure generation reads original artifacts directly.

## Manuscripts inspected

| Source | Inspection and reconciliation |
|---|---|
| `docs/report/report.tex` and README | Inspected cover, layout, methods, conclusion, references and outstanding tasks. The cover supplies Hanoi University of Science and Technology, the School of Information & Communication Technology, Seminar 2, Dr. Phuong Nguyen (University of L'Aquila, Italy), academic year 2026–2027 and all five team names/student IDs; these are restored verbatim in the rebuilt report. Legacy 0.6282 classical results and the incomplete neural comparison are not imported into the staged cohort. No separately identified official university template/rubric found. |
| `reports/CodeCommentClassification_Report_EN.pdf` | Extracted text (71 pages), inspected abstract and relevant methodological/result claims. Compatible ML results marked unavailable in this older report now exist in this snapshot. |
| `reports/CodeCommentClassification_Paper_EN_FINAL.pdf` | Extracted text (5 pages), inspected summary and main comparison/validity statements. Neural leader values agree with current run evidence. |
| Vietnamese report/paper PDFs | Discovered and text-extracted to assess manuscript coverage. Not used as authoritative numerical sources. |

Temporary manuscript text extractions are build intermediates, not newly authored reports. The final document replaces neither originals nor their claims silently.

## Claim-to-source mapping

| Report claim / output | Primary repository evidence | Verification / class |
|---|---|---|
| Cover institution, school, course, instructor, academic year and team identities | `docs/report/report.tex`, title page | Transcribed from the existing manuscript, as confirmed by the user; A |
| 6,738 sentences, 5/7/7 labels | `data/{python,java,pharo}.csv`; `data/k_fold_data/*/all.csv`; `label_cols.json` | Pivoted raw targets by ID/category, compared every processed target; A |
| Positive/negative counts, cardinality, multi-label prevalence | Same sentence-level files | Recalculated by `scripts/audit.py`; A |
| Taxonomy spelling/order | `data/k_fold_data/*/label_cols.json`; `configs/datasets/nlbse2023_code_comments.yaml` | Parsed all label orders; descriptive glosses are not an annotation manual; A/B boundary |
| Normalization and exact duplicate grouping | `src/data/prepare_two_stage.py:raw_to_rows` | Source inspected; counts computed from actual groups; A |
| Greedy balancing and split construction | `src/data/prepare_two_stage.py:grouped_parts, prepare` | Source inspected; no named iterative-stratification claim; A |
| Shared loader, schema, fingerprints, masks | `src/data/two_stage.py`; `src/data/nlbse2023.py`; `src/models/ml/data.py` | Source inspected; actual splits checked for group overlap and CV coverage; A |
| ~80/10/10 sample sizes | `data/full_training_data/*/{train,val,test}.csv` | Recounted; same target population verified against all.csv; A |
| Class overlap | `class` in final train/test CSVs | Counted exact class-string intersection; does not establish project identity; A |
| Word/char TF-IDF and fold-local fitting | `src/models/ml/features.py`; `experiments/ml/staged_experiment.py` | Source inspected; A |
| Five ML methods and dummy settings | `src/models/ml/models.py`; saved run `config.yaml` | Builders inspected, configurations inventoried; A |
| E1–E17 identities/hyperparameters | `src/models/dl/zoo.py`; all `configs/exp*.yaml` | Parsed all configurations; compared saved neural model config fields with current YAML (no differences found); A |
| BCE, weighted BCE, focal | `src/models/dl/roberta_baseline.py`, `roberta_weighted_bce.py`, `roberta_variants.py` | Objective code inspected; A |
| ASL, Bernoulli R-Drop, FGM, label graph | `src/models/dl/advanced_variants.py` | Objective/head implementation inspected; A |
| E9 v2 and different stopping policy | E9 YAML; all E9 screening fold records | All 15 folds report head_version 2; A |
| Differential/layer-wise learning rates | `src/training/optimizers.py`; E13/E14/E16 YAML | Formula and actual grouping code inspected; A |
| Epoch selection, patience, warmup, gradient clipping | `src/training/masked_classifier.py`; `experiments/dl/roberta_experiment.py` | Source inspected; same validation set used for checkpoint and score; A |
| Two-stage protocol and dependence | `experiments/dl/staged_experiment.py`; ML staged runner; data manifests; `docs/two_stage_training.md`; README | Source and cohort evidence agree on nonindependent final test; A |
| Top-four vs actual final subset | `results/stage1_summary/selection.json`; all final neural `metrics.json` | Parsed selected lists vs actual E1/E3/E9/E15/E17; mismatch preserved; A |
| Metric equations and zero denominators | `src/evaluation/multilabel.py`; `src/models/ml/metrics.py`; `src/models/ml/scoring.py` | Both implementations inspected; AP score distinctions documented; A |
| Mean and SD conventions | `src/evaluation/summaries.py` | Arithmetic mean and sample SD; recomputed every stored CV summary metric; A |
| ML CV results | `results/training/kfold/20261102-222649/<method>/<language>/metrics.json` | 18 records, including dummy; every fold summary recomputed; A |
| Transformer CV results | `results/training/kfold_gpu0/exp*/<language>/metrics.json`; `kfold_gpu1/...` | 51 records, five folds each; every summary recomputed; A |
| ML final results | `results/training/final/20261104-224046/<method>/<language>/metrics.json` | 15 records; F1 reconstructed from per-label evidence; A |
| Transformer final results | `results/training/final_gpu0/exp*/<language>/metrics.json`; `final_gpu1/...` | 15 records; F1 reconstructed from per-label evidence; A |
| Exported micro-F1 consistency | `results/stage1_summary/stage1_metrics.csv`; `results/stage2_summary/stage2_metrics.csv` | Every exported row compared against source run micro-F1; zero discrepancies; A |
| Label-level improvements/regressions | Final `test.per_label` dictionaries | Full label supports/P/R/F1 parsed, tables/heatmaps generated; A |
| Row-level error analysis unavailable | File search over `results/`; staged runner source | No prediction artifact or best_model.pt found; aggregate counts cannot identify sentences; A (absence in snapshot) |
| File persistence and serialization capabilities | Staged runners; `.gitignore` | Direct write paths verified; checkpoints/joblib not distributed; A implementation vs missing artifact |
| Replaceable memory/file storage | Data loaders and runners; searches for store/repository abstractions | No interchangeable adapter interface found; requirement missing; C for proposed interface |
| Environment and execution instructions | `requirements.txt`; CLI parsers; `scripts/train_kfold.sh`, `train_full.sh`; README | Commands derived from actual parsers; historical exact environment absent; A/B |
| Notebook support | All four `notebooks/*.ipynb`; `notebooks/README.md`; saved EDA outputs | Notebook cells/output counts inspected; three EDA notebooks contain saved results; execution not rerun; A/B |

Specific per-run paths and fingerprints are exhaustive in `audit_facts.json`; source hashes are exhaustive for the inventory patterns in `source_inventory.csv`.

## Numerical checks performed

1. Raw-to-wide target identity for all 6,738 sentence IDs.
2. Final population/target equality to screening population.
3. Zero exact-text-group overlap in each CV and final split; every CV sentence validated exactly once.
4. Population/fingerprint consistency among recorded same-language/stage ML and DL runs. Stored byte fingerprints are compared **between original records**, not falsely claimed to match Windows-converted local file bytes.
5. Every stored CV summary mean and sample SD recalculated from its fold metrics (69 runs).
6. Final micro-F1 from total true/predicted/support positives and macro-F1 from per-label F1 recalculated (30 runs).
7. Stage summary CSV micro-F1 values matched the raw corresponding metrics (51 screening + 15 final neural rows).
8. Exact counts, prevalences, overlaps, configuration coverage, final subset, and absent prediction/checkpoint artifacts recalculated.

A separate regeneration invoked the real `prepare('data', temporary_directory)` and `load_stage_data()` functions. All six generated dataset byte fingerprints matched their corresponding local archived data, and population, sentence order, labels, and folds matched. This strengthens the artifact validation beyond the original audit script. No archived data were regenerated in place.

Existing `test_two_stage.py` tests: two passed, one failed before actual-data checking because its hard-coded `data/raw/` is missing. The passing smoke test uses tiny mocked backbones and briefly optimizes fixture models; these are software tests, not new research-model experiments. Full research training was not rerun.

These checks do not reproduce research training, validate annotation semantics, establish statistical significance, or confirm external generalization.

## External scholarly sources verified

Verified on 5 November 2026 through primary paper records and repositories; bibliography metadata deliberately uses arXiv records where venue metadata was not separately checked.

- Official NLBSE'23 repository: https://github.com/nlbse2023/code-comment-classification
- Classifier Chains author manuscript record, University of Waikato (2009, pp. 254–269, DOI verified): https://hdl.handle.net/10289/3259
- RoBERTa: https://arxiv.org/abs/1907.11692
- CodeBERT: https://arxiv.org/abs/2002.08155
- Focal Loss: https://arxiv.org/abs/1708.02002
- R-Drop: https://arxiv.org/abs/2106.14448
- Asymmetric Loss arXiv record: https://arxiv.org/abs/2009.14119

No unverified competition-paper author list from the older manuscript was copied. The local bibliography's ASL entry follows the retrieved arXiv record rather than silently substituting a different author/version record. A broader systematic literature search and full annotation-manual verification remain open.
