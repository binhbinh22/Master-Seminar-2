# Code comment classification seminar report

**Seminar 2, 2026–2027 — Hanoi University of Science and Technology, School of Information & Communication Technology.** Instructor: **Dr. Phuong Nguyen (University of L'Aquila, Italy)**. Institution and all five team names/student IDs follow `docs/report/report.tex`.

Open the committed [main.pdf](main.pdf) directly after cloning. The current English report has **35 pages, 14 figures, 20 tables and 19 references**. Its cover includes the complete team/instructor metadata; keywords and acknowledgements are included. Vietnamese translation was explicitly excluded by the user during this revision.

The scientific evidence comprises 6,738 sentences, 69 screening records and 30 final-evaluation records. Original datasets/model metrics are preserved. New EDA/learning figures and annotated examples use existing data and stored histories; research models were not retrained. Evaluation limitations are consolidated in Section 8, with implementation scope recorded in `implementation_gaps.md`.

## Build locally or on Overleaf

Use **XeLaTeX + BibTeX** to preserve Vietnamese names. Times New Roman is selected when installed, with TeX Gyre Termes as fallback. From this directory:

```bash
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

For Overleaf, upload `main.tex`, `references.bib`, `sections/`, `tables/` and `figures/`, then select XeLaTeX. The locally generated `report_rebuilt_overleaf.zip` contains the same self-contained sources and can also be uploaded. The ZIP and temporary build files are ignored; the PDF is tracked.

## Regenerate report assets

From the repository root, with project dependencies installed:

```bash
python report_rebuilt/scripts/audit.py
python report_rebuilt/scripts/research_assets.py
python report_rebuilt/scripts/diagrams.py
python report_rebuilt/scripts/verify_prepare.py
```

The audit recalculates dataset counts, fold summaries, final F1 and exported-summary consistency. `research_assets.py` produces three EDA figures, two learning-dynamics figures and annotated sentence examples from actual artifacts. Neural curves are saved epoch histories; ML curves are separate refits at iteration limits, not guaranteed prefixes of one optimization run. E4 shares E1 screening scores and is represented by a table note instead of duplicate rows.

Aggregate F1 uses four decimal places; per-label tables use three. Percentage-point differences in the prose use the displayed aggregate values; original high-precision metric JSONs remain unchanged.

## Dataset storage and tests

`src/data/storage.py` defines ByteStore, FileStore and MemoryStore. The shared staged loader accepts store injection, and both Python staged-training CLI families accept `--storage file` or `--storage memory`. This changes dataset persistence without changing feature/model/evaluation logic. Memory mode reads an initial snapshot; run exports still use filesystem directories. See [storage documentation](../docs/storage.md).

```bash
python -m unittest discover -s tests -v
```

**35 offline tests pass**, including six new adapter/real-data equivalence tests. Offline model fixtures are tiny mocked software tests, not newly reported research experiments. The actual-data test now falls back to the checked-in `data/*.csv` when the optional downloader directory is absent. All six staged datasets also reproduce through the real preparation functions.

## Reproduce experiments

These are supported training commands, not computations rerun for the revised report. Use fresh output paths and an appropriate device/model cache.

```bash
python -m pip install -r requirements.txt
python -m src.data.prepare_two_stage --raw-dir data --output-dir report_rebuilt/reproduction_data --folds 5 --seed 42
python -m experiments.ml.train_baseline --stage kfold --language all --seed 42 --storage file --output-dir results/reproduction_ml_cv
python -m experiments.dl.train_exp1 --stage kfold --language python --device cpu --seed 42 --storage memory --output-dir results/reproduction_e1_cv/python
```

The training examples use original prepared roots. Pass `--data-dir report_rebuilt/reproduction_data/k_fold_data` to use regenerated CV data, or the corresponding `full_training_data` root for final training. To reproduce the historical fixed final neural subset in a Bash-capable shell:

```bash
bash scripts/train_full.sh --experiments 1,3,9,15,17 --seed 42 --device cuda:0 --output-dir results/reproduction_dl_final
```

Using a top-four selection manifest instead follows the implemented selection rule but a different candidate set from the archived fixed-five final cohort. Section 5 and Section 8 distinguish these protocols.

## Provenance and validation

- `evidence_audit.md`, `source_inventory.csv`, `audit_facts.json`: claims, run paths and source hashes.
- `research_assets.json`: EDA statistics, curve paths and exact annotated example IDs/text/labels.
- `split_verification.json`: regeneration checks for all six staged datasets.
- `pdf_validation.json`, `validation.md`: build, numerical, visual and software-test outcomes.
- `implementation_gaps.md`: remaining scientific/artifact gaps and the precise scope of completed storage adapters.

The original scientific snapshot is commit `8a84e2cc2294f42a09914ab0185114b2abc4dc9b`; report-branch revisions add adapters, tests and presentation improvements. Historical training environment details and sentence predictions remain incomplete, as stated in the limitations section.
