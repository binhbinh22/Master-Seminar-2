# Rebuilt research report

English research report for **Seminar 2**, academic year 2026–2027, **Hanoi University of Science and Technology — School of Information & Communication Technology**. Instructor: **Dr. Phuong Nguyen (University of L'Aquila, Italy)**. Institution, course, instructor and all five team names/student IDs follow the existing `docs/report/report.tex`; the technical scope follows the supplied Artificial Intelligence for Software Engineering task brief. Experimental source snapshot: repository commit `8a84e2cc2294f42a09914ab0185114b2abc4dc9b`.

Start with `main.pdf` (32-page compiled report, 9 figures, 17 tables), `evidence_audit.md` (claims and sources), and `implementation_gaps.md` (unresolved requirements). The report is a complete reconstruction of available evidence, **not a certification of submission readiness**. No research model was retrained. `report_rebuilt_overleaf.zip` was also compiled successfully after extraction into a fresh temporary directory.

Git tracks the editable sources, figures, tables, verification scripts and evidence documents. The generated `main.pdf`, Overleaf ZIP, compilation files and rendered review pages remain local and are ignored by Git. After cloning, build `main.tex` using the commands below to obtain the PDF; upload the tracked LaTeX sources, `sections/`, `tables/`, `figures/` and `references.bib` to Overleaf if the local ZIP is unavailable.

## Contents

- `main.tex`, `sections/*.tex`, `references.bib`: editable LaTeX sources.
- `figures/`: four source-derived diagrams, ML/reference and CV/test charts, three label heatmaps (PDF and PNG).
- `tables/`: generated dataset, support and result tables; model configuration tables are in the section sources.
- `audit_facts.json`: all parsed staged run paths and numerical audit facts.
- `source_inventory.csv`: candidate evidence paths, SHA-256 hashes and sizes.
- `scripts/audit.py`: read-only data/metrics verification and scientific figure/table generation.
- `scripts/diagrams.py`: implementation/protocol diagrams.
- `validation.md`: compilation, numerical checks and visual review outcomes.

All original datasets, metrics and manuscripts are preserved. Temporary extracted original-manuscript text and document renderings, if retained under `build/`, are audit intermediates.

## Build in Overleaf

Upload `report_rebuilt_overleaf.zip` as a new project, select `main.tex` and **XeLaTeX** to preserve Vietnamese names. Bibliography uses BibTeX. Times New Roman is used when installed, with TeX Gyre Termes as an Overleaf-compatible fallback. Figures and tables are bundled; original repository datasets and model weights are not needed to compile the document. The cover information follows the existing manuscript; the body uses a standard A4 article layout because no separately identified official template was available.

## Build locally

From `report_rebuilt/`:

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

## Regenerate the evidence-derived assets

From the repository root, with pandas, NumPy, PyYAML and matplotlib installed:

```powershell
python report_rebuilt/scripts/audit.py
python report_rebuilt/scripts/diagrams.py
```

These scripts do not train models or modify original experimental evidence. They fail on duplicate run identities or numerical inconsistencies rather than selecting an arbitrary result.

## Reproduce model computations

The following are supported command examples, **not training executed during report authoring**. Use a fresh output path on each repetition. Existing prepared datasets avoid the absent default `data/raw/` directory.

```powershell
python -m pip install -r requirements.txt
python -m src.data.prepare_two_stage --raw-dir data --output-dir report_rebuilt/reproduction_data --folds 5 --seed 42
python -m experiments.ml.train_baseline --stage kfold --language all --seed 42 --output-dir results/reproduction_ml_cv
python -m experiments.ml.train_baseline --stage final --language all --methods ovr_logreg classifier_chain_logreg ovr_linear_svc ovr_sgd ovr_random_forest --seed 42 --output-dir results/reproduction_ml_final
python -m experiments.dl.train_exp1 --stage kfold --language python --device cpu --seed 42 --output-dir results/reproduction_e1_cv/python
python -m experiments.dl.train_exp1 --stage final --language python --device cpu --seed 42 --output-dir results/reproduction_e1_final/python
```

The preparation example writes alternative splits; the training examples continue to use the original default prepared roots. To train on regenerated data, pass `--data-dir report_rebuilt/reproduction_data/k_fold_data` for CV and the corresponding `full_training_data` root for final runs. Model downloads/cache and suitable compute are prerequisites.

For a Bash-capable shell (Git Bash/WSL/Linux), complete neural orchestration can follow the implemented manifest selection:

```bash
bash scripts/train_kfold.sh --seed 42 --device cuda:0 --output-dir results/reproduction_dl_cv
python -m src.evaluation.two_stage_results select --results results/reproduction_dl_cv --top-k 4 --output-dir results/reproduction_selection
bash scripts/train_full.sh --selection results/reproduction_selection/selection.json --device cuda:0 --output-dir results/reproduction_dl_selected
```

Alternatively, to recreate the **historical fixed final subset**, use:

```bash
bash scripts/train_full.sh --experiments 1,3,9,15,17 --seed 42 --device cuda:0 --output-dir results/reproduction_dl_fixed_subset
```

The two alternatives are intentionally distinct: archived final runs do not follow each language's saved top-four manifest exactly. Both reproduce exploratory methodology; neither makes the later test independent of screening.

## Scope of verification

The report checks 6,738 raw-to-wide sentence targets, actual grouping/splits, all 69 CV summary records and all 30 final F1 records. The original exported summary micro-F1 values match raw neural runs. Five classical models and seventeen neural configurations are verified from code/configurations. No row-level predictions or distributed model checkpoints were available for replay or qualitative error examples.

Missing switchable persistence adapters, independent evaluation, repeated protocol seeds, environment lock and complete final coverage remain listed as gaps. Audit Python 3.12.10 is not claimed as the historical training Python version.
