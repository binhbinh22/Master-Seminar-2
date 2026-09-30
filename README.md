# ML baseline: One-vs-Rest Logistic Regression (TF-IDF)

Files:
- `run_ml_baseline.py`: runnable script to perform TF-IDF + One-vs-Rest LR with iterative stratified k-fold CV.
- `requirements.txt`: Python dependencies.

Quick start:

1. Create a virtualenv and install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run baseline (default uses `python.csv` in repo root):

```bash
python run_ml_baseline.py --data python.csv --folds 5
```

Outputs: `baseline_results.json` with averaged metrics.
# Master-Seminar-2
