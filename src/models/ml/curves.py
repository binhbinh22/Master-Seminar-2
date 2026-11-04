"""Approximate a per-step training curve for sklearn estimators that otherwise give no
visibility into their optimization progress (unlike DL's per-epoch loss).

sklearn's `.fit()` runs its solver's full loop internally and returns only the converged
result -- no intermediate values. We get an equivalent curve by sweeping the estimator's
"amount of training" hyperparameter (`max_iter` for the lbfgs/liblinear/SGD solvers,
`n_estimators` for Random Forest) and refitting from scratch at each checkpoint. Because
these solvers are deterministic given a fixed random_state, refitting at max_iter=k
reproduces exactly the state a longer fit would have been in after k iterations -- this
is a real reconstruction of the optimization trajectory, not a simulation. The cost is
repeated full fits instead of one streaming fit; for this dataset size (≤2,044 train
rows) that is a few extra seconds, not worth avoiding.

`dummy_most_frequent` has no such parameter (it doesn't optimize anything) and is
intentionally absent from CURVE_PARAM -- there is no curve to draw for it.
"""
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from src.models.ml.metrics import compute_metrics
from src.models.ml.scoring import predict_scores

CURVE_PARAM = {
    "ovr_logreg": ("estimator__max_iter", [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1600, 2000]),
    "classifier_chain_logreg": ("estimator__max_iter", [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1600, 2000]),
    "ovr_linear_svc": ("estimator__max_iter", [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]),
    "ovr_sgd": ("estimator__max_iter", [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1600, 2000]),
    "ovr_random_forest": ("estimator__n_estimators", [5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300]),
}

# What the swept parameter actually means for each solver -- only SGD is a true "epoch"
# (sklearn's own docs call SGDClassifier's max_iter "the number of epochs"). The lbfgs
# and liblinear solvers count optimizer iterations, not passes over the data; Random
# Forest has no iteration concept at all, it counts trees. Mislabeling all of these as
# "epoch" (as the shared DL plotting code does) would be actively misleading.
STEP_LABEL = {
    "ovr_logreg": "L-BFGS iteration",
    "classifier_chain_logreg": "L-BFGS iteration",
    "ovr_linear_svc": "Solver iteration (liblinear coordinate descent)",
    "ovr_sgd": "SGD epoch (pass over training data)",
    "ovr_random_forest": "Number of trees",
}


def training_curve(method, clf_template, X_train, y_train, X_val, y_val, label_names):
    if method not in CURVE_PARAM:
        return []
    param_name, checkpoints = CURVE_PARAM[method]
    param = param_name.split("__")[-1]
    history = []
    for step in checkpoints:
        clf = clone(clf_template)
        clf.set_params(**{param_name: step})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            clf.fit(X_train, y_train)

        train_pred = np.asarray(clf.predict(X_train))
        train_metrics = compute_metrics(y_train, predict_scores(clf, X_train), train_pred, label_names, per_label=False)

        val_pred = np.asarray(clf.predict(X_val))
        val_metrics = compute_metrics(y_val, predict_scores(clf, X_val), val_pred, label_names, per_label=False)

        history.append({
            "step": step, "step_param": param,
            "train_loss": train_metrics["hamming_loss"],
            "validation_micro_f1": val_metrics["micro_f1"],
            "validation_macro_f1": val_metrics["macro_f1"],
        })
    return history


def save_curve_plot(output_dir, method, curve):
    if not curve:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pathlib import Path
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    steps = [r["step"] for r in curve]
    label = STEP_LABEL.get(method, curve[0]["step_param"])

    fig, (loss_ax, val_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    loss_ax.plot(steps, [r["train_loss"] for r in curve], marker="o")
    loss_ax.set(title="Train Hamming loss", xlabel=label, ylabel="Loss")
    loss_ax.grid(alpha=0.25)

    val_ax.plot(steps, [r["validation_micro_f1"] for r in curve], marker="o", label="micro-F1")
    val_ax.plot(steps, [r["validation_macro_f1"] for r in curve], marker="o", label="macro-F1")
    val_ax.set(title="Validation score", xlabel=label, ylabel="Score")
    val_ax.grid(alpha=0.25)
    val_ax.legend(fontsize=8)

    path = directory / "training_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [str(path)]
