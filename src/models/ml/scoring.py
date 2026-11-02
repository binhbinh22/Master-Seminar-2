"""Get a continuous per-label score alongside the hard prediction for any sklearn
multi-label estimator, regardless of whether its base estimator exposes predict_proba
(LogisticRegression, RandomForest, DummyClassifier) or only decision_function
(LinearSVC). Average precision only needs a ranking-consistent score, so an uncalibrated
decision_function is fine for that purpose even though it isn't a probability.
"""
import numpy as np


def predict_scores(clf, X):
    if hasattr(clf, "predict_proba"):
        return np.asarray(clf.predict_proba(X), dtype=float)
    if hasattr(clf, "decision_function"):
        return np.asarray(clf.decision_function(X), dtype=float)
    return np.asarray(clf.predict(X), dtype=float)
