"""Classical multi-label model builders.

All four are linear classifiers wrapped for multi-label output. See
src/models/ml/README.md ("Vì sao chọn mô hình tuyến tính") for why linear models were
chosen over non-linear alternatives (Random Forest, gradient boosting, etc.) for this
dataset.
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.svm import LinearSVC


def build_models():
    return {
        'dummy_most_frequent': OneVsRestClassifier(
            DummyClassifier(strategy='most_frequent')
        ),
        'ovr_logreg': OneVsRestClassifier(
            LogisticRegression(max_iter=2000, class_weight='balanced', C=5.0)
        ),
        'classifier_chain_logreg': ClassifierChain(
            LogisticRegression(max_iter=2000, class_weight='balanced', C=5.0),
            order='random', random_state=42,
        ),
        'ovr_linear_svc': OneVsRestClassifier(
            LinearSVC(class_weight='balanced', C=1.0)
        ),
        'ovr_sgd': OneVsRestClassifier(
            SGDClassifier(loss='log_loss', max_iter=2000, tol=1e-3, class_weight='balanced', random_state=42)
        ),
    }
