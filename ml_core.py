"""Core ML logic (no Streamlit code here, so it can be tested separately)."""
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, fbeta_score, confusion_matrix, classification_report)


class KernelNaiveBayes(BaseEstimator, ClassifierMixin):
    """Naive Bayes whose per-feature class-conditional densities use KDE (kernel trick)."""

    def __init__(self, bandwidth=0.3, kernel="gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        self.log_priors_ = np.log(np.array([(y == c).mean() for c in self.classes_]))
        self.kdes_ = []
        for c in self.classes_:
            Xc = X[y == c]
            self.kdes_.append([KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(Xc[:, [j]])
                               for j in range(X.shape[1])])
        return self

    def _jll(self, X):
        X = np.asarray(X)
        out = np.zeros((X.shape[0], len(self.classes_)))
        for k in range(len(self.classes_)):
            out[:, k] = self.log_priors_[k] + sum(
                self.kdes_[k][j].score_samples(X[:, [j]]) for j in range(X.shape[1]))
        return out

    def predict_proba(self, X):
        jll = self._jll(X)
        return np.exp(jll - logsumexp(jll, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[np.argmax(self._jll(X), axis=1)]


def find_columns(df):
    """Locate sensor/target columns by keyword (UCI and Kaggle versions name them differently)."""
    def find(key):
        matches = [c for c in df.columns if key.lower() in c.lower()]
        if not matches:
            raise ValueError(f"Column containing '{key}' not found. Is this the AI4I 2020 CSV?")
        return matches[0]
    feats = [find("Air temperature"), find("Process temperature"),
             find("Rotational speed"), find("Torque"), find("Tool wear")]
    return feats, find("Machine failure")


def tune_threshold(model, X, y, beta=2.0):
    """Pick the probability cutoff that maximises F-beta using CV on TRAINING data only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    grid = np.linspace(0.01, 0.99, 99)
    scores = [fbeta_score(y, (proba >= t).astype(int), beta=beta, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def train_all(df, bandwidth=0.3):
    feats, target = find_columns(df)
    X, y = df[feats], df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    Xtr_s, Xte_s = scaler.transform(X_tr), scaler.transform(X_te)

    models = {"Gaussian Naive Bayes": GaussianNB(),
              "Kernel (KDE) Naive Bayes": KernelNaiveBayes(bandwidth=bandwidth),
              "AdaBoost Classifier": AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)}
    thresholds = {}
    for name, m in models.items():
        thresholds[name] = tune_threshold(m, Xtr_s, y_tr)
        m.fit(Xtr_s, y_tr)
    return dict(feats=feats, target=target, scaler=scaler, models=models, thresholds=thresholds,
                Xte_s=Xte_s, y_te=np.asarray(y_te), n_train=len(X_tr), n_test=len(X_te),
                fail_rate=float(y.mean()))


def evaluate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(
        Accuracy=accuracy_score(y_true, y_pred),
        Precision=precision_score(y_true, y_pred, zero_division=0),
        Recall=recall_score(y_true, y_pred, zero_division=0),
        F1=f1_score(y_true, y_pred, zero_division=0),
        TypeI=fp / (fp + tn) if (fp + tn) else 0.0,     # false-alarm rate
        TypeII=fn / (fn + tp) if (fn + tp) else 0.0,    # missed-failure rate
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
        report=classification_report(y_true, y_pred, target_names=["Normal", "Failure"],
                                     zero_division=0))


def predict_one(art, model_name, values, threshold):
    """values: list of 5 raw sensor readings in the order of art['feats']."""
    x = pd.DataFrame([values], columns=art["feats"])
    p_fail = float(art["models"][model_name].predict_proba(art["scaler"].transform(x))[0, 1])
    return p_fail, int(p_fail >= threshold)
