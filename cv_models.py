"""Compare candidate models with repeated stratified K-fold CV.

Reports mean/std CV AUROC and the train AUROC so the gap (overfitting)
is visible for each model.
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import load_train_test

warnings.filterwarnings("ignore")
SEED = 42


def cv_auc(model_fn, X, y, n_splits=5, n_repeats=5, seed=SEED):
    """Repeated stratified CV. Returns (fold AUCs, train AUCs, out-of-fold preds averaged over repeats)."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    val_aucs, tr_aucs = [], []
    oof = np.zeros(len(y))
    for tr_idx, va_idx in rskf.split(X, y):
        m = model_fn()
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p_va = m.predict_proba(X.iloc[va_idx])[:, 1]
        p_tr = m.predict_proba(X.iloc[tr_idx])[:, 1]
        val_aucs.append(roc_auc_score(y.iloc[va_idx], p_va))
        tr_aucs.append(roc_auc_score(y.iloc[tr_idx], p_tr))
        oof[va_idx] += p_va / n_repeats
    return np.array(val_aucs), np.array(tr_aucs), oof


def logreg(C=0.1):
    return lambda: make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=5000),
    )


def rf(**kw):
    params = dict(n_estimators=500, min_samples_leaf=10, max_features="sqrt", n_jobs=-1, random_state=SEED)
    params.update(kw)
    return lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(**params))


def et(**kw):
    params = dict(n_estimators=500, min_samples_leaf=10, max_features="sqrt", n_jobs=-1, random_state=SEED)
    params.update(kw)
    return lambda: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(**params))


def lgbm(**kw):
    params = dict(
        n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
        random_state=SEED, verbose=-1,
    )
    params.update(kw)
    return lambda: lgb.LGBMClassifier(**params)


class OneFeature:
    """Baseline: rank by a single column."""
    def __init__(self, col):
        self.col = col
    def fit(self, X, y):
        return self
    def predict_proba(self, X):
        p = X[self.col].fillna(0).to_numpy(dtype=float)
        return np.c_[1 - p, p]


if __name__ == "__main__":
    X, y, _ = load_train_test()
    X = X.drop(columns=["sub_n_rows"])
    print(f"X {X.shape}, positives {y.sum()}")

    models = {
        "baseline: sub_2013 only": lambda: OneFeature("sub_2013"),
        "baseline: sub_streak only": lambda: OneFeature("sub_streak"),
        "logreg C=0.01": logreg(0.01),
        "logreg C=0.1": logreg(0.1),
        "logreg C=1": logreg(1),
        "random forest": rf(),
        "extra trees": et(),
        "lightgbm (shallow)": lgbm(),
        "lightgbm (deep, little reg.)": lgbm(num_leaves=63, max_depth=-1, min_child_samples=5, reg_lambda=0, n_estimators=500, learning_rate=0.05),
    }
    rows = []
    for name, fn in models.items():
        va, tr, _ = cv_auc(fn, X, y)
        rows.append((name, va.mean(), va.std(), tr.mean(), tr.mean() - va.mean()))
        print(f"{name:32s} CV AUC {va.mean():.4f} ± {va.std():.4f} | train {tr.mean():.4f} | gap {tr.mean()-va.mean():+.4f}")
