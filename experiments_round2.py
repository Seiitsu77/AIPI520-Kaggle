"""Round 2: a few targeted ideas, judged only by paired repeated CV.

Acceptance rule (fixed before running):
  * mean paired improvement over the current logreg >= +0.002, and
  * better on >= 70% of the 25 folds, and
  * the gain must hold up on a second, fresh set of CV folds (different seed).
The public leaderboard is never used to choose.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from cv_models import cv_auc, logreg
from features import load_train_test

warnings.filterwarnings("ignore")

# Features whose effect may differ for current (2013-14) subscribers vs everyone else
SEG_COLS = ["don_n", "don_log_2013", "don_log_lifetime", "sub_n_seasons", "sub_streak", "sub_n_last5",
            "sub_years_since_last", "tk_n_rows", "tk_2013", "has_sub_history", "has_tk_history"]


def add_recency(X):
    """Replace the 'no history = 99 years' coding with bounded recency scores (better for a linear model)."""
    X = X.copy()
    X["sub_recency"] = np.where(X["has_sub_history"] == 1, 1 / X["sub_years_since_last"], 0)
    X["sub_years_since_last"] = X["sub_years_since_last"].clip(upper=25)
    X["tk_recency"] = np.where(X["has_tk_history"] == 1, 1 / (2014 - X["tk_last_yr"]), 0)
    X["lapsed_recent"] = ((X["sub_2013"] == 0) & (X[["sub_2010", "sub_2011", "sub_2012"]].sum(axis=1) > 0)).astype(int)
    return X


def add_segment_interactions(X):
    """Let the linear model use separate slopes for 2013-14 subscribers and non-subscribers."""
    X = X.copy()
    for c in SEG_COLS:
        X[f"{c}_x_sub13"] = X[c].fillna(0) * X["sub_2013"]
    return X


def spline_logreg(C):
    def build(X):
        cont = [c for c in X.columns if X[c].nunique() > 6]
        other = [c for c in X.columns if c not in cont]
        pre = ColumnTransformer([
            ("cont", make_pipeline(SimpleImputer(strategy="median"), SplineTransformer(n_knots=4, degree=2)), cont),
            ("other", SimpleImputer(strategy="most_frequent"), other),
        ])
        return make_pipeline(pre, StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    return build


class Segmented:
    """Separate logistic regressions for 2013-14 subscribers and for everyone else."""
    def __init__(self, C):
        self.C = C

    def fit(self, X, y):
        self.m = {}
        for s in (0, 1):
            idx = (X["sub_2013"] == s).to_numpy()
            self.m[s] = logreg(self.C)().fit(X[idx], y[idx])
        return self

    def predict_proba(self, X):
        p = np.zeros(len(X))
        for s in (0, 1):
            idx = (X["sub_2013"] == s).to_numpy()
            if idx.any():
                p[idx] = self.m[s].predict_proba(X[idx])[:, 1]
        return np.c_[1 - p, p]


def best_over_C(make_fn, X, y, seed, Cs=(0.01, 0.03, 0.1)):
    """Tune C on the same folds (small grid) and return fold AUCs of the best C."""
    best = None
    for C in Cs:
        v, t, _ = cv_auc(make_fn(C), X, y, seed=seed)
        if best is None or v.mean() > best[1].mean():
            best = (C, v, t)
    return best


if __name__ == "__main__":
    X, y, _ = load_train_test(final=True)
    Xr = add_recency(X)
    variants = {
        "A  recency coding": (lambda C: logreg(C), Xr),
        "B  segment interactions": (lambda C: logreg(C), add_segment_interactions(X)),
        "C  A + B": (lambda C: logreg(C), add_segment_interactions(Xr)),
        "D  splines (A features)": (lambda C: (lambda: spline_logreg(C)(Xr)), Xr),
        "E  segmented models (A features)": (lambda C: (lambda: Segmented(C)), Xr),
    }

    results = {}
    for seed in (42, 7):  # 42 = folds used for screening, 7 = fresh folds for confirmation
        print(f"\n===== CV folds seed {seed} (5-fold x 5 repeats) =====")
        C0, base, tb = best_over_C(lambda C: logreg(C), X, y, seed)
        print(f"current logreg (C={C0})           CV {base.mean():.4f}  train {tb.mean():.4f}")
        for name, (fn, Xv) in variants.items():
            C, v, t = best_over_C(fn, Xv, y, seed)
            d = v - base
            ok = d.mean() >= 0.002 and np.mean(d > 0) >= 0.7
            results.setdefault(name, []).append(ok)
            print(f"{name:34s} (C={C}) CV {v.mean():.4f}  train {t.mean():.4f}  "
                  f"paired diff {d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f}  better in {np.mean(d > 0):.0%}  "
                  f"{'PASS' if ok else 'fail'}")

    print("\nAccepted (passes on both fold sets):", [n for n, r in results.items() if all(r)] or "none")
