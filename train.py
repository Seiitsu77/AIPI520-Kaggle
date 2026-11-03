"""Final pipeline: tune -> honest (nested) CV estimate -> ensemble -> submission.

Overfitting safeguards
  * small, pre-defined hyperparameter grids (no open-ended search)
  * nested CV: hyperparameters are chosen on inner folds only, the outer folds
    give an unbiased estimate of the whole tuning procedure
  * train vs. validation AUROC reported for every model
  * the final ensemble is a plain rank average (no weights fitted on the data)
"""
import itertools
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from cv_models import SEED, et, lgbm, logreg
from features import DATA, load_train_test

warnings.filterwarnings("ignore")

GRIDS = {
    "logreg": (logreg, {"C": [0.01, 0.03, 0.1, 0.3]}),
    "extra_trees": (et, {"min_samples_leaf": [5, 10, 20], "max_features": ["sqrt", 0.3]}),
    "lightgbm": (lgbm, {"n_estimators": [200, 400], "num_leaves": [4, 7], "min_child_samples": [20, 50]}),
}


def configs(grid):
    keys = list(grid)
    for vals in itertools.product(*grid.values()):
        yield dict(zip(keys, vals))


def make(name, cfg):
    builder, _ = GRIDS[name]
    if name == "logreg":
        return builder(cfg["C"])()
    return builder(**cfg)()


def cv_score(name, cfg, X, y, splitter):
    aucs = []
    for tr, va in splitter.split(X, y):
        m = make(name, cfg).fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
    return np.mean(aucs)


def tune(name, X, y, n_repeats=2, seed=SEED):
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    scores = {tuple(cfg.items()): cv_score(name, cfg, X, y, splitter) for cfg in configs(GRIDS[name][1])}
    best = max(scores, key=scores.get)
    return dict(best), scores[best]


def rank_avg(preds):
    return np.mean([rankdata(p) / len(p) for p in preds], axis=0)


def nested_cv(X, y, n_outer_repeats=3):
    """Outer loop = honest estimate. Inner loop = hyperparameter choice."""
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_outer_repeats, random_state=SEED + 1)
    res = {n: [] for n in GRIDS} | {"ensemble": []}
    train_res = {n: [] for n in GRIDS}
    for i, (tr, va) in enumerate(outer.split(X, y)):
        Xtr, ytr, Xva, yva = X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]
        preds = []
        for name in GRIDS:
            cfg, _ = tune(name, Xtr, ytr, n_repeats=1, seed=SEED + i)
            m = make(name, cfg).fit(Xtr, ytr)
            p = m.predict_proba(Xva)[:, 1]
            preds.append(p)
            res[name].append(roc_auc_score(yva, p))
            train_res[name].append(roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1]))
        res["ensemble"].append(roc_auc_score(yva, rank_avg(preds)))
        print(f"  outer fold {i + 1:2d}: " + "  ".join(f"{k} {v[-1]:.4f}" for k, v in res.items()))
    return res, train_res


if __name__ == "__main__":
    X, y, X_test = load_train_test(final=True)
    print(f"train {X.shape}, test {X_test.shape}, positives {y.sum()} ({y.mean():.1%})")

    if "--no-nested" not in sys.argv:
        print("\n== Nested CV (unbiased estimate of tuning + model) ==")
        res, train_res = nested_cv(X, y)
        for k, v in res.items():
            gap = f" | train {np.mean(train_res[k]):.4f} gap {np.mean(train_res[k]) - np.mean(v):+.4f}" if k in train_res else ""
            print(f"{k:12s} {np.mean(v):.4f} ± {np.std(v):.4f}{gap}")

    print("\n== Tune on full training set, fit, predict test ==")
    test_preds = []
    for name in GRIDS:
        cfg, score = tune(name, X, y, n_repeats=3)
        print(f"{name:12s} best {cfg}  CV {score:.4f}")
        # average a few seeds for the tree models to reduce variance
        seeds = [0] if name == "logreg" else [0, 1, 2, 3, 4]
        p = np.zeros(len(X_test))
        for s in seeds:
            m = make(name, cfg)
            if s and hasattr(m, "set_params"):
                key = "random_state" if name == "lightgbm" else "extratreesclassifier__random_state"
                m.set_params(**{key: s})
            p += m.fit(X, y).predict_proba(X_test)[:, 1] / len(seeds)
        test_preds.append(p)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    # Two candidates: the rank-average ensemble and plain logistic regression
    for fname, pred in [("submission.csv", rank_avg(test_preds)), ("submission_logreg.csv", test_preds[0])]:
        sub = pd.DataFrame({"ID": X_test.index, "Predicted": pred})
        assert sub["ID"].tolist() == sample["ID"].tolist() and sub["Predicted"].notna().all()
        sub.to_csv(fname, index=False)
        print(f"\nwrote {fname}", sub.shape)
    print("correlation between model test predictions (Spearman):")
    print(pd.DataFrame(np.array(test_preds).T, columns=list(GRIDS)).corr(method="spearman").round(3))
