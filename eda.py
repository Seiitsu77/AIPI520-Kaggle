"""Exploratory numbers quoted in the writeup."""
import warnings

import pandas as pd
from sklearn.metrics import roc_auc_score

from cv_models import cv_auc, lgbm
from features import DATA, load_train_test

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    X, y, Xt = load_train_test()
    acc = pd.read_csv(DATA / "account.csv", encoding="latin-1")
    tk = pd.read_csv(DATA / "tickets_all.csv")
    d = X.assign(y=y.values)
    print(f"train {len(d)} accounts, {d.y.sum()} positives ({d.y.mean():.1%})")

    print("\nPurchase rate by season of the most recent subscription:")
    last = d.sub_last_yr.fillna(0).astype(int).rename("last_sub_season_start")
    print(d.groupby(last).y.agg(["mean", "sum", "count"]).tail(8).round(3).to_string())

    s13 = d.sub_2013 == 1
    print(f"\n2013-14 subscribers in train: {s13.sum()}")
    print(f"AUROC of sub_streak alone among 2013-14 subscribers: {roc_auc_score(d.y[s13], d.sub_streak[s13]):.3f}")

    ns = d[~s13]
    print("\nAmong accounts without a 2013-14 subscription, share that bought single tickets:")
    print(ns.groupby("y").has_tk_history.mean().round(3).rename({0: "negatives", 1: "positives"}).to_string())

    print("\nPurchase rate by 2013 donation:")
    print(d.groupby(d.don_2013 > 0).y.agg(["mean", "sum", "count"]).round(3).to_string())
    life = acc.set_index("account.id")["amount.donated.lifetime"].reindex(y.index)
    print(f"lifetime donation (train accounts): median {life.median():.0f}, max {life.max():,.0f}")

    # first.donated dates inside the 2014 sales window may be concurrent with the purchase
    fd = pd.to_datetime(acc["first.donated"], errors="coerce")
    late = acc.loc[fd >= "2014-04-01", ["account.id"]].assign(month=fd[fd >= "2014-04-01"].dt.to_period("M"))
    print(f"\nfirst.donated on/after 2014-04 (all accounts): {len(late)} | months {late.month.min()} .. {late.month.max()}")
    late = late.merge(y.rename("y"), left_on="account.id", right_index=True)
    print(f"  of which in train: {len(late)}, positive {late.y.sum()}")

    print(f"\ntickets_all.csv: {len(tk)} rows, {tk['account.id'].nunique()} accounts "
          f"({tk['account.id'].nunique() / len(acc):.1%} of all accounts)")

    # Adversarial validation: can a model tell train rows from test rows?
    Xa = pd.concat([X, Xt]).reset_index(drop=True)
    ya = pd.Series([0] * len(X) + [1] * len(Xt))
    v, _, _ = cv_auc(lgbm(), Xa, ya, n_repeats=1)
    print(f"\nadversarial validation AUROC (train vs test): {v.mean():.3f}")
