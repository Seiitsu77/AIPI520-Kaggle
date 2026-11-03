""""Back-test" renewal model built only from subscriptions.csv.

For every past season T (2000-01 ... 2013-14) we compute subscription-history
features using seasons before T and label = "subscribed in season T". That gives
tens of thousands of examples of renew / not-renew behavior without touching
train.csv labels. The model is then applied with T = 2014 to give every account
a renewal score, which the main model uses as one extra feature.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from features import DATA, _season_start


def history_features(seasons, T):
    """seasons: account x year 0/1 matrix plus a long table. Uses only years < T."""
    years = [y for y in seasons.columns if y < T]
    s = seasons[years]
    s = s[s.sum(axis=1) > 0]  # accounts with some history before T
    f = pd.DataFrame(index=s.index)
    for k in range(1, 6):
        f[f"lag{k}"] = s[T - k] if (T - k) in s.columns else 0
    f["n_seasons"] = s.sum(axis=1)
    f["n_last5"] = s[[y for y in years if y >= T - 5]].sum(axis=1)
    last = s.apply(lambda r: r[r > 0].index.max(), axis=1)
    first = s.apply(lambda r: r[r > 0].index.min(), axis=1)
    f["years_since_last"] = T - last
    f["tenure"] = last - first + 1
    f["frac"] = f["n_seasons"] / f["tenure"]
    streak = np.zeros(len(s), dtype=int)
    alive = np.ones(len(s), dtype=bool)
    for y in range(T - 1, min(years) - 1, -1):
        alive &= s[y].to_numpy() > 0 if y in s.columns else False
        streak += alive
    f["streak"] = streak
    return f


def detail_features(sub, T):
    """Seats / tier / package of the most recent season before T."""
    prev = sub[sub["yr"] < T]
    last_yr = prev.groupby("account.id")["yr"].transform("max")
    last = prev[prev["yr"] == last_yr].groupby("account.id")
    f = pd.DataFrame({
        "last_seats": last["no.seats"].sum(),
        "last_tier": last["subscription_tier"].sum(),
        "last_price_level": last["price.level"].min(),
        "last_full": last["package"].apply(lambda p: p.fillna("").str.startswith("Full").any()).astype(int),
        "mean_seats": prev.groupby("account.id")["no.seats"].mean(),
    })
    return f


def build_panel(sub, seasons, T):
    f = history_features(seasons, T).join(detail_features(sub, T))
    label = seasons[T].reindex(f.index).fillna(0).astype(int) if T in seasons.columns else None
    return f, label


def renewal_scores(train_years=range(2000, 2014)):
    sub = pd.read_csv(DATA / "subscriptions.csv")
    sub["yr"] = _season_start(sub["season"])
    seasons = sub.groupby(["account.id", "yr"]).size().unstack(fill_value=0).gt(0).astype(int)

    Xs, ys = [], []
    for T in train_years:
        f, lab = build_panel(sub, seasons, T)
        Xs.append(f)
        ys.append(lab)
    Xp, yp = pd.concat(Xs), pd.concat(ys)

    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                               random_state=42, verbose=-1)
    model.fit(Xp, yp)

    f2014, _ = build_panel(sub, seasons, 2014)
    return pd.Series(model.predict_proba(f2014)[:, 1], index=f2014.index, name="renew_score"), (Xp, yp)


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    sub = pd.read_csv(DATA / "subscriptions.csv")
    sub["yr"] = _season_start(sub["season"])
    seasons = sub.groupby(["account.id", "yr"]).size().unstack(fill_value=0).gt(0).astype(int)
    # Honest check of the panel model itself: train on T<2013, test on T=2013
    tr = [build_panel(sub, seasons, T) for T in range(2000, 2013)]
    Xtr, ytr = pd.concat([a for a, _ in tr]), pd.concat([b for _, b in tr])
    Xte, yte = build_panel(sub, seasons, 2013)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                           random_state=42, verbose=-1).fit(Xtr, ytr)
    print("panel rows", len(Xtr), "pos rate", ytr.mean().round(3))
    print("temporal holdout AUC (predict 2013-14 from <=2012-13):", roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    sub_prev = Xte["lag1"] == 1
    print("  among 2012-13 subscribers (renewal AUC):", roc_auc_score(yte[sub_prev], m.predict_proba(Xte[sub_prev])[:, 1]))
