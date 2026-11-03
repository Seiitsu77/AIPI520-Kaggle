"""Build one row of features per account from the raw competition files.

Everything is computed from data that exists *before* the 2014-15 season,
and the same function is used for train and test, so there is no
train/test mismatch.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "for_students"
TARGET_YR = 2014  # the 2014-15 season we are predicting

# Approximate venue coordinates (lat, long) used for distance features
VENUES = {
    "sf": (37.777, -122.420),        # San Francisco (Herbst / SF Symphony area)
    "berkeley": (37.869, -122.259),  # Berkeley (First Congregational)
    "peninsula": (37.444, -122.161), # Palo Alto / Peninsula
    "contra_costa": (37.902, -122.064),  # Walnut Creek
}


def _season_start(s):
    """'2013-2014' -> 2013"""
    return pd.to_numeric(s.astype(str).str[:4], errors="coerce")


def _haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    d = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(d))


def subscription_features(sub):
    sub = sub.copy()
    sub["yr"] = _season_start(sub["season"])
    sub["is_full"] = sub["package"].fillna("").str.startswith("Full").astype(int)
    g = sub.groupby("account.id")

    f = pd.DataFrame(index=g.size().index)
    f["sub_n_rows"] = g.size()
    f["sub_n_seasons"] = g["yr"].nunique()
    f["sub_first_yr"] = g["yr"].min()
    f["sub_last_yr"] = g["yr"].max()
    f["sub_years_since_last"] = TARGET_YR - f["sub_last_yr"]
    f["sub_tenure"] = f["sub_last_yr"] - f["sub_first_yr"] + 1
    f["sub_frac_seasons"] = f["sub_n_seasons"] / f["sub_tenure"]
    f["sub_mean_seats"] = g["no.seats"].mean()
    f["sub_mean_tier"] = g["subscription_tier"].mean()
    f["sub_mean_price_level"] = g["price.level"].mean()
    f["sub_n_locations"] = g["location"].nunique()
    f["sub_frac_full"] = g["is_full"].mean()

    # Seasons subscribed in each of the last 5 seasons, and in the last 3/5/10
    seasons = sub.groupby(["account.id", "yr"]).size().unstack(fill_value=0).gt(0).astype(int)
    for y in range(2009, 2014):
        f[f"sub_{y}"] = seasons.get(y, 0)
    for k in (3, 5, 10):
        cols = [y for y in seasons.columns if y >= TARGET_YR - k]
        f[f"sub_n_last{k}"] = seasons[cols].sum(axis=1)

    # Consecutive-season streak that ends at 2013-14 (0 if not subscribed in 2013-14)
    def streak(row):
        n, y = 0, 2013
        while y in row.index and row[y] == 1:
            n += 1
            y -= 1
        return n
    f["sub_streak"] = seasons.apply(streak, axis=1)

    # Details of the most recent (2013-14) subscription
    last = sub[sub["yr"] == 2013].groupby("account.id")
    f["sub13_seats"] = last["no.seats"].sum()
    f["sub13_tier"] = last["subscription_tier"].sum()
    f["sub13_price_level"] = last["price.level"].min()
    f["sub13_n_pkgs"] = last.size()
    f["sub13_full"] = last["is_full"].max()
    f["sub13_multi"] = last["multiple.subs"].apply(lambda s: (s == "yes").any()).astype(float)
    prev = sub[sub["yr"] == 2012].groupby("account.id")
    f["sub_tier_change"] = f["sub13_tier"].fillna(0) - prev["subscription_tier"].sum().reindex(f.index).fillna(0)

    # Most common location (categorical, one-hot later)
    f["sub_main_location"] = g["location"].agg(lambda s: s.mode().iat[0] if s.notna().any() else "None")
    return f


def ticket_features(tk):
    tk = tk.copy()
    tk["yr"] = _season_start(tk["season"])
    tk["price_num"] = pd.to_numeric(tk["price.level"], errors="coerce")
    g = tk.groupby("account.id")
    f = pd.DataFrame(index=g.size().index)
    f["tk_n_rows"] = g.size()
    f["tk_n_seats"] = g["no.seats"].sum()
    f["tk_n_seasons"] = g["yr"].nunique()
    f["tk_last_yr"] = g["yr"].max()
    f["tk_2013"] = (f["tk_last_yr"] == 2013).astype(int)
    f["tk_mean_price_level"] = g["price_num"].mean()
    f["tk_has_marketing"] = g["marketing.source"].apply(lambda s: s.notna().any()).astype(int)
    f["tk_family"] = g["location"].apply(lambda s: (s == "Family concert").any()).astype(int)
    return f


def account_features(acc, zips):
    f = acc.set_index("account.id")[["amount.donated.2013", "amount.donated.lifetime", "no.donations.lifetime"]].copy()
    f.columns = ["don_2013", "don_lifetime", "don_n"]
    f["don_log_2013"] = np.log1p(f["don_2013"])
    f["don_log_lifetime"] = np.log1p(f["don_lifetime"])
    f["don_avg"] = f["don_lifetime"] / f["don_n"].replace(0, np.nan)
    first = pd.to_datetime(acc.set_index("account.id")["first.donated"], errors="coerce")
    f["don_first_yr"] = first.dt.year
    f["don_years_since_first"] = TARGET_YR - f["don_first_yr"]
    f["has_relationship"] = acc.set_index("account.id")["relationship"].notna().astype(int)
    f["has_shipping_zip"] = acc.set_index("account.id")["shipping.zip.code"].notna().astype(int)

    # Billing zip -> location / income info (fall back to shipping zip)
    zip_raw = acc.set_index("account.id")["billing.zip.code"].fillna(acc.set_index("account.id")["shipping.zip.code"])
    zip5 = zip_raw.astype(str).str.extract(r"^(\d{3,5})")[0].str.zfill(5)
    z = zips.copy()
    z["zip5"] = z["Zipcode"].astype(str).str.zfill(5)
    z["income_per_return"] = z["TotalWages"] / z["TaxReturnsFiled"]
    z = z.drop_duplicates("zip5").set_index("zip5")
    loc = z.reindex(zip5.values)[["Lat", "Long", "State", "EstimatedPopulation", "income_per_return"]]
    loc.index = zip5.index
    f["zip_known"] = loc["Lat"].notna().astype(int)
    f["in_CA"] = (loc["State"] == "CA").astype(int)
    f["zip_population"] = loc["EstimatedPopulation"]
    f["zip_income"] = loc["income_per_return"]
    dists = []
    for name, (la, lo) in VENUES.items():
        d = _haversine(loc["Lat"], loc["Long"], la, lo)
        f[f"dist_{name}"] = d
        dists.append(d)
    f["dist_nearest_venue"] = np.vstack(dists).min(axis=0)  # NaN when zip unknown
    return f


def build_features():
    acc = pd.read_csv(DATA / "account.csv", encoding="latin-1", dtype={"billing.zip.code": str, "shipping.zip.code": str})
    sub = pd.read_csv(DATA / "subscriptions.csv")
    tk = pd.read_csv(DATA / "tickets_all.csv")
    zips = pd.read_csv(DATA / "zipcodes.csv")

    feats = account_features(acc, zips)
    feats = feats.join(subscription_features(sub), how="left")
    feats = feats.join(ticket_features(tk), how="left")

    # Accounts with no subscription/ticket history: counts are 0, "years since" is large
    count_cols = [c for c in feats.columns if c.startswith(("sub_n", "sub_20", "sub_streak", "sub13_", "tk_n", "tk_2013", "tk_has", "tk_family", "sub_frac"))]
    feats[count_cols] = feats[count_cols].fillna(0)
    feats["sub_years_since_last"] = feats["sub_years_since_last"].fillna(99)
    feats["sub_tier_change"] = feats["sub_tier_change"].fillna(0)
    feats["has_sub_history"] = feats["sub_first_yr"].notna().astype(int)
    feats["has_tk_history"] = feats["tk_last_yr"].notna().astype(int)
    feats["sub_main_location"] = feats["sub_main_location"].fillna("None")

    feats = pd.get_dummies(feats, columns=["sub_main_location"], prefix="loc", dtype=int)
    return feats


# Groups that did not improve CV AUC in the ablation study (cv_models.py) -> removed
DROPPED = ["sub_n_rows", "sub_tier_change"]
DROPPED_PREFIXES = ("loc_", "sub13_")
# Heavy-tailed columns -> log1p (helps logistic regression; trees are unaffected)
LOG_COLS = ["don_avg", "zip_population", "zip_income", "tk_n_seats",
            "dist_sf", "dist_berkeley", "dist_peninsula", "dist_contra_costa", "dist_nearest_venue"]


def final_feature_set(feats):
    feats = feats.drop(columns=DROPPED + [c for c in feats if c.startswith(DROPPED_PREFIXES)])
    feats = feats.drop(columns=["don_2013", "don_lifetime"])  # log versions are kept
    feats[LOG_COLS] = np.log1p(feats[LOG_COLS])
    return feats


def load_train_test(final=False):
    feats = build_features()
    if final:
        feats = final_feature_set(feats)
    train = pd.read_csv(DATA / "train.csv").set_index("account.id")
    test = pd.read_csv(DATA / "test.csv").set_index("ID")
    X_train = feats.reindex(train.index)
    X_test = feats.reindex(test.index)
    return X_train, train["label"], X_test


if __name__ == "__main__":
    X, y, Xt = load_train_test()
    print(X.shape, Xt.shape, y.mean())
    print(X.describe().T.to_string())
