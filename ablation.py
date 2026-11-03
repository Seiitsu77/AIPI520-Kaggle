"""Feature-group ablation: drop one group at a time and compare repeated CV AUROC.

Run on the full feature set (before final_feature_set). Groups whose removal does
not hurt CV AUROC are removed in features.final_feature_set.
"""
import warnings

import numpy as np

from cv_models import cv_auc, lgbm, logreg
from features import load_train_test

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    X, y, _ = load_train_test()
    X = X.drop(columns=["sub_n_rows"])  # duplicate of sub_n_seasons
    groups = {
        "donation": [c for c in X if c.startswith("don_")],
        "geo/zip": [c for c in X if c.startswith(("dist_", "zip_", "in_CA"))],
        "tickets": [c for c in X if c.startswith("tk_")] + ["has_tk_history"],
        "sub location dummies": [c for c in X if c.startswith("loc_")],
        "sub 2013 detail": [c for c in X if c.startswith("sub13_")] + ["sub_tier_change"],
        "account misc": ["has_relationship", "has_shipping_zip"],
    }
    print(f"X {X.shape}, positives {y.sum()}")
    for mname, fn in [("logreg", logreg(0.1)), ("lgbm", lgbm())]:
        base, _, _ = cv_auc(fn, X, y)
        print(f"\n[{mname}] all features: {base.mean():.4f}")
        for g, cols in groups.items():
            v, _, _ = cv_auc(fn, X.drop(columns=cols), y)
            print(f"[{mname}]  drop {g:22s} ({len(cols):2d} cols): {v.mean():.4f}  delta {v.mean() - base.mean():+.4f}")
        # Account-ID prefix is a property of the data export, not of the patron -> checked, not used
        XL = X.assign(id_prefix_L=(X.index.str[:11] == "001i000000L").astype(int))
        v, _, _ = cv_auc(fn, XL, y)
        print(f"[{mname}]  + ID prefix feature: {v.mean():.4f}  delta {v.mean() - base.mean():+.4f}")
