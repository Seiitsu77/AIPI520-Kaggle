"""How much does AUROC move when only ~half of the test set is scored (the public leaderboard)?

Scores the final model's out-of-fold predictions on many random subsets of public-leaderboard size.
"""
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

from cv_models import cv_auc, logreg
from features import load_train_test

warnings.filterwarnings("ignore")
N_PUBLIC = 1488  # about half of the 2,975 test accounts

if __name__ == "__main__":
    X, y, _ = load_train_test(final=True)
    _, _, oof = cv_auc(logreg(0.03), X, y)
    yv = y.to_numpy()
    print(f"out-of-fold AUROC, final model: {roc_auc_score(yv, oof):.4f}")

    rng = np.random.default_rng(0)
    aucs = np.array([roc_auc_score(yv[idx], oof[idx])
                     for idx in (rng.choice(len(yv), N_PUBLIC, replace=False) for _ in range(3000))])
    print(f"AUROC on random subsets of {N_PUBLIC} accounts: sd {aucs.std():.4f}, "
          f"5-95% range [{np.percentile(aucs, 5):.4f}, {np.percentile(aucs, 95):.4f}]")
