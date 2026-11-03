# AIPI 520 Kaggle: Classical Music Meets Classical ML

Predict which past patrons buy a 2014–15 season subscription. The metric is AUROC.

The final submission is a regularized logistic regression (C = 0.03) on 47 hand-built account features. The full write-up is in [writeup.md](writeup.md).

## Results

| Model | Nested CV AUROC (5-fold × 3) | Train AUROC | Gap | Public LB |
|---|---|---|---|---|
| **Logistic regression** (final) | 0.9786 ± 0.0076 | 0.9820 | +0.0035 | 0.97370 |
| Extra Trees | 0.9770 ± 0.0078 | 0.9921 | +0.0151 | – |
| LightGBM | 0.9754 ± 0.0093 | 0.9865 | +0.0111 | – |
| Rank-average ensemble | 0.9780 ± 0.0079 | – | – | 0.97198 |

## Files

| File | Purpose |
|---|---|
| `features.py` | Builds one row per account from the raw files (subscriptions, tickets, donations, ZIP/geography); `final_feature_set` applies the ablation decisions |
| `cv_models.py` | Repeated stratified K-fold CV helper (reports train and validation AUROC) and model builders; `python cv_models.py` runs the first model comparison |
| `eda.py` | Exploratory numbers quoted in the write-up, plus adversarial validation (train vs. test) |
| `ablation.py` | Feature-group ablation (drop one group at a time) → `ablation_log.txt` |
| `train.py` | Small fixed hyperparameter grids, nested CV, final fit and submission files → `train_log.txt` |
| `experiments_round2.py` | Second-round ideas judged by a pre-set paired-CV acceptance rule (none accepted) → `round2_log.txt` |
| `history_model.py` | Explored "back-test" renewal model built from past seasons; it gave no CV gain and is not used in the final model |
| `submission_logreg.csv` | Final submission (logistic regression) |
| `submission.csv` | Rank-average ensemble submission (not chosen) |

## How to run

The competition data is not included in this repository. Download it from the Kaggle competition page and place the CSV files in `for_students/`:

```
for_students/
  account.csv  concerts.csv  concerts_2014-15.csv  sample_submission.csv
  subscriptions.csv  test.csv  tickets_all.csv  train.csv  zipcodes.csv
```

Then:

```bash
pip install -r requirements.txt
python eda.py
python ablation.py
python train.py               # nested CV + submissions; add --no-nested to skip the slow nested CV
python experiments_round2.py
```

Tested with Python 3.13. All CV splits use fixed random seeds.
