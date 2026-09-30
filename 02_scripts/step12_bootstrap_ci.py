"""
Step 12 — Bootstrap Confidence Intervals
=========================================
Computes 95% bootstrap percentile confidence intervals for all held-out
test metrics (AUC, Sensitivity, Specificity, Balanced Accuracy, G-Mean)
for all seven classifiers.

Input : 03_outputs/step6/step6_test_probabilities.json
Output: 03_outputs/step12/step12_bootstrap_ci.csv
        03_outputs/step12/step12_bootstrap_ci_wide.csv  (LaTeX-ready)

Methodology:
  - 2,000 bootstrap resamples with replacement (random_state=42)
  - Percentile interval: [2.5th, 97.5th] percentile of bootstrap distribution
  - Resamples with zero Short Survivors are excluded from Sensitivity and
    G-Mean CIs (sensitivity is undefined); the count is reported.
  - AUC resamples with zero positives are excluded similarly.
  - Bootstrap applied to the default (0.50) threshold predictions.
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score,
    recall_score, confusion_matrix,
)

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__), '..')
STEP6_DIR = os.path.join(BASE_DIR, '03_outputs', 'step6')
OUT_DIR   = os.path.join(BASE_DIR, '03_outputs', 'step12')
os.makedirs(OUT_DIR, exist_ok=True)

N_BOOTSTRAP  = 2000
CI_LEVEL     = 0.95
RANDOM_STATE = 42

CLASSIFIERS_ORDER = ['KNN', 'NaiveBayes', 'DecisionTree',
                     'RandomForest', 'SVM', 'MLP', 'LightGBM']

DISPLAY_NAMES = {
    'KNN':          'KNN',
    'NaiveBayes':   'Naïve Bayes',
    'DecisionTree': 'Decision Tree',
    'RandomForest': 'Random Forest',
    'SVM':          'SVM',
    'MLP':          'MLP',
    'LightGBM':     'LightGBM',
}


# ══════════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS  (signature: y_true, y_proba, y_pred → float)
# ══════════════════════════════════════════════════════════════════════════

def metric_auc(yt, yp, yc):
    if len(np.unique(yt)) < 2:
        return np.nan
    return roc_auc_score(yt, yp)


def metric_sensitivity(yt, yp, yc):
    if yt.sum() == 0:
        return np.nan
    return recall_score(yt, yc, pos_label=1, zero_division=np.nan)


def metric_specificity(yt, yp, yc):
    if (yt == 0).sum() == 0:
        return np.nan
    return recall_score(yt, yc, pos_label=0, zero_division=np.nan)


def metric_balanced_acc(yt, yp, yc):
    if len(np.unique(yt)) < 2:
        return np.nan
    return balanced_accuracy_score(yt, yc)


def metric_gmean(yt, yp, yc):
    sens = metric_sensitivity(yt, yp, yc)
    spec = metric_specificity(yt, yp, yc)
    if np.isnan(sens) or np.isnan(spec):
        return np.nan
    return float(np.sqrt(sens * spec))


METRICS = [
    ('AUC',           metric_auc),
    ('Sensitivity',   metric_sensitivity),
    ('Specificity',   metric_specificity),
    ('Balanced_Acc',  metric_balanced_acc),
    ('GMean',         metric_gmean),
]


# ══════════════════════════════════════════════════════════════════════════
# BOOTSTRAP CI
# ══════════════════════════════════════════════════════════════════════════

def bootstrap_ci(y_true, y_proba, y_pred, metric_fn,
                 n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL, random_state=RANDOM_STATE):
    """
    Percentile bootstrap CI.
    Returns (point_estimate, lower, upper, n_excluded)
    n_excluded = resamples discarded because no positives in resample.
    """
    rng = np.random.RandomState(random_state)
    n   = len(y_true)
    scores     = []
    n_excluded = 0

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt  = y_true[idx]
        yp  = y_proba[idx]
        yc  = y_pred[idx]

        score = metric_fn(yt, yp, yc)
        if np.isnan(score):
            n_excluded += 1
            continue
        scores.append(score)

    alpha = (1 - ci) / 2
    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1 - alpha) * 100))
    point = metric_fn(y_true, y_proba, y_pred)

    return float(point), lower, upper, n_excluded


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_start = time.time()
    print("=" * 70)
    print("STEP 12 — BOOTSTRAP CONFIDENCE INTERVALS")
    print(f"  n_bootstrap={N_BOOTSTRAP}  CI={int(CI_LEVEL*100)}%  seed={RANDOM_STATE}")
    print("=" * 70)

    # ── Load test probabilities ────────────────────────────────────────
    prob_path = os.path.join(STEP6_DIR, 'step6_test_probabilities.json')
    with open(prob_path) as f:
        test_probs = json.load(f)

    rows_long = []   # one row per (classifier × metric)
    rows_wide = []   # one row per classifier, columns per metric

    for clf_name in CLASSIFIERS_ORDER:
        if clf_name not in test_probs:
            print(f"\n  WARNING: {clf_name} not in JSON, skipping")
            continue

        tp      = test_probs[clf_name]
        y_true  = np.array(tp['y_true'],  dtype=int)
        y_proba = np.array(tp['y_proba'], dtype=float)
        y_pred  = np.array(tp['y_pred'],  dtype=int)
        imb     = tp['imbalance_technique']

        n_pos = int(y_true.sum())
        n_tot = len(y_true)
        print(f"\n  {clf_name} + {imb}  "
              f"(n={n_tot}, Short Survivors={n_pos})")

        wide_row = {
            'classifier':          clf_name,
            'imbalance_technique': imb,
        }

        for metric_name, metric_fn in METRICS:
            pt, lo, hi, excl = bootstrap_ci(
                y_true, y_proba, y_pred, metric_fn
            )
            print(f"    {metric_name:<15s}: {pt:.3f}  "
                  f"95% CI [{lo:.3f}, {hi:.3f}]"
                  + (f"  (excluded: {excl})" if excl > 0 else ""))

            rows_long.append({
                'classifier':          clf_name,
                'imbalance_technique': imb,
                'metric':              metric_name,
                'point_estimate':      round(pt,  3),
                'ci_lower':            round(lo,  3),
                'ci_upper':            round(hi,  3),
                'n_excluded':          excl,
                'n_bootstrap':         N_BOOTSTRAP,
            })

            wide_row[f'{metric_name}_point'] = round(pt,  3)
            wide_row[f'{metric_name}_lo']    = round(lo,  3)
            wide_row[f'{metric_name}_hi']    = round(hi,  3)
            wide_row[f'{metric_name}_excl']  = excl

        rows_wide.append(wide_row)

    # ── Save CSVs ──────────────────────────────────────────────────────
    df_long = pd.DataFrame(rows_long)
    df_wide = pd.DataFrame(rows_wide)

    df_long.to_csv(os.path.join(OUT_DIR, 'step12_bootstrap_ci.csv'),      index=False)
    df_wide.to_csv(os.path.join(OUT_DIR, 'step12_bootstrap_ci_wide.csv'), index=False)

    print(f"\n  Saved: step12_bootstrap_ci.csv")
    print(f"  Saved: step12_bootstrap_ci_wide.csv")

    # ── Print LaTeX table rows ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("LATEX TABLE — held-out test results with 95% CIs")
    print("(rows ordered by AUC descending)")
    print("=" * 70)

    # Sort by AUC point estimate descending
    df_wide_sorted = df_wide.sort_values('AUC_point', ascending=False)

    col_order = [
        ('AUC',          'AUC'),
        ('Balanced_Acc', 'Bal.\\ Acc.'),
        ('Sensitivity',  'Sensitivity'),
        ('Specificity',  'Specificity'),
        ('GMean',        'G-Mean'),
    ]

    # Header
    print("\\begin{tabular}{llccccc}")
    print("  \\toprule")
    hdr_metrics = " & ".join(f"\\textbf{{{lbl}}}" for _, lbl in col_order)
    print(f"  \\textbf{{Classifier}} & \\textbf{{Technique}} & {hdr_metrics} \\\\")
    print("  \\midrule")

    for _, r in df_wide_sorted.iterrows():
        clf  = DISPLAY_NAMES.get(r['classifier'], r['classifier'])
        imb  = r['imbalance_technique']
        cells = []
        for metric_key, _ in col_order:
            pt = r[f'{metric_key}_point']
            lo = r[f'{metric_key}_lo']
            hi = r[f'{metric_key}_hi']
            cells.append(f"{pt:.3f} [{lo:.3f},\\,{hi:.3f}]")
        row_str = " & ".join(cells)
        print(f"  {clf} & {imb} & {row_str} \\\\")

    print("  \\bottomrule")
    print("\\end{tabular}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Step 12 completed in {elapsed:.1f}s")
    print(f"Outputs: {OUT_DIR}")
    print("=" * 70)
