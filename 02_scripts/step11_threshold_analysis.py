"""
Step 11 — Threshold Optimisation Analysis
==========================================
Addresses the reviewer requirement: show what happens when the decision
threshold is optimised beyond the default 0.5.

Three alternative thresholds are derived from OUT-OF-FOLD CV predictions
(never from the held-out test set) for each classifier:
  - Default (0.5)
  - Youden:             maximises sensitivity + specificity − 1
  - Sensitivity-opt.:   highest threshold achieving sensitivity >= 0.90
  - F1-optimal:         maximises minority-class F1

OOF probabilities are collected by re-running CV with the best params
already found by Optuna (step5) — no new optimisation is performed.

All thresholds are then applied to the held-out test set probabilities
saved by step6 (step6_test_probabilities.json).

Outputs:
  - step11_thresholds.csv       (derived thresholds per classifier)
  - step11_threshold_results.csv (all 4 × metrics per classifier)
  - step11_threshold_table.png   (figure-ready heatmap/table)
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    roc_curve, f1_score, recall_score, roc_auc_score,
    balanced_accuracy_score, confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
)
from imblearn.under_sampling import (
    RandomUnderSampler, NearMiss, TomekLinks
)
from imblearn.combine import SMOTETomek, SMOTEENN

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.join(os.path.dirname(__file__), '..')
PROCESSED_DIR = os.path.join(BASE_DIR, '01_data', 'processed')
STEP5_DIR     = os.path.join(BASE_DIR, '03_outputs', 'step5')
STEP6_DIR     = os.path.join(BASE_DIR, '03_outputs', 'step6')
OUT_DIR       = os.path.join(BASE_DIR, '03_outputs', 'step11')
FIG_DIR       = os.path.join(BASE_DIR, 'overleaf', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE   = 42
N_SPLITS       = 5
N_REPEATS      = 3
MIN_SENSITIVITY = 0.90   # target for sensitivity-optimised threshold

CLASSIFIERS_ORDER = ['KNN', 'NaiveBayes', 'DecisionTree',
                     'RandomForest', 'SVM', 'MLP', 'LightGBM']

# ── Colour palette ─────────────────────────────────────────────────────────
PAL = {
    'short':    '#f54f74',
    'standard': '#00b0be',
    'accent':   '#ffb255',
    'bg':       '#ffcd8e',
    'white':    '#FFFFFF',
}


# ══════════════════════════════════════════════════════════════════════════
# CLASSIFIER / SAMPLER FACTORIES (identical to step6)
# ══════════════════════════════════════════════════════════════════════════

def make_classifier(name, params):
    p = dict(params)
    if name == 'MLP' and isinstance(p.get('hidden_layer_sizes'), list):
        p['hidden_layer_sizes'] = tuple(p['hidden_layer_sizes'])
    if name == 'KNN':
        return KNeighborsClassifier(**p)
    if name == 'NaiveBayes':
        return GaussianNB(**p)
    if name == 'DecisionTree':
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **p)
    if name == 'RandomForest':
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **p)
    if name == 'SVM':
        return SVC(probability=True, random_state=RANDOM_STATE, **p)
    if name == 'MLP':
        return MLPClassifier(random_state=RANDOM_STATE, **p)
    if name == 'LightGBM':
        return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1, **p)
    raise ValueError(f"Unknown classifier: {name}")


def make_sampler(name):
    if name == 'None' or name is None:
        return None
    if name == 'ROS':
        return RandomOverSampler(random_state=RANDOM_STATE)
    if name == 'SMOTE':
        return SMOTE(random_state=RANDOM_STATE)
    if name == 'BorderlineSMOTE':
        return BorderlineSMOTE(random_state=RANDOM_STATE)
    if name == 'ADASYN':
        return ADASYN(random_state=RANDOM_STATE)
    if name == 'RUS':
        return RandomUnderSampler(random_state=RANDOM_STATE)
    if name == 'NearMiss':
        return NearMiss(version=1)
    if name == 'TomekLinks':
        return TomekLinks()
    if name == 'SMOTETomek':
        return SMOTETomek(random_state=RANDOM_STATE)
    if name == 'SMOTEENN':
        return SMOTEENN(random_state=RANDOM_STATE)
    raise ValueError(f"Unknown sampler: {name}")


# ══════════════════════════════════════════════════════════════════════════
# THRESHOLD DERIVATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def find_youden_threshold(y_true, y_proba):
    """Youden's J: maximises sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def find_sensitivity_threshold(y_true, y_proba, min_sensitivity=MIN_SENSITIVITY):
    """Highest threshold achieving at least min_sensitivity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    valid = np.where(tpr >= min_sensitivity)[0]
    if len(valid) == 0:
        # Best achievable
        return float(thresholds[np.argmax(tpr)])
    # Highest threshold that still meets the criterion (most conservative)
    return float(thresholds[valid[0]])


def find_f1_threshold(y_true, y_proba):
    """Threshold maximising minority-class F1."""
    candidates = np.linspace(0.01, 0.99, 300)
    f1_scores = [
        f1_score(y_true, (y_proba >= t).astype(int),
                 pos_label=1, zero_division=0)
        for t in candidates
    ]
    return float(candidates[np.argmax(f1_scores)])


# ══════════════════════════════════════════════════════════════════════════
# METRICS AT A GIVEN THRESHOLD
# ══════════════════════════════════════════════════════════════════════════

def metrics_at_threshold(y_true, y_proba, threshold):
    """Return dict of metrics for a given decision threshold."""
    y_pred = (np.array(y_proba) >= threshold).astype(int)
    y_true = np.array(y_true)

    if y_true.sum() == 0:
        return None  # no positive cases — skip

    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    gmean = float(np.sqrt(sens * spec))
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        'threshold':       round(threshold, 4),
        'sensitivity':     round(sens, 3),
        'specificity':     round(spec, 3),
        'balanced_acc':    round(bal_acc, 3),
        'gmean':           round(gmean, 3),
        'auc':             round(auc, 3),
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
    }


# ══════════════════════════════════════════════════════════════════════════
# OOF CV — collect out-of-fold probabilities
# ══════════════════════════════════════════════════════════════════════════

def collect_oof_probabilities(clf_name, params, imb_name, X_train, y_train):
    """
    Run RepeatedStratifiedKFold with fixed params (no Optuna).
    Returns concatenated OOF y_true and y_proba arrays.
    """
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=RANDOM_STATE)
    sampler = make_sampler(imb_name)
    oof_true, oof_proba = [], []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        if sampler is not None:
            try:
                X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)
            except Exception:
                continue

        clf = make_classifier(clf_name, params)
        try:
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_val)[:, 1]
        except Exception:
            continue

        oof_true.extend(y_val.tolist())
        oof_proba.extend(prob.tolist())

    return np.array(oof_true), np.array(oof_proba)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE — threshold comparison table / heatmap
# ══════════════════════════════════════════════════════════════════════════

def plot_threshold_table(results_df):
    """
    For each classifier, plot a side-by-side bar chart of sensitivity
    and specificity under the 4 thresholds.
    Focus on MLP and LightGBM+ROS (the key contrast).
    """
    focus = ['MLP', 'LightGBM']
    threshold_labels = ['Default\n(0.5)', 'Youden', 'Sens-opt\n(≥0.90)', 'F1-opt']
    threshold_keys   = ['default', 'youden', 'sens_opt', 'f1_opt']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle('Decision Threshold Analysis — MLP vs LightGBM+ROS',
                 fontsize=13, fontweight='bold')

    metrics = ['sensitivity', 'specificity']
    colours = [PAL['short'], PAL['accent']]

    for ax, clf_name in zip(axes, focus):
        subset = results_df[results_df['classifier'] == clf_name]
        x = np.arange(len(threshold_labels))
        width = 0.35

        for i, (metric, colour) in enumerate(zip(metrics, colours)):
            vals = []
            for tk in threshold_keys:
                row = subset[subset['threshold_type'] == tk]
                vals.append(float(row[metric].values[0]) if len(row) else 0.0)

            bars = ax.bar(x + (i - 0.5) * width, vals, width,
                          label=metric.capitalize(), color=colour,
                          edgecolor='white', linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f'{v:.3f}', ha='center', va='bottom',
                        fontsize=7.5, fontweight='bold')

        imb = subset['imbalance_technique'].iloc[0] if len(subset) else ''
        ax.set_title(f'{clf_name} + {imb}', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(threshold_labels, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Score', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'step11_threshold_comparison.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig_threshold_comparison.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: step11_threshold_comparison.png")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_start = time.time()
    print("=" * 70)
    print("STEP 11 — THRESHOLD OPTIMISATION ANALYSIS")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────
    import numpy as np
    data = np.load(os.path.join(PROCESSED_DIR, 'step4_arrays_unscaled.npz'),
                   allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    print(f"\nTrain: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    print(f"Target: Short={y_train.sum():.0f}, Non-Short={(y_train==0).sum():.0f}")

    # ── Load best configs (step5) ──────────────────────────────────────
    with open(os.path.join(STEP5_DIR, 'step5_best_configs.json')) as f:
        best_configs = json.load(f)

    # ── Load test probabilities (step6) ───────────────────────────────
    prob_path = os.path.join(STEP6_DIR, 'step6_test_probabilities.json')
    if not os.path.exists(prob_path):
        raise FileNotFoundError(
            "step6_test_probabilities.json not found.\n"
            "Please re-run step6_test_evaluation.py first."
        )
    with open(prob_path) as f:
        test_probs = json.load(f)

    # ── Main loop ──────────────────────────────────────────────────────
    threshold_rows = []
    results_rows   = []

    for clf_name in CLASSIFIERS_ORDER:
        if clf_name not in best_configs:
            print(f"\n  ⚠ {clf_name}: no config, skipping")
            continue

        cfg      = best_configs[clf_name]
        params   = cfg['best_params']
        imb_name = cfg['imbalance_technique']

        print(f"\n{'─' * 70}")
        print(f"  {clf_name} + {imb_name}")

        # ── Step A: OOF probabilities → derive thresholds ─────────────
        print(f"    Collecting OOF probabilities ({N_SPLITS}×{N_REPEATS} CV)...", end='', flush=True)
        t0 = time.time()
        oof_true, oof_proba = collect_oof_probabilities(
            clf_name, params, imb_name, X_train, y_train
        )
        print(f" done ({time.time()-t0:.1f}s)")

        thresh_youden   = find_youden_threshold(oof_true, oof_proba)
        thresh_sens_opt = find_sensitivity_threshold(oof_true, oof_proba, MIN_SENSITIVITY)
        thresh_f1       = find_f1_threshold(oof_true, oof_proba)

        print(f"    Thresholds — Default:0.500  Youden:{thresh_youden:.3f}"
              f"  Sens-opt:{thresh_sens_opt:.3f}  F1-opt:{thresh_f1:.3f}")

        threshold_rows.append({
            'classifier':           clf_name,
            'imbalance_technique':  imb_name,
            'thresh_default':       0.500,
            'thresh_youden':        round(thresh_youden, 4),
            'thresh_sens_opt':      round(thresh_sens_opt, 4),
            'thresh_f1_opt':        round(thresh_f1, 4),
            'oof_n_samples':        len(oof_true),
            'oof_n_positive':       int(oof_true.sum()),
        })

        # ── Step B: Apply thresholds to held-out test set ─────────────
        if clf_name not in test_probs:
            print(f"    ⚠ No test probabilities for {clf_name}, skipping test evaluation")
            continue

        tp_data  = test_probs[clf_name]
        y_true   = np.array(tp_data['y_true'])
        y_proba  = np.array(tp_data['y_proba'])

        for thresh_type, thresh_val in [
            ('default',  0.500),
            ('youden',   thresh_youden),
            ('sens_opt', thresh_sens_opt),
            ('f1_opt',   thresh_f1),
        ]:
            m = metrics_at_threshold(y_true, y_proba, thresh_val)
            if m is None:
                continue
            row = {
                'classifier':          clf_name,
                'imbalance_technique': imb_name,
                'threshold_type':      thresh_type,
                **m,
            }
            results_rows.append(row)
            print(f"    [{thresh_type:10s}] thresh={thresh_val:.3f}"
                  f"  sens={m['sensitivity']:.3f}  spec={m['specificity']:.3f}"
                  f"  BA={m['balanced_acc']:.3f}  G-Mean={m['gmean']:.3f}"
                  f"  TP={m['TP']} FP={m['FP']} FN={m['FN']}")

    # ── Save CSVs ──────────────────────────────────────────────────────
    thresh_df  = pd.DataFrame(threshold_rows)
    results_df = pd.DataFrame(results_rows)

    thresh_df.to_csv(os.path.join(OUT_DIR, 'step11_thresholds.csv'), index=False)
    results_df.to_csv(os.path.join(OUT_DIR, 'step11_threshold_results.csv'), index=False)
    print(f"\n  ✓ Saved: step11_thresholds.csv")
    print(f"  ✓ Saved: step11_threshold_results.csv")

    # ── Plot ───────────────────────────────────────────────────────────
    plot_threshold_table(results_df)

    # ── Print LaTeX-ready table for paper (MLP + LightGBM focus) ──────
    print("\n" + "=" * 70)
    print("LATEX TABLE SNIPPET — MLP and LightGBM+ROS")
    print("=" * 70)
    focus_clfs = ['MLP', 'LightGBM']
    thresh_labels = {
        'default':  'Default (0.50)',
        'youden':   'Youden',
        'sens_opt': f'Sensitivity-opt. ($\\geq${MIN_SENSITIVITY})',
        'f1_opt':   'F1-optimal',
    }
    for clf in focus_clfs:
        imb = best_configs[clf]['imbalance_technique']
        print(f"\n  {clf} + {imb}")
        subset = results_df[results_df['classifier'] == clf]
        for tk in ['default', 'youden', 'sens_opt', 'f1_opt']:
            row = subset[subset['threshold_type'] == tk]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            print(f"    {thresh_labels[tk]:<40s}  "
                  f"thresh={r['threshold']:.3f}  "
                  f"sens={r['sensitivity']:.3f}  "
                  f"spec={r['specificity']:.3f}  "
                  f"BA={r['balanced_acc']:.3f}  "
                  f"G-Mean={r['gmean']:.3f}  "
                  f"TP={r['TP']} FN={r['FN']}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Step 11 completed in {elapsed:.1f}s")
    print(f"Outputs: {OUT_DIR}")
    print("=" * 70)
