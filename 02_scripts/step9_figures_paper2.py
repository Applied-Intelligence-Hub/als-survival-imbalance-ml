"""
Step 9 — Additional Publication Figures (Paper 2)
==================================================
Generates:
  1. fig_pr_curves.png       — Precision-Recall curves for 3 key models
                               (LightGBM+ROS, SVM+SMOTEENN, MLP+None)
  2. fig_feature_bars.png    — Distribution of Diagnosis_Delay and
                               Q8_Walking_slope by Survival Group

Outputs saved to:
  - 03_outputs/step9/
  - overleaf/figures/
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score

from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN

warnings.filterwarnings('ignore')
matplotlib.rcParams.update({'font.size': 10})

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), '..')
PROCESSED   = os.path.join(BASE_DIR, '01_data', 'processed')
CONFIGS_JSON = os.path.join(BASE_DIR, '03_outputs', 'step5', 'step5_best_configs.json')
OUT_DIR     = os.path.join(BASE_DIR, '03_outputs', 'step9')
FIG_DIR     = os.path.join(BASE_DIR, 'overleaf', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE = 42

# ── Load train / test data ─────────────────────────────────────────────────
train = pd.read_csv(os.path.join(PROCESSED, 'step4_train.csv'))
test  = pd.read_csv(os.path.join(PROCESSED, 'step4_test.csv'))

FEATURE_COLS = [c for c in train.columns if c not in ('subject_id', 'Survival_Group')]

X_train = train[FEATURE_COLS].values
y_train = train['Survival_Group'].values
X_test  = test[FEATURE_COLS].values
y_test  = test['Survival_Group'].values

# ── Load best configs ──────────────────────────────────────────────────────
with open(CONFIGS_JSON) as f:
    configs = json.load(f)


# ══════════════════════════════════════════════════════════════════════════
# HELPER — build sampler
# ══════════════════════════════════════════════════════════════════════════
def get_sampler(name):
    if name == 'None':
        return None
    if name == 'ROS':
        return RandomOverSampler(random_state=RANDOM_STATE)
    if name == 'SMOTEENN':
        return SMOTEENN(random_state=RANDOM_STATE)
    raise ValueError(f"Unknown sampler: {name}")


# ══════════════════════════════════════════════════════════════════════════
# HELPER — build classifier
# ══════════════════════════════════════════════════════════════════════════
def build_clf(name, params):
    if name == 'LightGBM':
        return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, **params)
    if name == 'SVM':
        return SVC(probability=True, random_state=RANDOM_STATE, **params)
    if name == 'MLP':
        p = dict(params)
        p['hidden_layer_sizes'] = tuple(p['hidden_layer_sizes'])
        return MLPClassifier(random_state=RANDOM_STATE, **p)
    raise ValueError(f"Unknown classifier: {name}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Precision-Recall Curves
# ══════════════════════════════════════════════════════════════════════════
def plot_pr_curves():
    print("Generating PR curves...")

    MODELS = [
        ('LightGBM', 'ROS',      '#2196F3', 'LightGBM + ROS'),
        ('SVM',      'SMOTEENN', '#FF9800', 'SVM + SMOTEENN'),
        ('MLP',      'None',     '#9C27B0', 'MLP + None'),
    ]

    # Baseline (random classifier) AP = prevalence
    prevalence = y_test.mean()

    fig, ax = plt.subplots(figsize=(7, 5))

    for clf_name, sampler_name, colour, label in MODELS:
        params     = configs[clf_name]['best_params']
        best_tech  = configs[clf_name]['imbalance_technique']

        # Override sampler to the one we want for these 3 key models
        sampler = get_sampler(sampler_name)
        clf     = build_clf(clf_name, params)

        # Resample + train
        if sampler is not None:
            Xr, yr = sampler.fit_resample(X_train, y_train)
        else:
            Xr, yr = X_train, y_train

        clf.fit(Xr, yr)
        y_prob = clf.predict_proba(X_test)[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)

        ax.plot(recall, precision, colour, linewidth=2,
                label=f'{label}  (AP = {ap:.3f})')

    # Baseline
    ax.axhline(prevalence, color='grey', linestyle='--', linewidth=1,
               label=f'Random classifier (AP = {prevalence:.3f})')

    ax.set_xlabel('Recall (Sensitivity)', fontsize=11)
    ax.set_ylabel('Precision (PPV)', fontsize=11)
    ax.set_title('Precision-Recall Curves — Held-Out Test Set', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for dest in (OUT_DIR, FIG_DIR):
        path = os.path.join(dest, 'fig_pr_curves.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Feature Distribution Bars (Diagnosis_Delay & Q8_Walking_slope)
# ══════════════════════════════════════════════════════════════════════════
def plot_feature_bars():
    print("Generating feature distribution bars...")

    # Use the unscaled coded dataset (ordinal integers, train+test combined)
    unscaled = pd.read_csv(os.path.join(PROCESSED, 'step3_coded_unscaled.csv'))
    unscaled['Group'] = unscaled['Survival_Group'].map(
        {1: 'Short Survivor', 0: 'Standard Survival'}
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    colours = {'Short Survivor': '#E53935', 'Standard Survival': '#1E88E5'}

    # ── Panel A: Diagnosis_Delay  (0=Short≤8mo, 1=Average 9–18mo, 2=Long≥19mo)
    ax = axes[0]
    delay_labels = {0: 'Short\n(≤8 mo)', 1: 'Average\n(9–18 mo)', 2: 'Long\n(≥19 mo)'}
    categories   = [0, 1, 2]
    x            = np.arange(len(categories))
    width        = 0.35

    for i, (group, colour) in enumerate(colours.items()):
        sub    = unscaled[unscaled['Group'] == group]
        counts = sub['Diagnosis_Delay'].value_counts().reindex(categories, fill_value=0)
        total  = len(sub)
        pct    = counts / total * 100
        bars   = ax.bar(x + (i - 0.5) * width, pct, width, label=group, color=colour,
                        alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, pct):
            if val > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([delay_labels[c] for c in categories])
    ax.set_ylabel('Patients (%)', fontsize=10)
    ax.set_title('Diagnosis Delay', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis='y', alpha=0.3)

    # ── Panel B: Q8_Walking_slope  (0=Slow<0.05, 1=Average 0.05–0.13, 2=Rapid≥0.14)
    ax = axes[1]
    slope_labels = {0: 'Slow\n(<0.05)', 1: 'Average\n(0.05–0.13)', 2: 'Rapid\n(≥0.14)'}

    for i, (group, colour) in enumerate(colours.items()):
        sub    = unscaled[unscaled['Group'] == group]
        counts = sub['Q8_Walking_slope'].value_counts().reindex(categories, fill_value=0)
        total  = len(sub)
        pct    = counts / total * 100
        bars   = ax.bar(x + (i - 0.5) * width, pct, width, label=group, color=colour,
                        alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, pct):
            if val > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([slope_labels[c] for c in categories])
    ax.set_ylabel('Patients (%)', fontsize=10)
    ax.set_title('Q8 Walking Slope', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Feature Distributions by Survival Group', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    for dest in (OUT_DIR, FIG_DIR):
        path = os.path.join(dest, 'fig_feature_bars.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    plot_pr_curves()
    plot_feature_bars()
    print("\nDone.")
