"""
Step 7c — Bonferroni Statistical Comparison
=============================================
Replicates Papaiz et al. statistical analysis:
  - For each classifier: Wilcoxon signed-rank test comparing
    Single-Model vs Ensemble-Imbalance (BalancedBagging) per CV fold
  - Bonferroni correction (7 comparisons)
  - Generates figure with p-values annotated

Uses the same CV setup as step7b (5×3 RepeatedStratifiedKFold).
Does NOT modify any existing files — all outputs are new.
"""

import pandas as pd
import numpy as np
import os
import json
import time
import warnings

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import recall_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Configuration ──
BASE_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(BASE_DIR, '..', '01_data', 'processed')
STEP5_DIR = os.path.join(BASE_DIR, '..', '03_outputs', 'step5')
OUT_DIR = os.path.join(BASE_DIR, '..', '03_outputs', 'step7')
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_CLASSIFIERS = 7  # for Bonferroni correction


def make_estimator(clf_name, params):
    p = dict(params)
    if clf_name == 'KNN':
        return KNeighborsClassifier(**p)
    elif clf_name == 'NaiveBayes':
        return GaussianNB(**p)
    elif clf_name == 'DecisionTree':
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **p)
    elif clf_name == 'RandomForest':
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **p)
    elif clf_name == 'SVM':
        return SVC(probability=True, random_state=RANDOM_STATE, **p)
    elif clf_name == 'MLP':
        if isinstance(p.get('hidden_layer_sizes'), list):
            p['hidden_layer_sizes'] = tuple(p['hidden_layer_sizes'])
        return MLPClassifier(random_state=RANDOM_STATE, **p)
    elif clf_name == 'LightGBM':
        return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1,
                              n_jobs=-1, **p)


def run_bonferroni_analysis():
    start_time = time.time()

    print("=" * 70)
    print("STEP 7c — BONFERRONI STATISTICAL COMPARISON")
    print("=" * 70)

    # Load data
    data = np.load(os.path.join(PROCESSED_DIR, 'step4_arrays_unscaled.npz'),
                   allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']

    # Load baseline hyperparams
    cv_df = pd.read_csv(os.path.join(STEP5_DIR, 'step5_cv_results.csv'))
    cv_df['imbalance_technique'] = cv_df['imbalance_technique'].fillna('None')
    baseline_df = cv_df[cv_df['imbalance_technique'] == 'None']

    classifiers = ['KNN', 'NaiveBayes', 'DecisionTree', 'RandomForest',
                   'SVM', 'MLP', 'LightGBM']
    display_names = {
        'KNN': 'KNN', 'NaiveBayes': 'Naive Bayes',
        'DecisionTree': 'Decision Tree', 'RandomForest': 'Random Forest',
        'SVM': 'SVM', 'MLP': 'MLP (Neural Net)', 'LightGBM': 'LightGBM'
    }

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3,
                                 random_state=RANDOM_STATE)

    # ── Collect per-fold results ──
    fold_results = {}  # clf -> {metric -> {'single': [...], 'ensemble': [...]}}

    for clf_name in classifiers:
        row = baseline_df[baseline_df['classifier'] == clf_name].iloc[0]
        params = json.loads(row['best_params'])

        print(f"\n  CV for {clf_name}...")

        single = {'bal_acc': [], 'sensitivity': [], 'specificity': []}
        ensemble = {'bal_acc': [], 'sensitivity': [], 'specificity': []}

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            Xtr, Xval = X_train[train_idx], X_train[val_idx]
            ytr, yval = y_train[train_idx], y_train[val_idx]

            scaler = MinMaxScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xval = scaler.transform(Xval)

            # Single-Model
            est = make_estimator(clf_name, params)
            est.fit(Xtr, ytr)
            y_pred = est.predict(Xval)
            single['bal_acc'].append(balanced_accuracy_score(yval, y_pred))
            single['sensitivity'].append(recall_score(yval, y_pred, pos_label=1))
            single['specificity'].append(recall_score(yval, y_pred, pos_label=0))

            # BalancedBagging
            base = make_estimator(clf_name, params)
            bb = BalancedBaggingClassifier(
                estimator=base,
                n_estimators=10, random_state=RANDOM_STATE, n_jobs=-1
            )
            bb.fit(Xtr, ytr)
            y_pred_bb = bb.predict(Xval)
            ensemble['bal_acc'].append(balanced_accuracy_score(yval, y_pred_bb))
            ensemble['sensitivity'].append(recall_score(yval, y_pred_bb, pos_label=1))
            ensemble['specificity'].append(recall_score(yval, y_pred_bb, pos_label=0))

            if (fold_idx + 1) % 5 == 0:
                print(f"    fold {fold_idx + 1}/15 done")

        fold_results[clf_name] = {'single': single, 'ensemble': ensemble}

    # ── Statistical tests ──
    print(f"\n{'─' * 70}")
    print("  WILCOXON SIGNED-RANK TESTS + BONFERRONI CORRECTION")
    print(f"{'─' * 70}")

    metrics = ['bal_acc', 'sensitivity', 'specificity']
    metric_display = {
        'bal_acc': 'Balanced Accuracy',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity'
    }

    stat_rows = []
    for clf_name in classifiers:
        fr = fold_results[clf_name]
        clf_stats = {'classifier': clf_name}

        for metric in metrics:
            s = np.array(fr['single'][metric])
            e = np.array(fr['ensemble'][metric])
            diff = e - s

            mean_s = np.mean(s)
            mean_e = np.mean(e)
            mean_diff = np.mean(diff)

            # Wilcoxon signed-rank test (paired)
            # If all differences are 0, p=1
            if np.all(diff == 0):
                p_raw = 1.0
                stat = 0.0
            else:
                try:
                    stat, p_raw = wilcoxon(s, e, alternative='two-sided')
                except ValueError:
                    # All zeros or too few samples
                    p_raw = 1.0
                    stat = 0.0

            # Bonferroni correction
            p_corrected = min(p_raw * N_CLASSIFIERS, 1.0)

            clf_stats[f'{metric}_single_mean'] = mean_s
            clf_stats[f'{metric}_ens_mean'] = mean_e
            clf_stats[f'{metric}_diff'] = mean_diff
            clf_stats[f'{metric}_p_raw'] = p_raw
            clf_stats[f'{metric}_p_bonferroni'] = p_corrected
            clf_stats[f'{metric}_significant'] = p_corrected <= 0.05

            sig_str = "***" if p_corrected <= 0.001 else \
                      "**" if p_corrected <= 0.01 else \
                      "*" if p_corrected <= 0.05 else "n.s."

            print(f"  {clf_name:>14} | {metric_display[metric]:>18} | "
                  f"Single={mean_s:.4f} Ens={mean_e:.4f} "
                  f"Δ={mean_diff:+.4f} | p(raw)={p_raw:.6f} "
                  f"p(Bonf)={p_corrected:.6f} {sig_str}")

        stat_rows.append(clf_stats)

    stats_df = pd.DataFrame(stat_rows)
    csv_path = os.path.join(OUT_DIR, 'step7c_bonferroni_results.csv')
    stats_df.to_csv(csv_path, index=False)
    print(f"\n  ✓ {os.path.basename(csv_path)}")

    # ── Generate figure with p-values ──
    metric_labels = ['Balanced\nAccuracy', 'Sensitivity', 'Specificity']

    fig, axes = plt.subplots(2, 4, figsize=(24, 13))
    axes_flat = axes.flatten()

    x = np.arange(len(metric_labels))
    width = 0.32

    color_single = '#f4a742'
    color_ensemble = '#2171b5'

    for idx, clf_name in enumerate(classifiers):
        ax = axes_flat[idx]
        fr = fold_results[clf_name]
        sr = stats_df[stats_df['classifier'] == clf_name].iloc[0]

        single_means = [np.mean(fr['single'][m]) for m in metrics]
        ens_means = [np.mean(fr['ensemble'][m]) for m in metrics]
        single_stds = [np.std(fr['single'][m]) for m in metrics]
        ens_stds = [np.std(fr['ensemble'][m]) for m in metrics]

        bars1 = ax.bar(x - width/2 - 0.02, single_means, width,
                       yerr=single_stds, capsize=3,
                       label='Single-Model', color=color_single,
                       edgecolor='white', zorder=3, error_kw={'linewidth': 1})
        bars2 = ax.bar(x + width/2 + 0.02, ens_means, width,
                       yerr=ens_stds, capsize=3,
                       label='Ensemble-Imbalance', color=color_ensemble,
                       edgecolor='white', zorder=3, error_kw={'linewidth': 1})

        # Annotate mean values
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold', color='#7a5500')
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold', color='#08306b')

        # p-value annotations
        for j, metric in enumerate(metrics):
            p_bonf = sr[f'{metric}_p_bonferroni']
            sig = sr[f'{metric}_significant']

            if p_bonf <= 0.001:
                p_label = 'p<0.001 ***'
            elif p_bonf <= 0.01:
                p_label = f'p={p_bonf:.3f} **'
            elif p_bonf <= 0.05:
                p_label = f'p={p_bonf:.3f} *'
            else:
                p_label = f'p={p_bonf:.3f} n.s.'

            y_top = max(single_means[j] + single_stds[j],
                        ens_means[j] + ens_stds[j]) + 0.06

            color_p = '#d62728' if sig else '#888888'
            fontw = 'bold' if sig else 'normal'

            ax.text(x[j], y_top, p_label, ha='center', fontsize=7,
                    fontweight=fontw, color=color_p,
                    bbox=dict(boxstyle='round,pad=0.12',
                              facecolor='white', edgecolor=color_p,
                              alpha=0.85, linewidth=0.7))

        # Check if ensemble significantly better overall (bal_acc)
        is_sig = sr['bal_acc_significant']
        title_color = '#08519c' if is_sig else '#333333'

        ax.set_title(display_names[clf_name], fontsize=12,
                     fontweight='bold', color=title_color)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9.5, fontweight='bold')
        ax.set_ylim(0, 1.32)
        ax.set_ylabel('Performance', fontsize=9)
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide 8th subplot
    axes_flat[7].set_visible(False)

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_single,
                       edgecolor='white', label='Single-Model'),
        plt.Rectangle((0, 0), 1, 1, facecolor=color_ensemble,
                       edgecolor='white', label='Ensemble-Imbalance (BalancedBagging)'),
    ]
    fig.legend(handles=handles, loc='lower right',
               bbox_to_anchor=(0.97, 0.06), fontsize=12, frameon=True,
               fancybox=True, shadow=True)

    fig.suptitle(
        'Single-Model vs Ensemble-Imbalance — Wilcoxon + Bonferroni Correction\n'
        '(error bars = ±1 std across 15 CV folds)',
        fontsize=15, fontweight='bold', y=1.02
    )
    fig.text(0.5, -0.01,
             '*** p≤0.001  |  ** p≤0.01  |  * p≤0.05  |  n.s. = not significant  '
             '(Bonferroni-corrected for 7 comparisons)',
             ha='center', fontsize=10, style='italic', color='#555555')

    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'step7c_bonferroni_comparison.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {os.path.basename(fig_path)}")

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Step 7c completed in {total_time:.1f}s")
    print(f"{'=' * 70}")

    return stats_df


if __name__ == '__main__':
    run_bonferroni_analysis()
