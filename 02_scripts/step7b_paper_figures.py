"""
Step 7b — Paper-style SHAP Figures
====================================
Replicate figures similar to Papaiz et al.:
  Figure 3: Combined Feature Importance bar + Beeswarm (side by side)
  Figure 4: SHAP Decision Plots (individual patient trajectories)

Generates both for LightGBM+ROS and MLP+None.
"""

import pandas as pd
import numpy as np
import os
import json
import time
import warnings

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(BASE_DIR, '..', '01_data', 'processed')
STEP5_DIR = os.path.join(BASE_DIR, '..', '03_outputs', 'step5')
OUT_DIR = os.path.join(BASE_DIR, '..', '03_outputs', 'step7')
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42


def build_models_and_shap(X_train, y_train, X_test, y_test, feature_names,
                          best_configs):
    """Train both models and compute SHAP values."""
    models = {}

    # ── LightGBM + ROS ──
    cfg = best_configs['LightGBM']
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_tr, y_tr = ros.fit_resample(X_train, y_train)
    lgbm = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
                           **cfg['best_params'])
    lgbm.fit(X_tr, y_tr)

    explainer = shap.TreeExplainer(lgbm)
    sv = explainer.shap_values(X_test)
    if isinstance(sv, list):
        sv = sv[1]
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1]

    models['LightGBM'] = {
        'model': lgbm, 'shap_values': sv,
        'base_value': base_val, 'label': 'LightGBM + ROS'
    }

    # ── MLP + None ──
    cfg_mlp = best_configs['MLP']
    params_mlp = dict(cfg_mlp['best_params'])
    if isinstance(params_mlp.get('hidden_layer_sizes'), list):
        params_mlp['hidden_layer_sizes'] = tuple(params_mlp['hidden_layer_sizes'])
    mlp = MLPClassifier(random_state=RANDOM_STATE, **params_mlp)
    mlp.fit(X_train, y_train)

    background = shap.kmeans(X_train, 100)
    explainer_mlp = shap.KernelExplainer(
        lambda X: mlp.predict_proba(X)[:, 1], background
    )
    print("  Computing MLP SHAP values...")
    sv_mlp = explainer_mlp.shap_values(X_test, nsamples=200)
    base_val_mlp = explainer_mlp.expected_value

    models['MLP'] = {
        'model': mlp, 'shap_values': sv_mlp,
        'base_value': base_val_mlp, 'label': 'MLP'
    }

    return models


def plot_fig3_combined(shap_vals, X_data, feature_names, base_value,
                       model_label, out_path):
    """
    Paper Figure 3 style: Feature Importance bar (left) + Beeswarm (right)
    Side-by-side in one figure.
    """
    # Sort features by mean |SHAP|
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)  # ascending for horizontal bar

    sorted_features = [feature_names[i] for i in order]
    sorted_shap = shap_vals[:, order]
    sorted_X = X_data[:, order]
    sorted_mean = mean_abs[order]

    n_feat = len(feature_names)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(1, 2, width_ratios=[1, 1.4], wspace=0.05, figure=fig)

    # ── LEFT: Horizontal bar chart (mean |SHAP|) ──
    ax_bar = fig.add_subplot(gs[0])
    bars = ax_bar.barh(range(n_feat), sorted_mean, color='#4292c6',
                       edgecolor='white', height=0.7)
    ax_bar.set_yticks(range(n_feat))
    ax_bar.set_yticklabels(sorted_features, fontsize=9)
    ax_bar.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax_bar.set_title('Feature Importance', fontsize=13, fontweight='bold')
    ax_bar.invert_xaxis()  # values grow to the left

    # Annotate values
    max_val = sorted_mean.max()
    for i, (bar, v) in enumerate(zip(bars, sorted_mean)):
        ax_bar.text(v + max_val * 0.02, i, f'{v:.3f}', va='center',
                    ha='left', fontsize=7.5)

    # ── RIGHT: Beeswarm (dot plot) ──
    ax_bee = fig.add_subplot(gs[1], sharey=ax_bar)

    # Create the Explanation object for beeswarm
    explanation = shap.Explanation(
        values=sorted_shap,
        base_values=np.full(sorted_shap.shape[0], base_value),
        data=sorted_X,
        feature_names=sorted_features
    )

    plt.sca(ax_bee)
    shap.plots.beeswarm(explanation, max_display=n_feat, show=False,
                        color_bar=True)
    ax_bee.set_ylabel('')
    ax_bee.tick_params(axis='y', labelleft=False)
    ax_bee.set_title('SHAP Value Distribution', fontsize=13, fontweight='bold')

    fig.suptitle(f'SHAP Feature Analysis — {model_label}',
                 fontsize=15, fontweight='bold', y=1.01)

    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {os.path.basename(out_path)}")


def plot_fig4_decision(shap_vals, X_data, y_true, feature_names, base_value,
                       model_label, out_path):
    """
    Paper Figure 4 style: SHAP Decision Plot.
    Shows trajectories of individual predictions from base value to output.
    Highlights Short (red) and Non-Short (blue) patients.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Color map: Short = red, Non-Short = blue
    short_mask = y_true == 1
    n_short = short_mask.sum()
    n_nonshort = (~short_mask).sum()

    # All patients, colored by class
    # Use shap.decision_plot which supports highlighting
    plt.sca(ax)
    shap.decision_plot(
        base_value,
        shap_vals,
        features=X_data,
        feature_names=feature_names,
        highlight=np.where(short_mask)[0],
        show=False,
        feature_order='importance',
        return_objects=False,
        color_bar=False,
        plot_color='coolwarm',
        alpha=0.3,
    )

    ax.set_title(f'SHAP Decision Plot — {model_label}\n'
                 f'({n_short} Short patients highlighted)',
                 fontsize=14, fontweight='bold')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {os.path.basename(out_path)}")


def plot_fig4_decision_cases(shap_vals, X_data, y_true, y_pred,
                             feature_names, base_value, model_label, out_path):
    """
    Decision plot with 4 specific case studies:
    TP, FP, FN, TN — each in a subplot.
    """
    tn_mask = (y_true == 0) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tp_mask = (y_true == 1) & (y_pred == 1)

    cases = [
        ('True Positive (Short → Short)', tp_mask, '#d62728'),
        ('False Positive (Non-Short → Short)', fp_mask, '#ff7f0e'),
        ('False Negative (Short → Non-Short)', fn_mask, '#9467bd'),
        ('True Negative (Non-Short → Non-Short)', tn_mask, '#2ca02c'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    for idx, (title, mask, color) in enumerate(cases):
        ax = axes[idx // 2, idx % 2]
        indices = np.where(mask)[0]

        if len(indices) == 0:
            ax.text(0.5, 0.5, f'{title}\n(no samples)', ha='center',
                    va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title(title, fontsize=12, fontweight='bold')
            continue

        # Pick the most "extreme" case (highest absolute SHAP sum)
        shap_sums = np.abs(shap_vals[indices]).sum(axis=1)
        pick = indices[np.argmax(shap_sums)]

        plt.sca(ax)
        shap.decision_plot(
            base_value,
            shap_vals[pick:pick+1],
            features=X_data[pick:pick+1],
            feature_names=feature_names,
            show=False,
            feature_order='importance',
            return_objects=False,
            link='identity',
        )
        ax.set_title(f'{title}\n(patient #{pick})', fontsize=11,
                     fontweight='bold')

    fig.suptitle(f'SHAP Decision Plots — Case Studies ({model_label})',
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {os.path.basename(out_path)}")


def plot_paper_fig3_metrics(cv_results_path, out_path):
    """
    Paper Figure 3 style: One subplot per classifier (2 rows × 4 cols),
    each with 3 metric groups (Bal. Accuracy, Sensitivity, Specificity),
    Baseline vs Best Imbalance side by side. Values annotated on bars.
    """
    df = pd.read_csv(cv_results_path)
    df['imbalance_technique'] = df['imbalance_technique'].fillna('None')

    classifiers = ['KNN', 'NaiveBayes', 'DecisionTree', 'RandomForest',
                   'SVM', 'MLP', 'LightGBM']
    display_names = {
        'KNN': 'KNN', 'NaiveBayes': 'Naive Bayes',
        'DecisionTree': 'Decision Tree', 'RandomForest': 'Random Forest',
        'SVM': 'SVM', 'MLP': 'MLP', 'LightGBM': 'LightGBM'
    }

    # Collect baseline and best values per classifier
    rows = []
    for clf in classifiers:
        sub = df[df['classifier'] == clf]
        baseline = sub[sub['imbalance_technique'] == 'None']
        best = sub.loc[sub['mean_auc'].idxmax()]

        bal_recall = baseline['mean_recall'].values[0] if not baseline.empty else np.nan
        bal_spec = baseline['mean_specificity'].values[0] if not baseline.empty else np.nan
        bal_acc_base = (bal_recall + bal_spec) / 2

        best_recall = best['mean_recall']
        best_spec = best['mean_specificity']
        bal_acc_best = (best_recall + best_spec) / 2
        best_tech = best['imbalance_technique']

        rows.append({
            'classifier': clf,
            'best_technique': best_tech if best_tech != 'None' else '—',
            'base_vals': [bal_acc_base, bal_recall, bal_spec],
            'best_vals': [bal_acc_best, best_recall, best_spec],
            'improved': best_tech != 'None',
        })

    metric_labels = ['Balanced\nAccuracy', 'Sensitivity', 'Specificity']

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes_flat = axes.flatten()

    x = np.arange(len(metric_labels))
    width = 0.32

    color_base = '#6baed6'
    color_best = '#08519c'

    for idx, row in enumerate(rows):
        ax = axes_flat[idx]
        base = np.array(row['base_vals'])
        best = np.array(row['best_vals'])

        bars1 = ax.bar(x - width/2, base, width, label='Baseline',
                       color=color_base, edgecolor='white', zorder=3)
        bars2 = ax.bar(x + width/2, best, width, label='Best Imbalance',
                       color=color_best, edgecolor='white', zorder=3)

        # Annotate values
        for bar in bars1:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=8.5,
                        fontweight='bold', color='#444444')
        for bar in bars2:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=8.5,
                        fontweight='bold', color='#08306b')

        # Star if imbalance improved
        if row['improved']:
            for j in range(3):
                diff = best[j] - base[j]
                if diff > 0.01:
                    y_top = max(base[j], best[j]) + 0.07
                    ax.text(x[j], y_top, '\u2605', ha='center', fontsize=11,
                            color='#d62728')

        title = display_names[row['classifier']]
        tech_label = row['best_technique']
        if row['improved']:
            title += f'\n(best: {tech_label})'

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel('Performance', fontsize=9)
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide the 8th subplot (2×4 = 8, but only 7 classifiers)
    axes_flat[7].set_visible(False)

    # Legend in the empty 8th cell area
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_base, edgecolor='white',
                       label='Baseline (no imbalance)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=color_best, edgecolor='white',
                       label='Best Imbalance technique'),
    ]
    fig.legend(handles=handles, loc='lower right',
               bbox_to_anchor=(0.96, 0.08), fontsize=12, frameon=True,
               fancybox=True, shadow=True)

    fig.suptitle(
        'Comparison of Best Performances — Baseline vs Best Imbalance Scenario',
        fontsize=16, fontweight='bold', y=1.01
    )
    fig.text(0.5, -0.01,
             '\u2605 = Imbalance technique improved metric vs Baseline',
             ha='center', fontsize=10, style='italic', color='#555555')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  \u2713 {os.path.basename(out_path)}")


def plot_paper_fig3_balanced_bagging(out_path):
    """
    Exact replication of Papaiz et al. Figure 3:
    Single-Model (yellow) vs BalancedBagging Ensemble-Imbalance (blue).
    Trains each classifier with CV and computes BalAcc, Sensitivity, Specificity.
    """
    from imblearn.ensemble import BalancedBaggingClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import recall_score, balanced_accuracy_score
    from sklearn.preprocessing import MinMaxScaler

    # Load data
    data_arrays = np.load(os.path.join(PROCESSED_DIR, 'step4_arrays_unscaled.npz'),
                          allow_pickle=True)
    X_train = data_arrays['X_train']
    y_train = data_arrays['y_train']

    # Load baseline (None) hyperparams from CV results
    cv_df = pd.read_csv(
        os.path.join(STEP5_DIR, 'step5_cv_results.csv'))
    cv_df['imbalance_technique'] = cv_df['imbalance_technique'].fillna('None')
    baseline_df = cv_df[cv_df['imbalance_technique'] == 'None']

    classifiers_order = ['KNN', 'NaiveBayes', 'DecisionTree', 'RandomForest',
                         'SVM', 'MLP', 'LightGBM']
    display_names = {
        'KNN': 'KNN', 'NaiveBayes': 'Naive Bayes',
        'DecisionTree': 'Decision Tree', 'RandomForest': 'Random Forest',
        'SVM': 'SVM', 'MLP': 'MLP (Neural Net)', 'LightGBM': 'LightGBM'
    }

    def make_estimator(clf_name, params):
        p = dict(params)
        if clf_name == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(**p)
        elif clf_name == 'NaiveBayes':
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB(**p)
        elif clf_name == 'DecisionTree':
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(random_state=RANDOM_STATE, **p)
        elif clf_name == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
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

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3,
                                 random_state=RANDOM_STATE)

    results = []
    for clf_name in classifiers_order:
        row = baseline_df[baseline_df['classifier'] == clf_name].iloc[0]
        params = json.loads(row['best_params'])

        print(f"  CV for {clf_name}...")

        single_bal, single_sens, single_spec = [], [], []
        ens_bal, ens_sens, ens_spec = [], [], []

        for train_idx, val_idx in cv.split(X_train, y_train):
            Xtr, Xval = X_train[train_idx], X_train[val_idx]
            ytr, yval = y_train[train_idx], y_train[val_idx]

            scaler = MinMaxScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xval = scaler.transform(Xval)

            # Single-Model
            est = make_estimator(clf_name, params)
            est.fit(Xtr, ytr)
            y_pred = est.predict(Xval)
            single_bal.append(balanced_accuracy_score(yval, y_pred))
            single_sens.append(recall_score(yval, y_pred, pos_label=1))
            single_spec.append(recall_score(yval, y_pred, pos_label=0))

            # BalancedBagging Ensemble
            base = make_estimator(clf_name, params)
            bb = BalancedBaggingClassifier(
                estimator=base,
                n_estimators=10, random_state=RANDOM_STATE, n_jobs=-1
            )
            bb.fit(Xtr, ytr)
            y_pred_bb = bb.predict(Xval)
            ens_bal.append(balanced_accuracy_score(yval, y_pred_bb))
            ens_sens.append(recall_score(yval, y_pred_bb, pos_label=1))
            ens_spec.append(recall_score(yval, y_pred_bb, pos_label=0))

        results.append({
            'classifier': clf_name,
            'single_bal_acc': np.mean(single_bal),
            'single_sensitivity': np.mean(single_sens),
            'single_specificity': np.mean(single_spec),
            'ens_bal_acc': np.mean(ens_bal),
            'ens_sensitivity': np.mean(ens_sens),
            'ens_specificity': np.mean(ens_spec),
        })

    data = pd.DataFrame(results)

    # Save CSV as well
    csv_path = out_path.replace('.png', '.csv')
    data.to_csv(csv_path, index=False)
    print(f"  \u2713 {os.path.basename(csv_path)}")

    # ── Plot: one subplot per classifier ──
    metric_labels = ['Balanced\nAccuracy', 'Sensitivity', 'Specificity']

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes_flat = axes.flatten()

    x = np.arange(len(metric_labels))
    width = 0.32

    color_single = '#f4a742'    # yellow/orange (paper: Single-Model)
    color_ensemble = '#2171b5'  # blue (paper: Ensemble-Imbalance)

    for idx, row in data.iterrows():
        ax = axes_flat[idx]
        single = [row['single_bal_acc'], row['single_sensitivity'],
                  row['single_specificity']]
        ensemble = [row['ens_bal_acc'], row['ens_sensitivity'],
                    row['ens_specificity']]

        bars1 = ax.bar(x - width/2 - 0.02, single, width,
                       label='Single-Model', color=color_single,
                       edgecolor='white', zorder=3)
        bars2 = ax.bar(x + width/2 + 0.02, ensemble, width,
                       label='Ensemble-Imbalance', color=color_ensemble,
                       edgecolor='white', zorder=3)

        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8.5,
                    fontweight='bold', color='#7a5500')
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8.5,
                    fontweight='bold', color='#08306b')

        # Delta badges
        for j in range(3):
            diff = ensemble[j] - single[j]
            if abs(diff) > 0.005:
                y_top = max(single[j], ensemble[j]) + 0.06
                sign = '+' if diff > 0 else ''
                color_d = '#2a9d2a' if diff > 0 else '#d62728'
                ax.text(x[j], y_top, f'{sign}{diff:.2f}',
                        ha='center', fontsize=7.5, fontweight='bold',
                        color=color_d,
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', edgecolor=color_d,
                                  alpha=0.8, linewidth=0.8))

        clf_name = display_names[row['classifier']]
        ax.set_title(clf_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9.5, fontweight='bold')
        ax.set_ylim(0, 1.22)
        ax.set_ylabel('Performance', fontsize=9)
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide 8th subplot, put legend there
    axes_flat[7].set_visible(False)
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
        'Single-Model vs Ensemble-Imbalance (BalancedBagging) — '
        'Replication of Papaiz et al. Figure 3',
        fontsize=15, fontweight='bold', y=1.01
    )
    fig.text(0.5, -0.01,
             'Green deltas = ensemble improvement  |  '
             'Red deltas = ensemble trade-off',
             ha='center', fontsize=10, style='italic', color='#555555')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  \u2713 {os.path.basename(out_path)}")


def plot_paper_fig3_metrics_v2(cv_results_path, out_path):
    """
    Paper Figure 3 v2: One subplot per classifier (2 rows × 4 cols).
    When best == baseline, show a single centered bar (no duplication).
    When best != baseline, show paired bars with delta annotation.
    """
    df = pd.read_csv(cv_results_path)
    df['imbalance_technique'] = df['imbalance_technique'].fillna('None')

    classifiers = ['KNN', 'NaiveBayes', 'DecisionTree', 'RandomForest',
                   'SVM', 'MLP', 'LightGBM']
    display_names = {
        'KNN': 'KNN', 'NaiveBayes': 'Naive Bayes',
        'DecisionTree': 'Decision Tree', 'RandomForest': 'Random Forest',
        'SVM': 'SVM', 'MLP': 'MLP', 'LightGBM': 'LightGBM'
    }

    rows = []
    for clf in classifiers:
        sub = df[df['classifier'] == clf]
        baseline = sub[sub['imbalance_technique'] == 'None']
        best = sub.loc[sub['mean_auc'].idxmax()]

        bal_recall = baseline['mean_recall'].values[0] if not baseline.empty else np.nan
        bal_spec = baseline['mean_specificity'].values[0] if not baseline.empty else np.nan

        best_recall = best['mean_recall']
        best_spec = best['mean_specificity']
        best_tech = best['imbalance_technique']

        rows.append({
            'classifier': clf,
            'best_technique': best_tech,
            'base_vals': [(bal_recall + bal_spec) / 2, bal_recall, bal_spec],
            'best_vals': [(best_recall + best_spec) / 2, best_recall, best_spec],
            'improved': best_tech != 'None',
        })

    metric_labels = ['Balanced\nAccuracy', 'Sensitivity', 'Specificity']

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes_flat = axes.flatten()

    x = np.arange(len(metric_labels))
    width = 0.30

    color_base = '#6baed6'
    color_best = '#08519c'
    color_single = '#4292c6'

    for idx, row in enumerate(rows):
        ax = axes_flat[idx]
        base = np.array(row['base_vals'])
        best = np.array(row['best_vals'])
        improved = row['improved']

        if not improved:
            # Single centered bars — baseline IS the best
            bars = ax.bar(x, base, width * 1.3, color=color_single,
                          edgecolor='white', zorder=3)
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                            f'{h:.2f}', ha='center', va='bottom', fontsize=9,
                            fontweight='bold', color='#333333')

            title = display_names[row['classifier']]
            title += '\n(Baseline = Best)'
            ax.set_title(title, fontsize=11, fontweight='bold', color='#555555')
        else:
            # Paired bars
            bars1 = ax.bar(x - width/2 - 0.02, base, width,
                           label='Baseline', color=color_base,
                           edgecolor='white', zorder=3)
            bars2 = ax.bar(x + width/2 + 0.02, best, width,
                           label='Best Imbalance', color=color_best,
                           edgecolor='white', zorder=3)

            for bar in bars1:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                            f'{h:.2f}', ha='center', va='bottom', fontsize=8.5,
                            fontweight='bold', color='#555555')
            for bar in bars2:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                            f'{h:.2f}', ha='center', va='bottom', fontsize=8.5,
                            fontweight='bold', color='#08306b')

            # Delta annotations for improved metrics
            for j in range(3):
                diff = best[j] - base[j]
                if abs(diff) > 0.005:
                    y_top = max(base[j], best[j]) + 0.065
                    sign = '+' if diff > 0 else ''
                    color_d = '#2a9d2a' if diff > 0 else '#d62728'
                    ax.text(x[j], y_top, f'{sign}{diff:.2f}',
                            ha='center', fontsize=8, fontweight='bold',
                            color=color_d,
                            bbox=dict(boxstyle='round,pad=0.15',
                                      facecolor='white', edgecolor=color_d,
                                      alpha=0.8, linewidth=0.8))

            tech = row['best_technique']
            title = display_names[row['classifier']]
            title += f'\n(best: {tech})'
            ax.set_title(title, fontsize=11, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9.5, fontweight='bold')
        ax.set_ylim(0, 1.22)
        ax.set_ylabel('Performance', fontsize=9)
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide the 8th subplot
    axes_flat[7].set_visible(False)

    # Legend in the empty 8th subplot area
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_single, edgecolor='white',
                       label='Baseline = Best (no imbalance improved AUC)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=color_base, edgecolor='white',
                       label='Baseline (no imbalance)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=color_best, edgecolor='white',
                       label='Best Imbalance technique'),
    ]
    fig.legend(handles=handles, loc='lower right',
               bbox_to_anchor=(0.97, 0.06), fontsize=11, frameon=True,
               fancybox=True, shadow=True)

    fig.suptitle(
        'Comparison of Best Performances — Baseline vs Best Imbalance Scenario',
        fontsize=16, fontweight='bold', y=1.01
    )
    fig.text(0.5, -0.01,
             'Green deltas = improvement with imbalance technique  |  '
             'Red deltas = trade-off decrease',
             ha='center', fontsize=10, style='italic', color='#555555')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  \u2713 {os.path.basename(out_path)}")
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    start_time = time.time()

    print("=" * 70)
    print("STEP 7b — PAPER-STYLE SHAP FIGURES")
    print("=" * 70)

    # Load data
    data = np.load(os.path.join(PROCESSED_DIR, 'step4_arrays.npz'),
                   allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = list(data['feature_names'])

    with open(os.path.join(STEP5_DIR, 'step5_best_configs.json')) as f:
        best_configs = json.load(f)

    # ── Paper Figure 3: Metrics comparison bar plot ──
    print("\n  Generating Paper Figure 3 (metrics comparison)...")
    cv_results_path = os.path.join(STEP5_DIR, 'step5_cv_results.csv')
    plot_paper_fig3_metrics(
        cv_results_path,
        os.path.join(OUT_DIR, 'step7_paper_fig3_metrics_comparison.png')
    )

    # Build models and compute SHAP
    print("\n  Training models and computing SHAP values...")
    models = build_models_and_shap(X_train, y_train, X_test, y_test,
                                   feature_names, best_configs)

    for name, m in models.items():
        label = m['label']
        sv = m['shap_values']
        bv = m['base_value']
        model = m['model']

        print(f"\n{'─' * 70}")
        print(f"  Generating figures for {label}")
        print(f"{'─' * 70}")

        # Figure 3: Combined bar + beeswarm
        plot_fig3_combined(
            sv, X_test, feature_names, bv, label,
            os.path.join(OUT_DIR, f'step7_{name.lower()}_fig3_combined.png')
        )

        # Predictions for case studies
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Figure 4a: Decision plot (all patients, Short highlighted)
        plot_fig4_decision(
            sv, X_test, y_test, feature_names, bv, label,
            os.path.join(OUT_DIR, f'step7_{name.lower()}_fig4_decision.png')
        )

        # Figure 4b: Decision plot case studies (TP/FP/FN/TN)
        plot_fig4_decision_cases(
            sv, X_test, y_test, y_pred, feature_names, bv, label,
            os.path.join(OUT_DIR, f'step7_{name.lower()}_fig4_cases.png')
        )

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Step 7b completed in {total_time:.1f}s")
    print(f"{'=' * 70}")
