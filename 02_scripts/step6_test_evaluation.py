"""
Step 6 — Test-Set Evaluation
==============================
For each classifier, use the best (classifier × imbalance) config from Step 5:
  1. Train on the full training set (with resampling if applicable)
  2. Predict on the held-out test set
  3. Compute all metrics + confusion matrix + ROC curve

Produces:
  - step6_test_results.csv          (one row per classifier)
  - step6_confusion_matrices.json   (per-classifier TP/FP/FN/TN)
  - step6_roc_curves.png            (overlay ROC for all classifiers)
  - step6_roc_curves_focus.png      (MLP vs LightGBM only)
  - step6_metric_comparison.png     (grouped bar chart: all metrics)
"""

import pandas as pd
import numpy as np
import os
import json
import time
import warnings

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, balanced_accuracy_score,
    accuracy_score
)

from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
)
from imblearn.under_sampling import (
    RandomUnderSampler, NearMiss, TomekLinks
)
from imblearn.combine import SMOTETomek, SMOTEENN

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(BASE_DIR, '..', '01_data', 'processed')
STEP5_DIR = os.path.join(BASE_DIR, '..', '03_outputs', 'step5')
OUT_DIR = os.path.join(BASE_DIR, '..', '03_outputs', 'step6')
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42

CLASSIFIERS_ORDER = ['KNN', 'NaiveBayes', 'DecisionTree', 'RandomForest',
                     'SVM', 'MLP', 'LightGBM']


# ══════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS  (same as step5)
# ══════════════════════════════════════════════════════════════════════════

def make_classifier(name, params):
    if name == 'KNN':
        return KNeighborsClassifier(**params)
    if name == 'NaiveBayes':
        return GaussianNB(**params)
    if name == 'DecisionTree':
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    if name == 'RandomForest':
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if name == 'SVM':
        return SVC(probability=True, random_state=RANDOM_STATE, **params)
    if name == 'MLP':
        # hidden_layer_sizes may come as list — convert to tuple
        p = dict(params)
        if isinstance(p.get('hidden_layer_sizes'), list):
            p['hidden_layer_sizes'] = tuple(p['hidden_layer_sizes'])
        return MLPClassifier(random_state=RANDOM_STATE, **p)
    if name == 'LightGBM':
        return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
                              **params)
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
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    start_time = time.time()

    print("=" * 70)
    print("STEP 6 — TEST-SET EVALUATION")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────
    data = np.load(os.path.join(PROCESSED_DIR, 'step4_arrays.npz'),
                   allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = data['feature_names']

    print(f"\nTrain: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    print(f"Test:  {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
    print(f"Test target: Short={y_test.sum():.0f}, "
          f"Non-Short={(y_test == 0).sum():.0f}")

    # ── Load best configs from step5 ───────────────────────────────────
    with open(os.path.join(STEP5_DIR, 'step5_best_configs.json')) as f:
        best_configs = json.load(f)

    print(f"\nLoaded best configs for {len(best_configs)} classifiers")

    # ── Evaluate each classifier ───────────────────────────────────────
    results = []
    conf_matrices = {}
    test_probabilities = {}
    roc_data = {}   # {clf_name: (fpr, tpr, auc)}

    for clf_name in CLASSIFIERS_ORDER:
        if clf_name not in best_configs:
            print(f"\n  ⚠ {clf_name}: no config found, skipping")
            continue

        cfg = best_configs[clf_name]
        imb_name = cfg['imbalance_technique']
        params = cfg['best_params']
        cv_auc = cfg['mean_auc']

        print(f"\n{'─' * 70}")
        print(f"  {clf_name} + {imb_name}  (CV AUC={cv_auc:.4f})")
        t0 = time.time()

        # Resample training data
        sampler = make_sampler(imb_name)
        X_tr, y_tr = X_train.copy(), y_train.copy()
        if sampler is not None:
            X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)
            print(f"  Resampled: {X_tr.shape[0]:,} samples "
                  f"(Short={y_tr.sum():.0f}, Non-Short={(y_tr == 0).sum():.0f})")

        # Train
        clf = make_classifier(clf_name, params)
        clf.fit(X_tr, y_tr)

        # Predict
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        elapsed = time.time() - t0

        # Metrics
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        spec = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        gmean = np.sqrt(rec * spec)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        conf_matrices[clf_name] = {'TP': int(tp), 'FP': int(fp),
                                   'FN': int(fn), 'TN': int(tn)}

        test_probabilities[clf_name] = {
            'y_true': y_test.tolist(),
            'y_proba': y_prob.tolist(),
            'y_pred': y_pred.tolist(),
            'imbalance_technique': imb_name,
        }

        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[clf_name] = (fpr, tpr, auc)

        row = {
            'classifier': clf_name,
            'imbalance_technique': imb_name,
            'cv_auc': cv_auc,
            'test_auc': round(auc, 4),
            'test_f1': round(f1, 4),
            'test_gmean': round(gmean, 4),
            'test_balanced_acc': round(bal_acc, 4),
            'test_precision': round(prec, 4),
            'test_recall': round(rec, 4),
            'test_specificity': round(spec, 4),
            'test_accuracy': round(acc, 4),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'time_seconds': round(elapsed, 1),
        }
        results.append(row)

        print(f"  AUC={auc:.4f}  F1={f1:.4f}  G-Mean={gmean:.4f}  "
              f"BalAcc={bal_acc:.4f}")
        print(f"  Recall={rec:.4f}  Specificity={spec:.4f}  "
              f"Precision={prec:.4f}")
        print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"  ({elapsed:.1f}s)")

    # ══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════════════
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, 'step6_test_results.csv'),
                      index=False)

    with open(os.path.join(OUT_DIR, 'step6_confusion_matrices.json'), 'w') as f:
        json.dump(conf_matrices, f, indent=2)

    with open(os.path.join(OUT_DIR, 'step6_test_probabilities.json'), 'w') as f:
        json.dump(test_probabilities, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 1 — ROC CURVES (ALL CLASSIFIERS)
    # ══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = {
        'KNN': '#1f77b4', 'NaiveBayes': '#ff7f0e', 'DecisionTree': '#2ca02c',
        'RandomForest': '#d62728', 'SVM': '#9467bd',
        'MLP': '#e377c2', 'LightGBM': '#17becf',
    }

    for clf_name in CLASSIFIERS_ORDER:
        if clf_name not in roc_data:
            continue
        fpr, tpr, auc_val = roc_data[clf_name]
        lw = 2.5 if clf_name in ('MLP', 'LightGBM') else 1.5
        ls = '-' if clf_name in ('MLP', 'LightGBM') else '--'
        ax.plot(fpr, tpr, color=colors[clf_name], lw=lw, linestyle=ls,
                label=f'{clf_name} (AUC={auc_val:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('ROC Curves — Test Set (All Classifiers)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'step6_roc_curves.png'), dpi=200)
    plt.close(fig)
    print(f"\n  ✓ Saved: step6_roc_curves.png")

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 2 — ROC CURVES FOCUS (MLP vs LightGBM)
    # ══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(7, 6))
    for clf_name in ['MLP', 'LightGBM']:
        if clf_name not in roc_data:
            continue
        fpr, tpr, auc_val = roc_data[clf_name]
        ax.plot(fpr, tpr, color=colors[clf_name], lw=2.5,
                label=f'{clf_name} (AUC={auc_val:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('ROC Curves — MLP vs LightGBM (Test Set)', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'step6_roc_focus.png'), dpi=200)
    plt.close(fig)
    print(f"  ✓ Saved: step6_roc_focus.png")

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 3 — METRIC COMPARISON BAR CHART
    # ══════════════════════════════════════════════════════════════════════
    metrics_to_plot = ['test_auc', 'test_f1', 'test_gmean',
                       'test_recall', 'test_specificity', 'test_precision']
    metric_labels = ['AUC', 'F1', 'G-Mean', 'Recall', 'Specificity', 'Precision']

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metric_labels))
    n_clf = len(results_df)
    width = 0.8 / n_clf

    for i, (_, row) in enumerate(results_df.iterrows()):
        vals = [row[m] for m in metrics_to_plot]
        clf_name = row['classifier']
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                      label=clf_name, color=colors.get(clf_name, '#888'),
                      edgecolor='white', linewidth=0.5)
        # Annotate values
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=6.5,
                    fontweight='bold', rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Test-Set Metrics — All Classifiers', fontsize=14)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.set_ylim([0, 1.18])
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'step6_metric_comparison.png'), dpi=200)
    plt.close(fig)
    print(f"  ✓ Saved: step6_metric_comparison.png")

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 4 — CV vs TEST AUC (overfitting check)
    # ══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(9, 5))
    clfs = results_df['classifier'].tolist()
    cv_aucs = results_df['cv_auc'].tolist()
    test_aucs = results_df['test_auc'].tolist()

    x = np.arange(len(clfs))
    w = 0.35
    bars_cv = ax.bar(x - w / 2, cv_aucs, w, label='CV AUC', color='#5b9bd5',
                     edgecolor='white')
    bars_test = ax.bar(x + w / 2, test_aucs, w, label='Test AUC', color='#ed7d31',
                       edgecolor='white')

    for bar, v in zip(bars_cv, cv_aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, v in zip(bars_test, test_aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Delta annotation
    for i, (cv, te) in enumerate(zip(cv_aucs, test_aucs)):
        delta = te - cv
        sign = '+' if delta >= 0 else ''
        ax.text(x[i], max(cv, te) + 0.025, f'Δ={sign}{delta:.4f}',
                ha='center', va='bottom', fontsize=7, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(clfs, fontsize=10)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Cross-Validation AUC vs Test-Set AUC', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim([0.80, 1.00])
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'step6_cv_vs_test.png'), dpi=200)
    plt.close(fig)
    print(f"  ✓ Saved: step6_cv_vs_test.png")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 6 — TEST-SET RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  {'Classifier':<15s} {'Imbalance':<15s} {'CV AUC':>8s} "
          f"{'Test AUC':>9s} {'F1':>7s} {'G-Mean':>8s} "
          f"{'Recall':>8s} {'Spec':>7s}")
    print("  " + "─" * 85)

    for _, r in results_df.iterrows():
        delta = r['test_auc'] - r['cv_auc']
        print(f"  {r['classifier']:<15s} {r['imbalance_technique']:<15s} "
              f"{r['cv_auc']:>8.4f} {r['test_auc']:>9.4f} "
              f"{r['test_f1']:>7.4f} {r['test_gmean']:>8.4f} "
              f"{r['test_recall']:>8.4f} {r['test_specificity']:>7.4f}  "
              f"(Δ={delta:+.4f})")

    # Highlight MLP vs LightGBM
    print(f"\n{'─' * 70}")
    print("  ★ FOCUS: MLP vs LightGBM")
    print(f"{'─' * 70}")
    for _, r in results_df[results_df['classifier'].isin(['MLP', 'LightGBM'])].iterrows():
        print(f"  {r['classifier']:<12s}: AUC={r['test_auc']:.4f}  "
              f"F1={r['test_f1']:.4f}  G-Mean={r['test_gmean']:.4f}  "
              f"Recall={r['test_recall']:.4f}  Spec={r['test_specificity']:.4f}")

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Step 6 completed in {total_time:.1f}s")
    print(f"Results: {os.path.join(OUT_DIR, 'step6_test_results.csv')}")
    print(f"{'=' * 70}")
