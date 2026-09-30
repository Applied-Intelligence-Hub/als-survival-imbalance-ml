"""
Step 7 — SHAP Interpretability Analysis
=========================================
Compute SHAP values for the two focus models:
  1. LightGBM + ROS  (TreeExplainer — exact, fast)
  2. MLP + None       (KernelExplainer — approximate, slower)

Produces per model:
  - Beeswarm plot   (global feature importance + direction)
  - Bar plot        (mean |SHAP|)
  - Waterfall plots (individual patient examples)

Combined:
  - Side-by-side bar comparison (LightGBM vs MLP feature ranking)
  - SHAP values CSV for further analysis
"""

import pandas as pd
import numpy as np
import os
import json
import time
import warnings

from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(BASE_DIR, '..', '01_data', 'processed')
STEP5_DIR = os.path.join(BASE_DIR, '..', '03_outputs', 'step5')
OUT_DIR = os.path.join(BASE_DIR, '..', '03_outputs', 'step7')
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
# Number of background samples for KernelExplainer (MLP)
KERNEL_BG_SAMPLES = 100
# Number of test samples to explain with KernelExplainer
KERNEL_EXPLAIN_SAMPLES = None  # None = all test samples


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    start_time = time.time()

    print("=" * 70)
    print("STEP 7 — SHAP INTERPRETABILITY ANALYSIS")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────
    data = np.load(os.path.join(PROCESSED_DIR, 'step4_arrays.npz'),
                   allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = list(data['feature_names'])

    print(f"\nTrain: {X_train.shape[0]:,} × {X_train.shape[1]} features")
    print(f"Test:  {X_test.shape[0]:,} × {X_test.shape[1]} features")

    # ── Load best configs ──────────────────────────────────────────────
    with open(os.path.join(STEP5_DIR, 'step5_best_configs.json')) as f:
        best_configs = json.load(f)

    # ══════════════════════════════════════════════════════════════════════
    # MODEL 1: LightGBM + ROS  (TreeExplainer)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  MODEL 1: LightGBM + ROS")
    print(f"{'=' * 70}")

    cfg_lgbm = best_configs['LightGBM']
    params_lgbm = cfg_lgbm['best_params']

    # Resample training data
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_tr_lgbm, y_tr_lgbm = ros.fit_resample(X_train, y_train)
    print(f"  Resampled train: {X_tr_lgbm.shape[0]:,} samples")

    # Train
    t0 = time.time()
    lgbm_model = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1,
                                n_jobs=-1, **params_lgbm)
    lgbm_model.fit(X_tr_lgbm, y_tr_lgbm)
    print(f"  Trained in {time.time() - t0:.1f}s")

    # SHAP — TreeExplainer (exact)
    t0 = time.time()
    explainer_lgbm = shap.TreeExplainer(lgbm_model)
    shap_values_lgbm = explainer_lgbm.shap_values(X_test)

    # For binary classification, shap_values may be a list [class0, class1]
    if isinstance(shap_values_lgbm, list):
        shap_values_lgbm = shap_values_lgbm[1]  # class 1 = Short

    print(f"  SHAP computed in {time.time() - t0:.1f}s")
    print(f"  SHAP values shape: {shap_values_lgbm.shape}")

    # Create Explanation object for newer shap API
    base_val_lgbm = explainer_lgbm.expected_value
    if isinstance(base_val_lgbm, (list, np.ndarray)):
        base_val_lgbm = base_val_lgbm[1]  # class 1 = Short
    base_val_lgbm = np.full(X_test.shape[0], base_val_lgbm)
    shap_explanation_lgbm = shap.Explanation(
        values=shap_values_lgbm,
        base_values=base_val_lgbm,
        data=X_test,
        feature_names=feature_names
    )

    # ── LightGBM: Beeswarm plot ──
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_explanation_lgbm, max_display=23, show=False)
    plt.title('SHAP Beeswarm — LightGBM + ROS (Test Set)', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'step7_lgbm_beeswarm.png'), dpi=200,
                bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ step7_lgbm_beeswarm.png")

    # ── LightGBM: Bar plot (mean |SHAP|) ──
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_explanation_lgbm, max_display=23, show=False)
    plt.title('Mean |SHAP| — LightGBM + ROS', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'step7_lgbm_bar.png'), dpi=200,
                bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ step7_lgbm_bar.png")

    # ── LightGBM: Waterfall for most extreme Short patient ──
    short_mask = y_test == 1
    short_indices = np.where(short_mask)[0]
    if len(short_indices) > 0:
        # Patient with highest SHAP sum (most "Short-like")
        shap_sums = shap_values_lgbm[short_indices].sum(axis=1)
        most_short_idx = short_indices[np.argmax(shap_sums)]

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(shap_explanation_lgbm[most_short_idx],
                             max_display=23, show=False)
        plt.title(f'Waterfall — LightGBM (Short patient #{most_short_idx})',
                  fontsize=13, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'step7_lgbm_waterfall_short.png'),
                    dpi=200, bbox_inches='tight')
        plt.close('all')
        print(f"  ✓ step7_lgbm_waterfall_short.png (patient #{most_short_idx})")

    # ── LightGBM: Waterfall for a Non-Short patient ──
    nonshort_indices = np.where(~short_mask)[0]
    if len(nonshort_indices) > 0:
        shap_sums_ns = shap_values_lgbm[nonshort_indices].sum(axis=1)
        most_nonshort_idx = nonshort_indices[np.argmin(shap_sums_ns)]

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(shap_explanation_lgbm[most_nonshort_idx],
                             max_display=23, show=False)
        plt.title(f'Waterfall — LightGBM (Non-Short patient #{most_nonshort_idx})',
                  fontsize=13, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'step7_lgbm_waterfall_nonshort.png'),
                    dpi=200, bbox_inches='tight')
        plt.close('all')
        print(f"  ✓ step7_lgbm_waterfall_nonshort.png (patient #{most_nonshort_idx})")

    # Save SHAP values to CSV
    shap_df_lgbm = pd.DataFrame(shap_values_lgbm, columns=feature_names)
    shap_df_lgbm.insert(0, 'y_true', y_test)
    shap_df_lgbm.to_csv(os.path.join(OUT_DIR, 'step7_lgbm_shap_values.csv'),
                        index=False)
    print(f"  ✓ step7_lgbm_shap_values.csv")

    # Mean |SHAP| ranking
    mean_abs_shap_lgbm = np.abs(shap_values_lgbm).mean(axis=0)
    rank_lgbm = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap_lgbm
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    rank_lgbm.index = rank_lgbm.index + 1
    rank_lgbm.index.name = 'rank'
    rank_lgbm.to_csv(os.path.join(OUT_DIR, 'step7_lgbm_feature_ranking.csv'))
    print(f"  ✓ step7_lgbm_feature_ranking.csv")
    print(f"\n  Top-5 features (LightGBM):")
    for i, row in rank_lgbm.head(5).iterrows():
        print(f"    {i}. {row['feature']:<30s} mean|SHAP|={row['mean_abs_shap']:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # MODEL 2: MLP + None  (KernelExplainer)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  MODEL 2: MLP + None")
    print(f"{'=' * 70}")

    cfg_mlp = best_configs['MLP']
    params_mlp = dict(cfg_mlp['best_params'])
    if isinstance(params_mlp.get('hidden_layer_sizes'), list):
        params_mlp['hidden_layer_sizes'] = tuple(params_mlp['hidden_layer_sizes'])

    # Train (no resampling)
    t0 = time.time()
    mlp_model = MLPClassifier(random_state=RANDOM_STATE, **params_mlp)
    mlp_model.fit(X_train, y_train)
    print(f"  Trained in {time.time() - t0:.1f}s")

    # SHAP — KernelExplainer (approximate)
    # Use k-means summary of training data as background
    t0 = time.time()
    print(f"  Computing KernelExplainer background (k-means, "
          f"k={KERNEL_BG_SAMPLES})...")
    background = shap.kmeans(X_train, KERNEL_BG_SAMPLES)

    def mlp_predict_proba(X):
        return mlp_model.predict_proba(X)[:, 1]

    explainer_mlp = shap.KernelExplainer(mlp_predict_proba, background)

    X_explain = X_test
    if KERNEL_EXPLAIN_SAMPLES is not None:
        X_explain = X_test[:KERNEL_EXPLAIN_SAMPLES]

    print(f"  Computing SHAP values for {X_explain.shape[0]} samples "
          f"(this may take a few minutes)...")
    shap_values_mlp = explainer_mlp.shap_values(X_explain, nsamples=200)
    elapsed_mlp = time.time() - t0
    print(f"  SHAP computed in {elapsed_mlp:.1f}s")
    print(f"  SHAP values shape: {shap_values_mlp.shape}")

    # Create Explanation object
    # base_values must be an array (one per sample) for shap.plots.bar
    base_val_mlp = explainer_mlp.expected_value
    if isinstance(base_val_mlp, (float, np.floating)):
        base_val_mlp = np.full(X_explain.shape[0], base_val_mlp)
    shap_explanation_mlp = shap.Explanation(
        values=shap_values_mlp,
        base_values=base_val_mlp,
        data=X_explain,
        feature_names=feature_names
    )

    # ── MLP: Beeswarm plot ──
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_explanation_mlp, max_display=23, show=False)
    plt.title('SHAP Beeswarm — MLP (Test Set)', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'step7_mlp_beeswarm.png'), dpi=200,
                bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ step7_mlp_beeswarm.png")

    # ── MLP: Bar plot ──
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_explanation_mlp, max_display=23, show=False)
    plt.title('Mean |SHAP| — MLP', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'step7_mlp_bar.png'), dpi=200,
                bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ step7_mlp_bar.png")

    # ── MLP: Waterfall for most extreme Short patient ──
    y_explain = y_test[:len(X_explain)]
    short_mask_mlp = y_explain == 1
    short_indices_mlp = np.where(short_mask_mlp)[0]
    if len(short_indices_mlp) > 0:
        shap_sums_mlp = shap_values_mlp[short_indices_mlp].sum(axis=1)
        most_short_idx_mlp = short_indices_mlp[np.argmax(shap_sums_mlp)]

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(shap_explanation_mlp[most_short_idx_mlp],
                             max_display=23, show=False)
        plt.title(f'Waterfall — MLP (Short patient #{most_short_idx_mlp})',
                  fontsize=13, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'step7_mlp_waterfall_short.png'),
                    dpi=200, bbox_inches='tight')
        plt.close('all')
        print(f"  ✓ step7_mlp_waterfall_short.png")

    # Save SHAP values
    shap_df_mlp = pd.DataFrame(shap_values_mlp, columns=feature_names)
    shap_df_mlp.insert(0, 'y_true', y_explain)
    shap_df_mlp.to_csv(os.path.join(OUT_DIR, 'step7_mlp_shap_values.csv'),
                       index=False)
    print(f"  ✓ step7_mlp_shap_values.csv")

    # Mean |SHAP| ranking
    mean_abs_shap_mlp = np.abs(shap_values_mlp).mean(axis=0)
    rank_mlp = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap_mlp
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    rank_mlp.index = rank_mlp.index + 1
    rank_mlp.index.name = 'rank'
    rank_mlp.to_csv(os.path.join(OUT_DIR, 'step7_mlp_feature_ranking.csv'))
    print(f"  ✓ step7_mlp_feature_ranking.csv")
    print(f"\n  Top-5 features (MLP):")
    for i, row in rank_mlp.head(5).iterrows():
        print(f"    {i}. {row['feature']:<30s} mean|SHAP|={row['mean_abs_shap']:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # COMBINED: Side-by-side feature ranking comparison
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  COMBINED COMPARISON")
    print(f"{'=' * 70}")

    # Merge rankings
    rank_combined = rank_lgbm[['feature', 'mean_abs_shap']].rename(
        columns={'mean_abs_shap': 'LightGBM'}
    ).merge(
        rank_mlp[['feature', 'mean_abs_shap']].rename(
            columns={'mean_abs_shap': 'MLP'}
        ), on='feature'
    )
    # Sort by LightGBM importance
    rank_combined = rank_combined.sort_values('LightGBM', ascending=False
                                             ).reset_index(drop=True)
    rank_combined.index = rank_combined.index + 1
    rank_combined.index.name = 'rank'
    rank_combined.to_csv(os.path.join(OUT_DIR, 'step7_feature_ranking_comparison.csv'))
    print(f"  ✓ step7_feature_ranking_comparison.csv")

    # ── Figure: Side-by-side horizontal bar chart ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)

    # Sort features by LightGBM for consistent ordering
    order = rank_combined['feature'].tolist()[::-1]  # bottom to top

    # LightGBM
    vals_lgbm = [rank_combined.loc[rank_combined['feature'] == f, 'LightGBM'].values[0]
                 for f in order]
    axes[0].barh(range(len(order)), vals_lgbm, color='#17becf', edgecolor='white')
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=9)
    axes[0].set_xlabel('Mean |SHAP value|', fontsize=11)
    axes[0].set_title('LightGBM + ROS', fontsize=13, fontweight='bold')
    for i, v in enumerate(vals_lgbm):
        axes[0].text(v + max(vals_lgbm) * 0.01, i, f'{v:.4f}', va='center',
                     fontsize=8)

    # MLP
    vals_mlp = [rank_combined.loc[rank_combined['feature'] == f, 'MLP'].values[0]
                for f in order]
    axes[1].barh(range(len(order)), vals_mlp, color='#e377c2', edgecolor='white')
    axes[1].set_xlabel('Mean |SHAP value|', fontsize=11)
    axes[1].set_title('MLP', fontsize=13, fontweight='bold')
    for i, v in enumerate(vals_mlp):
        axes[1].text(v + max(vals_mlp) * 0.01, i, f'{v:.4f}', va='center',
                     fontsize=8)

    fig.suptitle('Feature Importance Comparison — Mean |SHAP|', fontsize=15,
                 fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, 'step7_comparison_bar.png'), dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ step7_comparison_bar.png")

    # ── Figure: Rank correlation scatter ──
    fig, ax = plt.subplots(figsize=(8, 8))
    # Assign ranks
    rc = rank_combined.copy()
    rc['rank_lgbm'] = rc['LightGBM'].rank(ascending=False).astype(int)
    rc['rank_mlp'] = rc['MLP'].rank(ascending=False).astype(int)

    ax.scatter(rc['rank_lgbm'], rc['rank_mlp'], s=80, c='#2ca02c', zorder=3)
    for _, row in rc.iterrows():
        ax.annotate(row['feature'], (row['rank_lgbm'], row['rank_mlp']),
                    fontsize=7.5, ha='left', va='bottom',
                    xytext=(4, 4), textcoords='offset points')

    ax.plot([0.5, 23.5], [0.5, 23.5], 'k--', alpha=0.3, lw=1)
    ax.set_xlabel('LightGBM rank', fontsize=12)
    ax.set_ylabel('MLP rank', fontsize=12)
    ax.set_title('Feature Importance Rank Correlation\n(LightGBM vs MLP)',
                 fontsize=13)
    ax.set_xlim([0, 24])
    ax.set_ylim([0, 24])
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Spearman correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(rc['rank_lgbm'], rc['rank_mlp'])
    ax.text(0.05, 0.95, f'Spearman ρ = {rho:.3f}\np = {pval:.4f}',
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'step7_rank_correlation.png'), dpi=200)
    plt.close(fig)
    print(f"  ✓ step7_rank_correlation.png")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("STEP 7 — SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  {'Rank':<6s} {'Feature':<30s} {'LightGBM |SHAP|':>16s} "
          f"{'MLP |SHAP|':>12s}")
    print("  " + "─" * 68)
    for i, row in rank_combined.iterrows():
        print(f"  {i:<6d} {row['feature']:<30s} {row['LightGBM']:>16.4f} "
              f"{row['MLP']:>12.4f}")

    print(f"\n  Spearman rank correlation: ρ = {rho:.3f} (p = {pval:.4f})")

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Step 7 completed in {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Output directory: {OUT_DIR}")
    print(f"{'=' * 70}")
