"""
step8_figures_paper.py
Generates publication-ready figures for the EPIA paper:
  1. Confusion matrix panel (LightGBM+ROS, SVM+SMOTEENN, MLP+None)
  2. AUC-Sensitivity decoupling bar chart
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "03_outputs" / "step8"
FIGURES_DIR = BASE_DIR / "overleaf" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette (colourblind-safe) ────────────────────────────────────────
C_AUC  = "#2196F3"   # blue  – cross-validation AUC
C_SENS = "#FF9800"   # orange – held-out sensitivity
C_TP   = "#4CAF50"   # green  – correct minority
C_FN   = "#F44336"   # red    – missed minority
C_FP   = "#FFB300"   # amber  – false alarm
C_TN   = "#90A4AE"   # grey   – correct majority


# =============================================================================
# Figure 1 — Confusion Matrix Panel
# =============================================================================

def plot_confusion_matrices():
    cm_path = BASE_DIR / "03_outputs" / "step6" / "step6_confusion_matrices.json"
    with open(cm_path) as f:
        raw = json.load(f)

    models = {
        "LightGBM + ROS":    raw["LightGBM"],
        "SVM + SMOTEENN":    raw["SVM"],
        "MLP + None":        raw["MLP"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Confusion Matrices — Held-Out Test Set (n = 301)",
                 fontsize=13, fontweight="bold", y=1.02)

    labels = ["Standard Survival\n(Majority)", "Short Survivor\n(Minority)"]

    for ax, (title, d) in zip(axes, models.items()):
        # Build 2x2 matrix: rows = actual, cols = predicted
        # Actual Standard Survival: TN (pred Standard), FP (pred Short)
        # Actual Short Survivor:    FN (pred Standard), TP (pred Short)
        matrix = np.array([[d["TN"], d["FP"]],
                            [d["FN"], d["TP"]]])
        total  = matrix.sum()

        colors = [[C_TN, C_FP],
                  [C_FN, C_TP]]

        for r in range(2):
            for c in range(2):
                val = matrix[r, c]
                pct = 100 * val / total
                ax.add_patch(plt.Rectangle(
                    (c, 1 - r), 1, 1,
                    color=colors[r][c], alpha=0.80, transform=ax.transData
                ))
                ax.text(c + 0.5, 1.5 - r, f"{val}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=11, fontweight="bold")

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["Pred: Standard\nSurvival", "Pred: Short\nSurvivor"],
                           fontsize=8.5)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["Actual: Short\nSurvivor", "Actual: Standard\nSurvival"],
                           fontsize=8.5)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    legend_patches = [
        mpatches.Patch(color=C_TP, alpha=0.80, label="True Positive (TP)"),
        mpatches.Patch(color=C_FN, alpha=0.80, label="False Negative (FN) — missed"),
        mpatches.Patch(color=C_FP, alpha=0.80, label="False Positive (FP)"),
        mpatches.Patch(color=C_TN, alpha=0.80, label="True Negative (TN)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.06), frameon=False)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_confusion_matrices.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# =============================================================================
# Figure 2 — AUC-Sensitivity Decoupling Bar Chart
# =============================================================================

def plot_auc_sensitivity():
    # Cross-validation AUC and held-out test sensitivity from Tables 2 & 3
    classifiers = ["KNN", "Naïve Bayes", "Decision Tree",
                   "Random Forest", "SVM\n+SMOTEENN", "MLP\n+None", "LightGBM\n+ROS"]

    cv_auc  = [0.880, 0.891, 0.861, 0.906, 0.911, 0.915, 0.914]
    test_sens = [0.800, 0.714, 0.286, 0.257, 0.886, 0.029, 0.857]

    x = np.arange(len(classifiers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))

    bars_auc  = ax.bar(x - width / 2, cv_auc,  width, label="CV AUC (ranking)",
                       color=C_AUC,  alpha=0.88, zorder=3)
    bars_sens = ax.bar(x + width / 2, test_sens, width, label="Test Sensitivity (detection)",
                       color=C_SENS, alpha=0.88, zorder=3)

    # Annotate each bar
    for bar in bars_auc:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#1565C0")

    for bar in bars_sens:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#E65100")

    # Highlight the MLP contrast
    mlp_idx = classifiers.index("MLP\n+None")
    ax.annotate("AUC–Sensitivity\ndecoupling",
                xy=(mlp_idx + width / 2, test_sens[mlp_idx] + 0.04),
                xytext=(mlp_idx + 1.1, 0.25),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                fontsize=8.5, ha="center")

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Cross-Validation AUC vs Held-Out Test Sensitivity per Classifier",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, zorder=2)
    ax.legend(fontsize=8, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_auc_sensitivity_decoupling.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_auc_sensitivity_decoupling.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    print("Generating paper figures...")
    plot_confusion_matrices()
    plot_auc_sensitivity()
    print("Done.")
