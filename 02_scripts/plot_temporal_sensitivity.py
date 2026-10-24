"""Plot completed temporal sensitivity scores, without data access or model fits."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=ROOT / "03_outputs/reviewer_followup/revised_analyses")
    parser.add_argument("--published-results", type=Path,
                        default=ROOT / "assets/reviewer_followup/results.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "03_outputs/reviewer_followup/revised_analyses")
    args = parser.parse_args()
    published = json.loads(args.published_results.read_text(encoding="utf-8"))
    fresh = args.results / "s2_revised_metrics.csv"
    revised = pd.read_csv(fresh) if fresh.exists() else pd.DataFrame(published["metrics"])
    data = pd.concat([pd.DataFrame(published["reference"]), revised], ignore_index=True)
    if data.duplicated(["cohort", "classifier"]).any():
        raise ValueError("Duplicate classifier/cohort results")
    variants = ["reference", "entry_18m", "landmark_18m", "entry_12m"]
    selected = data.loc[data.classifier.isin(["LightGBM", "MLP"]) & data.cohort.isin(variants)]
    if len(selected) != 8:
        raise ValueError("Expected all four comparisons for LightGBM and MLP")
    labels = []
    for key, title in zip(variants, ["Reference", "Entry $\\leq18$m", "18m landmark", "Entry $\\leq12$m"]):
        row = selected.loc[selected.cohort.eq(key) & selected.classifier.eq("MLP")].iloc[0]
        labels.append(f"{title}\n{int(row.n_test)} / {int(row.test_short)}")
    navy, cyan, magenta, gold = "#16324f", "#00b0be", "#f54f74", "#ffb255"
    plt.rcdefaults()
    plt.rcParams.update({"text.color": navy, "axes.labelcolor": navy, "axes.edgecolor": navy,
                         "xtick.color": navy, "ytick.color": navy, "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7), sharey=True)
    for axis, metric, title in zip(axes, ["roc_auc", "average_precision"], ["ROC-AUC", "Average precision"]):
        for offset, name, colour in [(-.10, "LightGBM", cyan), (.10, "MLP", magenta)]:
            group = selected.loc[selected.classifier.eq(name)].set_index("cohort").loc[variants]
            centre = group[metric].to_numpy()
            errors = np.vstack([centre - group[metric + "_lower"], group[metric + "_upper"] - centre])
            axis.errorbar(np.arange(4) + offset, centre, yerr=errors, fmt="o", color=colour,
                          capsize=3, markersize=5, label="LightGBM+ROS" if name == "LightGBM" else name)
        axis.set(xticks=np.arange(4), xticklabels=labels, ylim=(0, 1.04), title=title)
        axis.grid(axis="y", alpha=.15, color=navy)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(.5, color=navy, ls=":", lw=1)
    prevalence = selected.loc[selected.classifier.eq("MLP")].set_index("cohort").loc[variants, "test_prevalence"]
    axes[1].scatter(np.arange(4), prevalence, color=gold, marker="x", s=45,
                    label="AP prevalence reference", zorder=5)
    fig.supxlabel("Analysis: TEST participants / positive outcomes", y=.015)
    fig.suptitle("Study II: temporal sensitivity with fixed configurations", color=navy, fontweight="bold")
    handles, legend = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend, loc="upper center", bbox_to_anchor=(.5, .93), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .08, 1, .84))
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        fig.savefig(args.output / f"review_s2_timing_sensitivity.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Temporal figure exported from aggregate scores; no patient data or fitted models were read.")


if __name__ == "__main__":
    main()
