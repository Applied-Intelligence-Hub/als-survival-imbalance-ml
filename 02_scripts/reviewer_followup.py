"""Study II aggregate reviewer checks; reuse and validate saved test predictions.

Analysis functions are exported from the dissertation reviewer_followup.py.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import binomtest, chi2_contingency, fisher_exact, mannwhitneyu
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
INPUTS: dict[str, str] = {}


def fingerprint(path: Path) -> None:
    INPUTS[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    fingerprint(path)
    return pd.read_csv(path)


def write_table(out: Path, name: str, caption: str, label: str,
                headers: list[str], rows: list[list[str]], note: str = "") -> None:
    body = [r"\begin{table}[ht!]", r"\centering", r"\small",
            rf"\caption[{caption}]{{{caption}}}", rf"\label{{{label}}}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}", r"\toprule",
            " & ".join(headers) + r" \\", r"\midrule"]
    body.extend(" & ".join(row) + r" \\" for row in rows)
    body.extend([r"\bottomrule", r"\end{tabular}"])
    if note:
        body.extend([r"\par\smallskip", r"\begin{minipage}{0.98\linewidth}\footnotesize",
                     note, r"\end{minipage}"])
    body.append(r"\end{table}")
    (out / f"{name}.tex").write_text("\n".join(body) + "\n", encoding="utf-8")


def summary(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "--"
    q = s.quantile([0.25, 0.5, 0.75])
    return (r"\shortstack{" + f"{q.loc[0.5]:.1f} [{q.loc[0.25]:.1f}, {q.loc[0.75]:.1f}]"
            + r" \\ $n=" + str(len(s)) + "$}")


def calibration_diagnostics(y: np.ndarray, p: np.ndarray) -> dict:
    logit = np.log(np.clip(p, 1e-8, 1 - 1e-8) / np.clip(1 - p, 1e-8, 1))
    def loss(beta):
        eta = beta[0] + beta[1] * logit
        return np.sum(np.logaddexp(0, eta) - y * eta)
    fit = minimize(loss, [0.0, 1.0], method="BFGS", tol=1e-7)
    # A small numerical gradient is acceptable when BFGS reports precision loss.
    eta = fit.x[0] + fit.x[1] * logit
    grad = np.array([np.sum(expit(eta) - y), np.sum((expit(eta) - y) * logit)])
    if not np.isfinite(fit.fun) or np.linalg.norm(grad) > 1e-3:
        raise RuntimeError(f"Calibration diagnostic did not converge: {fit.message}")
    return {"intercept": float(fit.x[0]), "slope": float(fit.x[1])}


def study2(out: Path, probabilities: Path, boots: int) -> dict:
    plt.rcdefaults()
    plt.rcParams.update({"axes.labelcolor":"#16324f", "axes.edgecolor":"#16324f",
                         "axes.titlecolor":"#16324f", "axes.titleweight":"bold",
                         "xtick.color":"#16324f", "ytick.color":"#16324f"})
    fingerprint(probabilities)
    data = json.loads(probabilities.read_text(encoding="utf-8"))
    arrays_path = ROOT / "01_data/processed/step4_arrays_unscaled.npz"
    fingerprint(arrays_path)
    arrays = np.load(arrays_path, allow_pickle=True)
    reference = read_csv(ROOT / "03_outputs/step6/step6_test_results.csv").set_index("classifier")
    rows, metrics = [], []
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2), sharex=True, sharey=True)
    for i, (name, d) in enumerate(data.items()):
        y, p, pred = np.array(d["y_true"]), np.array(d["y_proba"]), np.array(d["y_pred"])
        assert np.array_equal(y, arrays["y_test"]), name
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        r = reference.loc[name]
        assert [tn,fp,fn,tp] == [int(r.TN),int(r.FP),int(r.FN),int(r.TP)], name
        assert abs(roc_auc_score(y,p)-r.test_auc) <= 5.01e-5, name
        assert d["imbalance_technique"] == (r.imbalance_technique if pd.notna(r.imbalance_technique) else "None")
        brier, ap = brier_score_loss(y,p), average_precision_score(y,p)
        record = dict(classifier=name, average_precision=ap, brier=brier,
                      mean_probability=p.mean(), prevalence=y.mean(), **calibration_diagnostics(y,p))
        metrics.append(record)
        rows.append([name, f"{ap:.3f}", f"{brier:.3f}", f"{p.mean():.3f}"])
        observed, predicted = calibration_curve(y, p, n_bins=5, strategy="quantile")
        ax = axes.flat[i]
        ax.plot([0,1],[0,1], ls="--", color="0.6", lw=.8)
        ax.plot(predicted, observed, "o-", color="#00b0be", ms=4)
        ax.set(title=name, xlim=(0,1), ylim=(0,1))
        ax.grid(alpha=.15)
    axes.flat[-1].axis("off")
    fig.supxlabel("Mean predicted probability (five quantile bins)")
    fig.supylabel("Observed Short Survivor frequency")
    fig.tight_layout()
    fig.savefig(out / "review_s2_calibration.pdf", bbox_inches="tight")
    fig.savefig(out / "review_s2_calibration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(metrics).to_csv(out / "review_s2_probability_metrics.csv", index=False)
    write_table(out, "review_s2_probability_metrics", "Study II held-out probability assessment",
                "tab:review_s2_probability_metrics", ["Classifier", "Average precision", "Brier score", "Mean probability"], rows,
                "Selected configurations are unchanged from the primary analysis. Test prevalence is 35/301 (0.116); a constant prediction equal to that prevalence has Brier score 0.103. Lower Brier scores are better. No probability recalibration was performed.")

    a = read_csv(ROOT / "01_data/interim/step2_patient_features.csv")
    final = read_csv(ROOT / "01_data/processed/step3_coded_unscaled.csv")
    dead = a.Event_Dead.astype(bool)
    short = dead & a.Event_Dead_Time_from_Onset.notna() & a.Event_Dead_Time_from_Onset.le(24)
    standard = (dead & a.Event_Dead_Time_from_Onset.gt(24)) | (~dead & a.Last_Visit_from_Onset.ge(24))
    censored = ~dead & a.Last_Visit_from_Onset.notna() & a.Last_Visit_from_Onset.lt(24)
    unknown = ~(short|standard|censored)
    assert [int(short.sum()),int(standard.sum()),int(censored.sum()),int(unknown.sum())] == [656,5464,1396,1967]
    eligible = a.loc[(short|standard) & ~a.Site_of_Onset.isin(["Other","Limb_and_Bulbar"])].copy()
    assert len(eligible) == 5802 and len(final) == 1502
    retained, excluded = eligible.loc[eligible.subject_id.isin(final.subject_id)], eligible.loc[~eligible.subject_id.isin(final.subject_id)]
    rows = [["Participants", f"{len(retained):,}", f"{len(excluded):,}"]]
    for col, label in [("Age_at_Onset", "Age at onset (years)"), ("Diagnosis_Delay", "Diagnostic delay (months)"),
                       ("FVC_at_Diagnosis", r"FVC (\% predicted)"), ("BMI_at_Diagnosis", "BMI")]:
        rows.append([label, summary(retained[col]), summary(excluded[col])])
    for col, value, label in [("Sex","Male","Male / recorded sex"), ("Site_of_Onset","Bulbar","Bulbar / recorded site")]:
        rows.append([label,*[f"{c[col].eq(value).sum()} / {c[col].notna().sum()}" for c in [retained,excluded]]])
    rows.append(["Short Survivors",*[str(int(short.loc[c.index].sum())) for c in [retained,excluded]]])
    write_table(out, "review_s2_selection", "Study II complete cases and excluded participants",
                "tab:review_s2_selection", ["Characteristic", "Complete cases", "Incomplete cases"], rows,
                "Continuous values are median [interquartile range]; number observed. Both groups belong to the 5,802 outcome-eligible participants after the two onset-site exclusions. Measurements retain the original closest-to-diagnosis definition.")

    # PRO-ACT defines delta zero as trial onset; it is not a verified admission date.
    timing = retained.merge(final[["subject_id","Survival_Group"]], on="subject_id", validate="one_to_one")
    timing["entry_months"] = -timing.Onset_Delta / 30.44
    timing_rows, timing_records = [], []
    for value, label in [(1,"Short Survivor"),(0,"Standard Survival")]:
        s = timing.loc[timing.Survival_Group.eq(value),"entry_months"].dropna()
        record = dict(group=label, n=len(s), q25=s.quantile(.25), median=s.median(), q75=s.quantile(.75),
                      at_or_after_24=int(s.ge(24).sum()), min=s.min(), max=s.max())
        timing_records.append(record)
        timing_rows.append([label, str(len(s)), f"{s.median():.1f} [{s.quantile(.25):.1f}, {s.quantile(.75):.1f}]",
                            f"{s.ge(24).sum()} ({100*s.ge(24).mean():.1f}\\%)"])
    write_table(out, "review_s2_delayed_entry", "Study II symptom onset to the source-trial reference day",
                "tab:review_s2_delayed_entry", ["Outcome", "$N$", "Months: median [IQR]", "Entry $\\geq24$ months"], timing_rows,
                "Calculated as minus the recorded symptom-onset offset divided by 30.44. PRO-ACT defines day zero as trial onset; it is used here as an entry-time proxy, not a verified individual enrolment date. The audit does not correct left truncation or establish when every predictor became available.")

    positive = arrays["y_test"].astype(bool)
    lgb, svm = np.array(data["LightGBM"]["y_pred"])[positive], np.array(data["SVM"]["y_pred"])[positive]
    lgb_only, svm_only = int(((lgb==1)&(svm==0)).sum()), int(((lgb==0)&(svm==1)).sum())
    paired_p = binomtest(lgb_only, lgb_only+svm_only, .5).pvalue if lgb_only+svm_only else 1.0
    rng = np.random.default_rng(42)
    indices = rng.integers(0,len(lgb),size=(boots,len(lgb)))
    differences = (lgb[indices]-svm[indices]).mean(axis=1)
    lower, upper = np.quantile(differences,[.025,.975])

    history = read_csv(ROOT / "01_data/raw/PROACT_ALSHISTORY.csv")
    text = history.Site_of_Onset.astype("string").str.strip()
    recognised = text.str.contains("Bulbar|Other", regex=True, na=False) | text.isin(["Onset: Limb","Limb","Onset: Spine","Spine","Spinal"])
    fallback = history.loc[text.notna() & ~recognised,"subject_id"]
    return dict(probability_metrics=metrics, timing=timing_records,
                unknown_outcomes=int(unknown.sum()), unknown_missing_onset=int(a.loc[unknown,"Onset_Delta"].isna().sum()),
                unknown_missing_death_time=int((dead[unknown]&a.loc[unknown,"Event_Dead_Time_from_Onset"].isna()).sum()),
                unknown_missing_followup=int((~dead[unknown]&a.loc[unknown,"Last_Visit_from_Onset"].isna()).sum()),
                site_fallback_unique_patients=int(fallback.nunique()),
                site_fallback_in_final=int(final.subject_id.isin(fallback).sum()),
                paired_sensitivity=dict(lgb_only=lgb_only,svm_only=svm_only,p=paired_p,
                                        difference=float((lgb-svm).mean()),lower=lower,upper=upper))



def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--probabilities", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    ROOT = args.project_root.resolve()
    out = args.output or ROOT / "03_outputs/reviewer_followup"
    probabilities = args.probabilities or ROOT / "03_outputs/step6/step6_test_probabilities.json"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": "Study II fixed-cohort, saved-prediction checks; no model refits.",
        "bootstrap_resamples": args.bootstrap,
        "study2": study2(out, probabilities, args.bootstrap),
        "packages": {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]},
    }
    result["input_sha256"] = INPUTS
    (out / "review_followup_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["study2"], indent=2))


if __name__ == "__main__":
    main()
