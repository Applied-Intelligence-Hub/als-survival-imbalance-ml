"""Fixed-parameter Study II temporal sensitivity with explicit failure records."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

from reviewer_temporal_cohorts import build_cohorts


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY
NAMES = ["KNN", "NaiveBayes", "DecisionTree", "RandomForest", "SVM", "MLP", "LightGBM"]


def factories():
    spec = importlib.util.spec_from_file_location("study2_reference_factories", STUDY / "02_scripts/step5_modeling.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_predict(factory, name, parameters, strategy, x, y, target, jobs):
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    target = scaler.transform(target)
    sampler = factory.make_sampler(strategy)
    if sampler is not None:
        x, y = sampler.fit_resample(x, y)
    if len(np.unique(y)) != 2:
        raise ValueError("Resampling removed an outcome class")
    parameters = dict(parameters)
    if name == "MLP":
        parameters["hidden_layer_sizes"] = tuple(parameters["hidden_layer_sizes"])
    model = factory.make_classifier(name, parameters)
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=jobs)
    model.fit(x, y)
    probabilities = model.predict_proba(target)[:, 1]
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities")
    return probabilities, model.predict(target)


def metrics(y, probability, prediction):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"roc_auc": roc_auc_score(y, probability),
            "average_precision": average_precision_score(y, probability),
            "brier": brier_score_loss(y, probability),
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "balanced_accuracy": balanced_accuracy_score(y, prediction),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def thresholds(y, probability):
    fpr, tpr, values = roc_curve(y, probability)
    finite = np.isfinite(values)
    youden = int(np.argmax(np.where(finite, tpr - fpr, -np.inf)))
    sensitivity = np.flatnonzero((tpr >= .90) & finite)[0]
    candidates = np.linspace(.01, .99, 300)
    scores = [f1_score(y, probability >= value, zero_division=0) for value in candidates]
    return {"probability_0_5": .5, "youden": float(values[youden]),
            "sensitivity_0_90": float(values[sensitivity]),
            "f1": float(candidates[int(np.argmax(scores))])}


def intervals(y, probability, prediction, boots):
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(y), size=(boots, len(y)))
    by, bp, bd = y[indices], probability[indices], prediction[indices]
    valid = (by.sum(axis=1) > 0) & (by.sum(axis=1) < len(y))
    by, bp, bd = by[valid], bp[valid], bd[valid]
    values = {"roc_auc": [], "average_precision": []}
    for ys, ps in zip(by, bp):
        values["roc_auc"].append(roc_auc_score(ys, ps))
        values["average_precision"].append(average_precision_score(ys, ps))
    values["brier"] = np.mean((bp - by) ** 2, axis=1)
    values["sensitivity"] = (by * bd).sum(axis=1) / by.sum(axis=1)
    values["specificity"] = ((1 - by) * (1 - bd)).sum(axis=1) / (1 - by).sum(axis=1)
    result = {"valid_bootstraps": int(valid.sum())}
    for name, vals in values.items():
        result[name + "_lower"], result[name + "_upper"] = np.quantile(vals, [.025, .975]).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "03_outputs/reviewer_followup/revised_analyses")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(args.jobs))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.bootstrap < 1 or args.jobs < 1:
        parser.error("--bootstrap and --jobs must be positive")
    cohorts, metadata = build_cohorts(STUDY)
    factory = factories()
    config_path = STUDY / "03_outputs/step5/step5_best_configs.json"
    configurations = json.loads(config_path.read_text())
    original_train = pd.read_csv(STUDY / "01_data/processed/step4_train_unscaled.csv")
    original_test = pd.read_csv(STUDY / "01_data/processed/step4_test_unscaled.csv")
    features = [c for c in original_train if c not in ["subject_id", "Survival_Group"]]
    private = STUDY / "01_data/interim/reviewer_revision"
    private.mkdir(parents=True, exist_ok=True)
    summary, all_thresholds, fold_records, warning_records, cohort_records, status = [], [], [], [], [], []
    fit_count = 0
    for cohort_name, data in cohorts.items():
        lookup = data.set_index("subject_id")
        train_ids = original_train.loc[original_train.subject_id.isin(lookup.index), "subject_id"]
        test_ids = original_test.loc[original_test.subject_id.isin(lookup.index), "subject_id"]
        train, test = lookup.loc[train_ids], lookup.loc[test_ids]
        assert set(train.index).isdisjoint(test.index)
        assert len(train) + len(test) == len(data)
        x, y = train[features].to_numpy(), train.Survival_Group.to_numpy().astype(int)
        xt, yt = test[features].to_numpy(), test.Survival_Group.to_numpy().astype(int)
        cohort_record = {"cohort": cohort_name, "n": len(data), "short": int(data.Survival_Group.sum()),
                         "n_train": len(y), "train_short": int(y.sum()), "n_test": len(yt),
                         "test_short": int(yt.sum()), "test_prevalence": float(yt.mean()), **metadata[cohort_name]}
        cohort_records.append(cohort_record)
        pd.DataFrame(cohort_records).to_csv(args.output / "s2_revised_cohorts.csv", index=False)
        data.to_csv(private / f"{cohort_name}.csv", index=False)
        if min(np.bincount(y, minlength=2)) < 5 or min(np.bincount(yt, minlength=2)) < 1:
            status.append({"cohort": cohort_name, "status": "infeasible_predeclared_validation",
                           "reason": "Fewer than five development cases in an outcome class or single-class TEST."})
            print(cohort_name, "infeasible for predeclared validation", flush=True)
            continue
        folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(x, y))
        for name in NAMES:
            cfg = configurations[name]
            started = time.perf_counter()
            context = {"cohort": cohort_name, "classifier": name,
                       "resampling": cfg["imbalance_technique"] or "None"}
            oof_y, oof_p, oof_seen = [], [], np.zeros(len(y), dtype=int)
            phase = "initialisation"
            try:
                for fold, (tr, va) in enumerate(folds):
                    phase = f"fold_{fold}"
                    with warnings.catch_warnings(record=True) as caught, threadpool_limits(args.jobs):
                        warnings.simplefilter("always")
                        probability, prediction = fit_predict(factory, name, cfg["best_params"],
                                                              cfg["imbalance_technique"], x[tr], y[tr], x[va], args.jobs)
                    fit_count += 1
                    warning_records.extend({**context, "phase": phase, "category": w.category.__name__,
                                            "message": str(w.message)} for w in caught)
                    fold_records.append({**context, "fold": fold,
                                         "roc_auc": roc_auc_score(y[va], probability),
                                         "average_precision": average_precision_score(y[va], probability)})
                    oof_y.extend(y[va].tolist())
                    oof_p.extend(probability.tolist())
                    oof_seen[va] += 1
                assert np.all(oof_seen == 3)
                selected = thresholds(np.array(oof_y), np.array(oof_p))
                phase = "final_development_fit"
                with warnings.catch_warnings(record=True) as caught, threadpool_limits(args.jobs):
                    warnings.simplefilter("always")
                    probability, prediction = fit_predict(factory, name, cfg["best_params"],
                                                          cfg["imbalance_technique"], x, y, xt, args.jobs)
                fit_count += 1
                warning_records.extend({**context, "phase": phase, "category": w.category.__name__,
                                        "message": str(w.message)} for w in caught)
                row = {**context, "n_test": len(yt), "test_short": int(yt.sum()),
                       "test_prevalence": float(yt.mean()), **metrics(yt, probability, prediction),
                       **intervals(yt, probability, prediction, args.bootstrap)}
                constrained = (probability >= selected["sensitivity_0_90"]).astype(int)
                constrained_metrics = metrics(yt, probability, constrained)
                row.update({"sensitivity_threshold": selected["sensitivity_0_90"],
                            "constrained_sensitivity": constrained_metrics["sensitivity"],
                            "constrained_specificity": constrained_metrics["specificity"],
                            "seconds": time.perf_counter() - started})
                summary.append(row)
                all_thresholds.append({**context, "rule": "native", "threshold": None,
                                       **metrics(yt, probability, prediction)})
                for rule, value in selected.items():
                    all_thresholds.append({**context, "rule": rule, "threshold": value,
                                           **metrics(yt, probability, probability >= value)})
                pd.DataFrame({"subject_id": test.index, "y_true": yt, "probability": probability,
                              "native_prediction": prediction}).to_csv(
                    private / f"predictions_{cohort_name}_{name}.csv", index=False)
                status.append({**context, "status": "completed", "folds": 15})
                print(cohort_name, name, f"AUC={row['roc_auc']:.4f}",
                      f"recall={row['sensitivity']:.3f}", flush=True)
            except Exception:
                failure = {**context, "phase": phase, "status": "failed",
                           "traceback": traceback.format_exc()}
                status.append(failure)
                with (args.output / "s2_revised_failures.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(failure) + "\n")
                print(cohort_name, name, "FAILED", phase, flush=True)
            pd.DataFrame(summary).to_csv(args.output / "s2_revised_metrics.csv", index=False)
            pd.DataFrame(all_thresholds).to_csv(args.output / "s2_revised_thresholds.csv", index=False)
            pd.DataFrame(fold_records).to_csv(args.output / "s2_revised_folds.csv", index=False)
        pd.DataFrame(cohort_records).to_csv(args.output / "s2_revised_cohorts.csv", index=False)
    fingerprints = {str(p.relative_to(STUDY)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [config_path, STUDY / "01_data/processed/step3_coded_unscaled.csv",
                              STUDY / "01_data/processed/step4_train_unscaled.csv",
                              STUDY / "01_data/processed/step4_test_unscaled.csv"]}
    (args.output / "s2_revised_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (args.output / "s2_revised_warnings.json").write_text(json.dumps(warning_records, indent=2), encoding="utf-8")
    (args.output / "s2_revised_metadata.json").write_text(json.dumps({
        "input_sha256": fingerprints, "n_completed_fits": fit_count, "bootstrap_resamples": args.bootstrap,
        "cohort_details": metadata, "original_results_replaced": False, "new_optuna_search": False,
        "entry_timestamp": "Minus Onset_Delta / 30.44, a source-trial-reference proxy.",
        "threshold_source": "Concatenated predictions from 15 development folds, no TEST optimisation.",
        "warning_count": len(warning_records),
        "limitations": ["Patient-level split, not trial-level validation.",
                        "Fixed parameters selected in the original development population.",
                        "Landmarks predict conditional survival to month 24, not prognosis at diagnosis.",
                        "Complete-case and trial recruitment selection remain.",
                        "No known dated riluzole record is coded as no recorded use."]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
