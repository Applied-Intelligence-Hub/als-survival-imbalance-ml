"""Compare ranking under the original single/10-bag protocol, without tuning.

Uses DEV only, the saved baseline parameters and the original 15 folds.
Only aggregate fold scores are exported; primary outputs are never replaced.
"""

import argparse
import hashlib
import importlib.metadata
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier


ROOT = Path(__file__).resolve().parents[1]


def estimator(name, parameters):
    p = dict(parameters)
    if name == "MLP" and isinstance(p.get("hidden_layer_sizes"), list):
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
    factories = {
        "KNN": lambda: KNeighborsClassifier(**p),
        "NaiveBayes": lambda: GaussianNB(**p),
        "DecisionTree": lambda: DecisionTreeClassifier(random_state=42, **p),
        "RandomForest": lambda: RandomForestClassifier(random_state=42, n_jobs=1, **p),
        "SVM": lambda: SVC(probability=True, random_state=42, **p),
        "MLP": lambda: MLPClassifier(random_state=42, **p),
        "LightGBM": lambda: LGBMClassifier(random_state=42, verbose=-1, n_jobs=1, **p),
    }
    return factories[name]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "03_outputs/reviewer_followup")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = ROOT / "01_data/processed/step4_arrays_unscaled.npz"
    arrays = np.load(path, allow_pickle=True)
    x, y = arrays["X_train"], arrays["y_train"]
    assert x.shape == (1201, 23) and int(y.sum()) == 139
    params_path = ROOT / "03_outputs/step5/step5_cv_results.csv"
    configurations = pd.read_csv(params_path).fillna({"imbalance_technique": "None"})
    configurations = configurations.loc[configurations.imbalance_technique.eq("None")]
    names = ["KNN", "NaiveBayes", "DecisionTree", "RandomForest", "SVM", "MLP", "LightGBM"]
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(x,y))
    scores = []
    for name in names:
        params = json.loads(configurations.loc[configurations.classifier.eq(name), "best_params"].iloc[0])
        for fold, (tr, va) in enumerate(folds):
            scaler = MinMaxScaler()
            xt, xv = scaler.fit_transform(x[tr]), scaler.transform(x[va])
            for mode in ["single", "bagging"]:
                model = estimator(name, params)
                if mode == "bagging":
                    model = BalancedBaggingClassifier(estimator=model, n_estimators=10, random_state=42, n_jobs=1)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(xt,y[tr])
                    p = model.predict_proba(xv)[:,1]
                    pred = model.predict(xv)
                scores.append(dict(classifier=name, fold=fold, mode=mode,
                                   roc_auc=roc_auc_score(y[va],p),
                                   balanced_accuracy=balanced_accuracy_score(y[va],pred),
                                   convergence_warnings=sum(issubclass(w.category, ConvergenceWarning) for w in caught)))
            print(f"{name}: fold {fold+1}/15", flush=True)
        pd.DataFrame(scores).to_csv(args.output / "review_s2_bagging_folds.csv", index=False)
    df = pd.DataFrame(scores)
    lines = [r"\begin{table}[ht!]", r"\centering\small",
             r"\caption[BalancedBagging and ranking discrimination]{Supplementary comparison of ROC-AUC and default-operating-point balanced accuracy on the original Study II development folds.}",
             r"\label{tab:review_s2_bagging_auc}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{ROC-AUC} & \multicolumn{2}{c}{Balanced accuracy} \\",
             r"Classifier & Single & Bagging & Single & Bagging \\", r"\midrule"]
    summary = []
    for name in names:
        row = [name]
        record = {"classifier":name}
        for metric in ["roc_auc", "balanced_accuracy"]:
            for mode in ["single","bagging"]:
                vals = df.loc[df.classifier.eq(name)&df["mode"].eq(mode),metric]
                row.append(f"${vals.mean():.3f}\\pm{vals.std(ddof=0):.3f}$")
                record[f"{mode}_{metric}"] = float(vals.mean())
        lines.append(" & ".join(row) + r" \\")
        summary.append(record)
    lines.extend([r"\bottomrule\end{tabular}", r"\par\smallskip",
                  r"\begin{minipage}{0.98\linewidth}\footnotesize",
                  r"Means $\pm$ population standard deviations across the original 15 folds. Parameters, folds and ten-estimator bagging design are unchanged; no hyperparameter search was repeated. Fold training sets overlap, so these are descriptive paired summaries, not independent replications.",
                  r"\end{minipage}", r"\end{table}"])
    (args.output / "review_s2_bagging_auc.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    pd.DataFrame(summary).to_csv(args.output / "review_s2_bagging_summary.csv", index=False)
    metadata = {"input_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,params_path]},
                "packages":{p:importlib.metadata.version(p) for p in ["numpy","pandas","scikit-learn","imbalanced-learn","lightgbm"]},
                "n_fits_single":105,"n_fits_bags":1050,
                "convergence_warnings":int(df.convergence_warnings.sum())}
    (args.output / "review_s2_bagging_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
