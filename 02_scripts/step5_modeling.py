"""
Step 5 — Model Training with Optuna Hyperparameter Optimization
================================================================
For each (classifier × imbalance_technique) combination:
  1. Use Optuna to find best hyperparameters (optimize AUC)
  2. Evaluate best config with full metrics (AUC, F1, G-Mean, etc.)
  3. Save results incrementally

7 Classifiers: KNN, NaiveBayes, DecisionTree, RandomForest, SVM, MLP, LightGBM
10 Imbalance techniques: None, ROS, SMOTE, BorderlineSMOTE, ADASYN,
                         RUS, NearMiss, TomekLinks, SMOTETomek, SMOTEENN

CV: RepeatedStratifiedKFold(5 splits, 3 repeats)
Optimization metric: AUC (ROC-AUC)

Modifications from Papaiz et al. (2024):
  - Optuna replaces Grid Search
  - LightGBM added as 7th classifier
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import time
import warnings
import optuna

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score
)

from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
)
from imblearn.under_sampling import (
    RandomUnderSampler, NearMiss, TomekLinks
)
from imblearn.combine import SMOTETomek, SMOTEENN

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Configuration ──────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', '01_data', 'processed')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '03_outputs', 'step5')
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3
N_TRIALS = 50          # Optuna trials per combination
RESULTS_CSV = os.path.join(OUT_DIR, 'step5_cv_results.csv')

CLASSIFIERS = ['KNN', 'NaiveBayes', 'DecisionTree', 'RandomForest',
               'SVM', 'MLP', 'LightGBM']

IMBALANCE_TECHNIQUES = ['None', 'ROS', 'SMOTE', 'BorderlineSMOTE', 'ADASYN',
                        'RUS', 'NearMiss', 'TomekLinks', 'SMOTETomek', 'SMOTEENN']


# ══════════════════════════════════════════════════════════════════════════
# SEARCH SPACES — one function per classifier
# ══════════════════════════════════════════════════════════════════════════

def suggest_knn(trial):
    return {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 31),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
    }

def suggest_nb(trial):
    return {
        'var_smoothing': trial.suggest_float('var_smoothing', 1e-12, 1e-1, log=True),
    }

def suggest_dt(trial):
    return {
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 2, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
    }

def suggest_rf(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 2, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
    }

def suggest_svm(trial):
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    params = {
        'C': trial.suggest_float('C', 0.01, 100.0, log=True),
        'kernel': kernel,
    }
    if kernel in ('rbf', 'poly'):
        params['gamma'] = trial.suggest_float('gamma', 1e-3, 10.0, log=True)
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
    return params

def suggest_mlp(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = tuple(
        trial.suggest_int(f'n_units_l{i}', 16, 256)
        for i in range(n_layers)
    )
    return {
        'hidden_layer_sizes': layers,
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
        'learning_rate': trial.suggest_categorical('learning_rate',
                                                    ['constant', 'adaptive']),
        'max_iter': 1000,
    }

def suggest_lgbm(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
    }

SUGGEST_FN = {
    'KNN': suggest_knn, 'NaiveBayes': suggest_nb,
    'DecisionTree': suggest_dt, 'RandomForest': suggest_rf,
    'SVM': suggest_svm, 'MLP': suggest_mlp, 'LightGBM': suggest_lgbm,
}


# ══════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def make_classifier(name, params):
    """Create a classifier instance from name + params dict."""
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
        return MLPClassifier(random_state=RANDOM_STATE, **params)
    if name == 'LightGBM':
        return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
                              **params)
    raise ValueError(f"Unknown classifier: {name}")


def make_sampler(name):
    """Create a resampling object (or None)."""
    if name == 'None':
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
# CV EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════

def compute_fold_metrics(y_true, y_pred, y_prob):
    """Compute all metrics for one fold."""
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)     # sensitivity
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)    # specificity
    gmean = np.sqrt(rec * spec)
    return {'auc': auc, 'f1': f1, 'precision': prec,
            'recall': rec, 'specificity': spec, 'gmean': gmean}


def run_cv(clf_name, params, sampler_name, X, y, return_all=False):
    """Run RepeatedStratifiedKFold CV and return mean AUC (or all metrics)."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=RANDOM_STATE)
    sampler = make_sampler(sampler_name)
    fold_metrics = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Fit preprocessing within the training portion of each CV fold.
        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        # Apply resampling to training fold only
        if sampler is not None:
            try:
                X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)
            except Exception:
                # If resampling fails, skip this fold
                fold_metrics.append({'auc': 0.5, 'f1': 0, 'precision': 0,
                                     'recall': 0, 'specificity': 0, 'gmean': 0})
                continue

        clf = make_classifier(clf_name, params)
        try:
            clf.fit(X_tr, y_tr)
            y_prob = clf.predict_proba(X_val)[:, 1]
            y_pred = clf.predict(X_val)
        except Exception:
            fold_metrics.append({'auc': 0.5, 'f1': 0, 'precision': 0,
                                 'recall': 0, 'specificity': 0, 'gmean': 0})
            continue

        fold_metrics.append(compute_fold_metrics(y_val, y_pred, y_prob))

    if not fold_metrics:
        if return_all:
            return {m: 0.0 for m in ['auc', 'f1', 'precision', 'recall',
                                      'specificity', 'gmean']}, {}
        return 0.5

    results = {}
    stds = {}
    for metric in ['auc', 'f1', 'precision', 'recall', 'specificity', 'gmean']:
        vals = [fm[metric] for fm in fold_metrics]
        results[metric] = np.mean(vals)
        stds[metric] = np.std(vals)

    if return_all:
        return results, stds
    return results['auc']


# ══════════════════════════════════════════════════════════════════════════
# OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════

def create_objective(clf_name, sampler_name, X, y):
    """Create an Optuna objective function for a given combination."""
    def objective(trial):
        params = SUGGEST_FN[clf_name](trial)
        auc = run_cv(clf_name, params, sampler_name, X, y)
        return auc
    return objective


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    start_time = time.time()

    print("=" * 70)
    print("STEP 5 — MODEL TRAINING WITH OPTUNA OPTIMIZATION")
    print(f"  Classifiers:  {len(CLASSIFIERS)}")
    print(f"  Imbalance:    {len(IMBALANCE_TECHNIQUES)}")
    print(f"  Combinations: {len(CLASSIFIERS) * len(IMBALANCE_TECHNIQUES)}")
    print(f"  Optuna trials per combination: {N_TRIALS}")
    print(f"  CV: {N_SPLITS}-fold × {N_REPEATS} repeats = "
          f"{N_SPLITS * N_REPEATS} evaluations/trial")
    print("=" * 70)

    # Load data
    data = np.load(os.path.join(PROCESSED_DIR, 'step4_arrays_unscaled.npz'),
                   allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    feature_names = data['feature_names']

    print(f"\nTrain set: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    print(f"Target: Short={y_train.sum():.0f}, Non-Short={(y_train == 0).sum():.0f}")

    # Check for existing partial results (resume support)
    all_results = []
    done_combos = set()
    if os.path.exists(RESULTS_CSV):
        prev = pd.read_csv(RESULTS_CSV)
        # Normalize: NaN/empty → 'None' so resume matching works
        prev['imbalance_technique'] = (prev['imbalance_technique']
                                       .fillna('None').replace('', 'None'))
        # Drop duplicates from earlier interrupted runs (keep latest)
        prev = prev.drop_duplicates(subset=['classifier', 'imbalance_technique'],
                                    keep='last').reset_index(drop=True)
        all_results = prev.to_dict('records')
        done_combos = set(zip(prev['classifier'], prev['imbalance_technique']))
        print(f"\nResuming: {len(done_combos)} combinations already completed")

    total = len(CLASSIFIERS) * len(IMBALANCE_TECHNIQUES)
    combo_idx = 0

    for clf_name in CLASSIFIERS:
        for imb_name in IMBALANCE_TECHNIQUES:
            combo_idx += 1

            if (clf_name, imb_name) in done_combos:
                print(f"\n[{combo_idx}/{total}] {clf_name} + {imb_name} — SKIPPED (already done)")
                continue

            print(f"\n{'─' * 70}")
            print(f"[{combo_idx}/{total}] {clf_name} + {imb_name}")
            t0 = time.time()

            # Optuna study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
            )
            objective = create_objective(clf_name, imb_name, X_train, y_train)

            try:
                study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
            except Exception as e:
                print(f"  ⚠ Optimization failed: {e}")
                # Record failure
                row = {
                    'classifier': clf_name,
                    'imbalance_technique': imb_name,
                    'n_trials': 0,
                    'best_trial': -1,
                    'mean_auc': 0.0, 'std_auc': 0.0,
                    'mean_f1': 0.0, 'std_f1': 0.0,
                    'mean_gmean': 0.0, 'std_gmean': 0.0,
                    'mean_precision': 0.0, 'std_precision': 0.0,
                    'mean_recall': 0.0, 'std_recall': 0.0,
                    'mean_specificity': 0.0, 'std_specificity': 0.0,
                    'best_params': '{}',
                    'time_seconds': time.time() - t0,
                }
                all_results.append(row)
                pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)
                continue

            best_params = study.best_trial.params
            # Re-construct full params for classifier (handle MLP layers)
            if clf_name == 'MLP':
                n_layers = best_params.pop('n_layers')
                layers = tuple(best_params.pop(f'n_units_l{i}')
                               for i in range(n_layers))
                best_params['hidden_layer_sizes'] = layers
                best_params['max_iter'] = 1000

            # Full evaluation with all metrics
            means, stds = run_cv(clf_name, best_params, imb_name,
                                 X_train, y_train, return_all=True)

            elapsed = time.time() - t0

            row = {
                'classifier': clf_name,
                'imbalance_technique': imb_name,
                'n_trials': len(study.trials),
                'best_trial': study.best_trial.number,
                'mean_auc': round(means['auc'], 4),
                'std_auc': round(stds['auc'], 4),
                'mean_f1': round(means['f1'], 4),
                'std_f1': round(stds['f1'], 4),
                'mean_gmean': round(means['gmean'], 4),
                'std_gmean': round(stds['gmean'], 4),
                'mean_precision': round(means['precision'], 4),
                'std_precision': round(stds['precision'], 4),
                'mean_recall': round(means['recall'], 4),
                'std_recall': round(stds['recall'], 4),
                'mean_specificity': round(means['specificity'], 4),
                'std_specificity': round(stds['specificity'], 4),
                'best_params': json.dumps(best_params),
                'time_seconds': round(elapsed, 1),
            }
            all_results.append(row)

            print(f"  AUC={means['auc']:.4f}±{stds['auc']:.4f}  "
                  f"F1={means['f1']:.4f}  G-Mean={means['gmean']:.4f}  "
                  f"({elapsed:.1f}s)")

            # Save incrementally
            pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)

    # ══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("STEP 5 — RESULTS SUMMARY")
    print("=" * 70)

    # Best combination per classifier
    print("\nBest imbalance technique per classifier (by AUC):")
    print(f"  {'Classifier':<15s} {'Technique':<18s} {'AUC':>8s} {'F1':>8s} "
          f"{'G-Mean':>8s} {'Recall':>8s} {'Spec':>8s}")
    print("  " + "─" * 75)

    best_per_clf = []
    for clf in CLASSIFIERS:
        mask = results_df['classifier'] == clf
        if mask.sum() == 0:
            continue
        best_idx = results_df.loc[mask, 'mean_auc'].idxmax()
        r = results_df.loc[best_idx]
        print(f"  {r['classifier']:<15s} {r['imbalance_technique']:<18s} "
              f"{r['mean_auc']:>8.4f} {r['mean_f1']:>8.4f} "
              f"{r['mean_gmean']:>8.4f} {r['mean_recall']:>8.4f} "
              f"{r['mean_specificity']:>8.4f}")
        best_per_clf.append(r.to_dict())

    # Overall best
    overall_best = results_df.loc[results_df['mean_auc'].idxmax()]
    print(f"\n  ★ Overall best: {overall_best['classifier']} + "
          f"{overall_best['imbalance_technique']}  "
          f"(AUC={overall_best['mean_auc']:.4f})")

    # Save best configs for step 6
    best_configs = {}
    for r in best_per_clf:
        best_configs[r['classifier']] = {
            'imbalance_technique': r['imbalance_technique'],
            'best_params': json.loads(r['best_params']) if isinstance(r['best_params'], str) else r['best_params'],
            'mean_auc': r['mean_auc'],
        }
    with open(os.path.join(OUT_DIR, 'step5_best_configs.json'), 'w') as f:
        json.dump(best_configs, f, indent=2)

    # Save best-per-classifier table
    pd.DataFrame(best_per_clf).to_csv(
        os.path.join(OUT_DIR, 'step5_best_per_classifier.csv'), index=False)

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Step 5 completed in {total_time / 60:.1f} minutes")
    print(f"Results: {RESULTS_CSV}")
    print(f"Best configs: {os.path.join(OUT_DIR, 'step5_best_configs.json')}")
    print(f"{'=' * 70}")
