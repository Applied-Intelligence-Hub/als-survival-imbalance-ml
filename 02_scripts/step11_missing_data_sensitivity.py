"""Focused missing-data sensitivity analysis for Study II.

The selected LightGBM+ROS configuration is evaluated without retuning under
three cohort/preprocessing conditions: the primary complete-case analysis,
median imputation on the expanded eligible cohort, and five stochastic
iterative imputations on that expanded cohort. All preprocessing is fitted on
training data only. The expanded-cohort results are a post-hoc sensitivity
analysis and do not replace the primary replication cohort.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier


BASE_DIR = os.path.dirname(__file__)
INTERIM_DIR = os.path.join(BASE_DIR, '..', '01_data', 'interim')
PROCESSED_DIR = os.path.join(BASE_DIR, '..', '01_data', 'processed')
STEP5_DIR = os.path.join(BASE_DIR, '..', '03_outputs', 'step5')
OUT_DIR = os.path.join(BASE_DIR, '..', '03_outputs', 'step11')
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_IMPUTATIONS = 5

warnings.filterwarnings(
    'ignore', message='X does not have valid feature names')


def build_expanded_cohort():
    df = pd.read_csv(os.path.join(INTERIM_DIR, 'step2_patient_features.csv'))

    short = df['Event_Dead'] & df['Event_Dead_Time_from_Onset'].le(24)
    non_short_dead = df['Event_Dead'] & df['Event_Dead_Time_from_Onset'].gt(24)
    non_short_alive = (~df['Event_Dead']) & df['Last_Visit_from_Onset'].ge(24)
    df['Survival_Group'] = np.select(
        [short, non_short_dead | non_short_alive], [1.0, 0.0], default=np.nan)
    df = df[df['Survival_Group'].notna()].copy()
    df = df[~df['Site_of_Onset'].isin(['Limb_and_Bulbar', 'Other'])].copy()

    coded = pd.DataFrame(index=df.index)
    coded['Sex_Male'] = df['Sex'].map({'Female': 0.0, 'Male': 1.0})
    coded['Site_Onset'] = df['Site_of_Onset'].map(
        {'Bulbar': 0.0, 'Limb/Spinal': 1.0})
    coded['Age_at_Onset'] = pd.cut(
        df['Age_at_Onset'], [-np.inf, 39, 49, 59, 69, np.inf],
        labels=False,
    )
    coded['Riluzole'] = df['Riluzole'].map({'No': 0.0, 'Yes': 1.0})
    coded['Diagnosis_Delay'] = pd.cut(
        df['Diagnosis_Delay'], [-np.inf, 8, 18, np.inf], labels=False)
    coded['FVC_at_Diagnosis'] = (
        df['FVC_at_Diagnosis'].lt(80).where(df['FVC_at_Diagnosis'].notna())
        .astype(float)
    )
    coded['BMI_at_Diagnosis'] = pd.cut(
        df['BMI_at_Diagnosis'], [-np.inf, 18.5, 25, 30, np.inf],
        labels=False, right=False,
    )

    direct = {
        'Patient_with_Gastrostomy_at_Diagnosis': 'Patient_with_Gastrostomy',
        'Qty_Regions_Involved_at_Diagnosis': 'Qty_Regions_Involved',
        'Region_Bulbar_at_Diagnosis': 'Region_Involved_Bulbar',
        'Region_Upper_Limb_at_Diagnosis': 'Region_Involved_Upper_Limb',
        'Region_Lower_Limb_at_Diagnosis': 'Region_Involved_Lower_Limb',
        'Region_Respiratory_at_Diagnosis': 'Region_Involved_Respiratory',
    }
    for source, target in direct.items():
        coded[target] = pd.to_numeric(df[source], errors='coerce')

    slopes = {
        'Slope_Q1_Speech_at_Diagnosis': 'Q1_Speech_slope',
        'Slope_Q2_Salivation_at_Diagnosis': 'Q2_Salivation_slope',
        'Slope_Q3_Swallowing_at_Diagnosis': 'Q3_Swallowing_slope',
        'Slope_Q4_Handwriting_at_Diagnosis': 'Q4_Handwriting_slope',
        'Slope_Q5_Cutting_at_Diagnosis': 'Q5_Cutting_slope',
        'Slope_Q6_Dressing_and_Hygiene_at_Diagnosis': 'Q6_Dressing_slope',
        'Slope_Q7_Turning_in_Bed_at_Diagnosis': 'Q7_Turning_slope',
        'Slope_Q8_Walking_at_Diagnosis': 'Q8_Walking_slope',
        'Slope_Q9_Climbing_Stairs_at_Diagnosis': 'Q9_Climbing_slope',
        'Slope_Q10_Respiratory_at_Diagnosis': 'Q10_Respiratory_slope',
    }
    for source, target in slopes.items():
        coded[target] = pd.cut(
            df[source], [-np.inf, 0.05, 0.14, np.inf],
            labels=False, right=False,
        )

    coded = coded.astype(float)
    return coded, df['Survival_Group'].astype(int).to_numpy()


def load_model_params():
    with open(os.path.join(STEP5_DIR, 'step5_best_configs.json'), encoding='utf-8') as f:
        return json.load(f)['LightGBM']['best_params']


def fit_predict(X_train, y_train, X_test, params, random_state):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sampler = RandomOverSampler(random_state=random_state)
    X_train, y_train = sampler.fit_resample(X_train, y_train)
    model = LGBMClassifier(
        random_state=random_state, verbose=-1, n_jobs=-1, **params)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def constrain_to_codes(train, transformed):
    lower = np.nanmin(train, axis=0)
    upper = np.nanmax(train, axis=0)
    return np.rint(np.clip(transformed, lower, upper))


def metric_row(analysis, cohort_n, test_n, y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        'analysis': analysis,
        'cohort_n': cohort_n,
        'test_n': test_n,
        'test_short_n': int(y_true.sum()),
        'roc_auc': roc_auc_score(y_true, probabilities),
        'pr_auc': average_precision_score(y_true, probabilities),
        'balanced_accuracy': balanced_accuracy_score(y_true, predictions),
        'sensitivity': recall_score(y_true, predictions),
        'specificity': recall_score(y_true, predictions, pos_label=0),
        'precision': precision_score(y_true, predictions, zero_division=0),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }


def main():
    params = load_model_params()
    rows = []

    complete = np.load(
        os.path.join(PROCESSED_DIR, 'step4_arrays_unscaled.npz'),
        allow_pickle=True,
    )
    complete_prob = fit_predict(
        complete['X_train'], complete['y_train'], complete['X_test'],
        params, RANDOM_STATE,
    )
    rows.append(metric_row(
        'Complete-case primary analysis', 1502, len(complete['y_test']),
        complete['y_test'], complete_prob,
    ))

    X, y = build_expanded_cohort()
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    X_train = X.iloc[train_idx].to_numpy()
    X_test = X.iloc[test_idx].to_numpy()
    y_train, y_test = y[train_idx], y[test_idx]

    median = SimpleImputer(strategy='median')
    median_train = median.fit_transform(X_train)
    median_test = median.transform(X_test)
    median_prob = fit_predict(
        median_train, y_train, median_test, params, RANDOM_STATE)
    rows.append(metric_row(
        'Expanded cohort: median imputation', len(y), len(y_test),
        y_test, median_prob,
    ))

    imputed_probabilities = []
    for imputation in range(N_IMPUTATIONS):
        seed = RANDOM_STATE + imputation
        imputer = IterativeImputer(
            max_iter=10,
            sample_posterior=True,
            random_state=seed,
            skip_complete=True,
            keep_empty_features=True,
        )
        imp_train = imputer.fit_transform(X_train)
        imp_test = imputer.transform(X_test)
        imp_train = constrain_to_codes(X_train, imp_train)
        imp_test = constrain_to_codes(X_train, imp_test)
        imputed_probabilities.append(
            fit_predict(imp_train, y_train, imp_test, params, seed))
        print(f'Completed stochastic imputation {imputation + 1}/{N_IMPUTATIONS}')

    pooled_prob = np.mean(imputed_probabilities, axis=0)
    rows.append(metric_row(
        f'Expanded cohort: iterative imputation (m={N_IMPUTATIONS})',
        len(y), len(y_test), y_test, pooled_prob,
    ))

    results = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, 'step11_missing_data_sensitivity.csv')
    results.to_csv(path, index=False)
    print()
    print(results.to_string(index=False))
    print(f'\nSaved: {path}')


if __name__ == '__main__':
    main()
