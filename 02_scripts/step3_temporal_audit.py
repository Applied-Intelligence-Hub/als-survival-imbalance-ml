"""Audit Study II source-record timing relative to diagnosis."""

import os

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, '..', '01_data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, '..', '01_data', 'processed')
OUT_DIR = os.path.join(BASE_DIR, '..', '03_outputs', 'step3')
os.makedirs(OUT_DIR, exist_ok=True)


def closest_delta(records, delta_column, diagnosis):
    records = records.copy()
    records[delta_column] = records[delta_column].fillna(0)
    records = records.merge(diagnosis, on='subject_id', how='inner')
    records['_distance'] = (
        records[delta_column] - records['Diagnosis_Delta']).abs()
    selected = records.loc[
        records.groupby('subject_id')['_distance'].idxmin(),
        ['subject_id', delta_column, 'Diagnosis_Delta'],
    ].copy()
    selected['days_from_diagnosis'] = (
        selected[delta_column] - selected['Diagnosis_Delta'])
    return selected[['subject_id', 'days_from_diagnosis']]


cohort = pd.read_csv(
    os.path.join(PROCESSED_DIR, 'step3_final_dataset.csv'),
    usecols=['subject_id'],
)
history = pd.read_csv(os.path.join(RAW_DIR, 'PROACT_ALSHISTORY.csv'))
diagnosis = (
    history.dropna(subset=['Diagnosis_Delta'])
    [['subject_id', 'Diagnosis_Delta']]
    .drop_duplicates('subject_id')
)

alsfrs = pd.read_csv(
    os.path.join(RAW_DIR, 'PROACT_ALSFRS.csv'),
    usecols=['subject_id', 'ALSFRS_Delta'],
)
alsfrs_selected = closest_delta(alsfrs, 'ALSFRS_Delta', diagnosis)

fvc = pd.read_csv(os.path.join(RAW_DIR, 'PROACT_FVC.csv'))
fvc['FVC_Pct'] = fvc[
    ['pct_of_Normal_Trial_1', 'pct_of_Normal_Trial_2',
     'pct_of_Normal_Trial_3']
].max(axis=1)
fvc['_max_liters'] = fvc[
    ['Subject_Liters_Trial_1', 'Subject_Liters_Trial_2',
     'Subject_Liters_Trial_3']
].max(axis=1)
fallback = (
    fvc['FVC_Pct'].isna() & fvc['_max_liters'].notna()
    & fvc['subject_normal'].notna() & fvc['subject_normal'].gt(0)
)
fvc.loc[fallback, 'FVC_Pct'] = (
    fvc.loc[fallback, '_max_liters']
    / fvc.loc[fallback, 'subject_normal'] * 100
)
fvc_selected = closest_delta(
    fvc.dropna(subset=['FVC_Pct']),
    'Forced_Vital_Capacity_Delta',
    diagnosis,
)

vitals = pd.read_csv(os.path.join(RAW_DIR, 'PROACT_VITALSIGNS.csv'))
vitals['Weight_kg'] = vitals['Weight']
pounds = (
    vitals['Weight_Units'].eq('Pounds')
    | (vitals['Weight_Units'].isna() & vitals['Weight'].gt(200))
)
vitals.loc[pounds, 'Weight_kg'] = vitals.loc[pounds, 'Weight'] * 0.453592
vitals.loc[vitals['Weight_kg'].lt(25), 'Weight_kg'] = np.nan
weight_selected = closest_delta(
    vitals.dropna(subset=['Weight_kg']), 'Vital_Signs_Delta', diagnosis)

sources = {
    'ALSFRS assessment': alsfrs_selected,
    'FVC assessment': fvc_selected,
    'Weight assessment': weight_selected,
}
rows = []
merged_offsets = cohort.copy()
for source, selected in sources.items():
    column = source.split()[0] + '_days_from_diagnosis'
    selected = selected.rename(columns={'days_from_diagnosis': column})
    merged_offsets = merged_offsets.merge(selected, on='subject_id', how='left')
    offsets = cohort.merge(selected, on='subject_id', how='left')[column].dropna()
    rows.append({
        'source': source,
        'available_n': len(offsets),
        'before_or_at_diagnosis_n': int((offsets <= 0).sum()),
        'after_diagnosis_n': int((offsets > 0).sum()),
        'after_diagnosis_pct': float((offsets > 0).mean() * 100),
        'median_days_from_diagnosis': float(offsets.median()),
        'q1_days_from_diagnosis': float(offsets.quantile(0.25)),
        'q3_days_from_diagnosis': float(offsets.quantile(0.75)),
    })

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT_DIR, 'step3_temporal_audit.csv'), index=False)

offset_columns = [
    'ALSFRS_days_from_diagnosis',
    'FVC_days_from_diagnosis',
    'Weight_days_from_diagnosis',
]
complete_offsets = merged_offsets[offset_columns].dropna()
overall = pd.DataFrame([{
    'final_cohort_n': len(cohort),
    'all_three_available_n': len(complete_offsets),
    'all_three_before_or_at_diagnosis_n': int(
        (complete_offsets <= 0).all(axis=1).sum()),
}])
overall.to_csv(
    os.path.join(OUT_DIR, 'step3_temporal_audit_overall.csv'), index=False)

print(summary.to_string(index=False))
print()
print(overall.to_string(index=False))
