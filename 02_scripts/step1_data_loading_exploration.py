"""
Step 1 — Data Loading & Exploration
====================================
Load all relevant PRO-ACT CSV files and perform initial exploratory analysis.
Produces summary tables and prints key statistics to understand the raw data
before preprocessing.

Relevant tables (based on Papaiz et al., 2024):
  - DEMOGRAPHICS  : Age, Sex
  - ALSHISTORY    : Site of Onset, Disease Duration, Onset/Diagnosis deltas
  - ALSFRS        : Functional scores Q1-Q10 (+ ALSFRS-R Q10/Dyspnea)
  - FVC           : Forced Vital Capacity (% of normal)
  - SVC           : Slow Vital Capacity (will be excluded — ~87% missing)
  - VITALSIGNS    : Weight, Height (for BMI calculation)
  - RILUZOLE      : Riluzole usage
  - FAMILYHISTORY : Family history of ALS
  - DEATHDATA     : Death records (for target variable)
  - ELESCORIAL    : El Escorial criteria (will be excluded — ~71% missing)
"""

import pandas as pd
import numpy as np
import os
import sys
import time

# ── Configuration ──────────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', '01_data', 'raw')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '03_outputs', 'step1')
os.makedirs(OUT_DIR, exist_ok=True)

start_time = time.time()

# ── 1. Load all relevant CSVs ─────────────────────────────────────────────
print("=" * 70)
print("STEP 1 — DATA LOADING & EXPLORATION")
print("=" * 70)

tables = {
    'demographics':   'PROACT_DEMOGRAPHICS.csv',
    'alshistory':     'PROACT_ALSHISTORY.csv',
    'alsfrs':         'PROACT_ALSFRS.csv',
    'fvc':            'PROACT_FVC.csv',
    'svc':            'PROACT_SVC.csv',
    'vitalsigns':     'PROACT_VITALSIGNS.csv',
    'riluzole':       'PROACT_RILUZOLE.csv',
    'familyhistory':  'PROACT_FAMILYHISTORY.csv',
    'deathdata':      'PROACT_DEATHDATA.csv',
    'elescorial':     'PROACT_ELESCORIAL.csv',
}

data = {}
for name, filename in tables.items():
    filepath = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(filepath)
    data[name] = df
    print(f"\n{'─' * 70}")
    print(f"Table: {filename}")
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}  |  Unique subjects: {df['subject_id'].nunique():,}")

# ── 2. Summary of each table ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("DETAILED SUMMARY PER TABLE")
print("=" * 70)

summary_rows = []

for name, df in data.items():
    n_rows = len(df)
    n_cols = len(df.columns)
    n_subjects = df['subject_id'].nunique()
    missing_pct = (df.isnull().sum().sum() / (n_rows * n_cols)) * 100

    summary_rows.append({
        'Table': tables[name],
        'Rows': n_rows,
        'Columns': n_cols,
        'Unique Subjects': n_subjects,
        'Overall Missing (%)': round(missing_pct, 1),
    })

summary_df = pd.DataFrame(summary_rows)
print("\n" + summary_df.to_string(index=False))

# ── 3. Detailed missing values per column ──────────────────────────────────
print("\n" + "=" * 70)
print("MISSING VALUES PER COLUMN (KEY TABLES)")
print("=" * 70)

key_tables = ['demographics', 'alshistory', 'alsfrs', 'fvc', 'vitalsigns',
              'deathdata', 'riluzole']

for name in key_tables:
    df = data[name]
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    miss_df = pd.DataFrame({
        'Column': missing.index,
        'Missing': missing.values,
        'Missing (%)': missing_pct.values
    })
    miss_df = miss_df[miss_df['Missing'] > 0].sort_values('Missing (%)', ascending=False)
    print(f"\n{'─' * 50}")
    print(f"  {tables[name]}")
    print(f"{'─' * 50}")
    if len(miss_df) == 0:
        print("  No missing values.")
    else:
        print(miss_df.to_string(index=False))

# ── 4. Subject overlap analysis ───────────────────────────────────────────
print("\n" + "=" * 70)
print("SUBJECT OVERLAP ANALYSIS")
print("=" * 70)

all_subjects = {}
for name, df in data.items():
    all_subjects[name] = set(df['subject_id'].unique())

# Total unique subjects across all tables
all_unique = set()
for s in all_subjects.values():
    all_unique |= s
print(f"\nTotal unique subjects across all tables: {len(all_unique):,}")

# Subjects present in EACH key table
key_sets = ['demographics', 'alshistory', 'alsfrs', 'fvc', 'vitalsigns',
            'deathdata', 'riluzole']
common = all_subjects[key_sets[0]]
for k in key_sets[1:]:
    common = common & all_subjects[k]
print(f"Subjects present in ALL key tables:      {len(common):,}")

# Subjects with death data
print(f"Subjects with death records:             {len(all_subjects['deathdata']):,}")

# Show subject overlaps in pairwise fashion for important pairs
print(f"\nPairwise overlaps with DEATHDATA:")
for name in key_sets:
    if name != 'deathdata':
        overlap = all_subjects[name] & all_subjects['deathdata']
        print(f"  {name:20s} ∩ deathdata = {len(overlap):,}")

# ── 5. Key statistics for feature extraction planning ──────────────────────
print("\n" + "=" * 70)
print("KEY STATISTICS FOR FEATURE EXTRACTION")
print("=" * 70)

# DEMOGRAPHICS
demo = data['demographics']
print(f"\n--- DEMOGRAPHICS ---")
print(f"  Age: available for {demo['Age'].notna().sum():,}/{len(demo):,} subjects "
      f"({demo['Age'].notna().mean()*100:.1f}%)")
print(f"  Age range: {demo['Age'].min():.0f} - {demo['Age'].max():.0f} years")
print(f"  Sex distribution:")
print(f"    {demo['Sex'].value_counts().to_string()}")

# ALS HISTORY
hist = data['alshistory']
print(f"\n--- ALS HISTORY ---")
print(f"  Site_of_Onset available: {hist['Site_of_Onset'].notna().sum():,}/{len(hist):,}")
print(f"  Site of Onset distribution:")
if hist['Site_of_Onset'].notna().any():
    print(f"    {hist['Site_of_Onset'].value_counts().head(10).to_string()}")
print(f"  Onset_Delta available:     {hist['Onset_Delta'].notna().sum():,}")
print(f"  Diagnosis_Delta available: {hist['Diagnosis_Delta'].notna().sum():,}")
# Disease duration from Onset_Delta and Diagnosis_Delta
has_both = hist.dropna(subset=['Onset_Delta', 'Diagnosis_Delta'])
if len(has_both) > 0:
    dur = has_both['Diagnosis_Delta'] - has_both['Onset_Delta']
    print(f"  Computed Disease Duration (Diagnosis - Onset delta) for {len(has_both):,} records")
    print(f"    Mean: {dur.mean():.1f} days | Median: {dur.median():.1f} days")

# ALSFRS — check how many have ALSFRS vs ALSFRS-R
alsfrs = data['alsfrs']
print(f"\n--- ALSFRS ---")
has_alsfrs_total = alsfrs['ALSFRS_Total'].notna().sum()
has_alsfrsr_total = alsfrs['ALSFRS_R_Total'].notna().sum()
total_rows = len(alsfrs)
print(f"  Total rows: {total_rows:,}")
print(f"  Rows with ALSFRS_Total:   {has_alsfrs_total:,} ({has_alsfrs_total/total_rows*100:.1f}%)")
print(f"  Rows with ALSFRS_R_Total: {has_alsfrsr_total:,} ({has_alsfrsr_total/total_rows*100:.1f}%)")
# Check at diagnosis (delta closest to 0)
at_diag = alsfrs[alsfrs['ALSFRS_Delta'].abs() <= 7]  # within ~1 week of baseline
print(f"  Rows near diagnosis (|delta| <= 7 days): {len(at_diag):,} "
      f"({at_diag['subject_id'].nunique():,} subjects)")

# ALSFRS Q5: Gastrostomy presence
has_gastrostomy = alsfrs['Q5b_Cutting_with_Gastrostomy'].notna().sum()
print(f"  Rows with Q5b (gastrostomy): {has_gastrostomy:,} ({has_gastrostomy/total_rows*100:.1f}%)")

# FVC
fvc = data['fvc']
print(f"\n--- FVC ---")
print(f"  Total rows: {len(fvc):,}  |  Unique subjects: {fvc['subject_id'].nunique():,}")
print(f"  pct_of_Normal_Trial_1 available: {fvc['pct_of_Normal_Trial_1'].notna().sum():,} "
      f"({fvc['pct_of_Normal_Trial_1'].notna().mean()*100:.1f}%)")
fvc_diag = fvc[fvc['Forced_Vital_Capacity_Delta'].abs() <= 7]
print(f"  FVC near diagnosis (|delta| <= 7 days): {len(fvc_diag):,} "
      f"({fvc_diag['subject_id'].nunique():,} subjects)")

# VITALSIGNS
vs = data['vitalsigns']
print(f"\n--- VITAL SIGNS ---")
print(f"  Total rows: {len(vs):,}  |  Unique subjects: {vs['subject_id'].nunique():,}")
print(f"  Height available: {vs['Height'].notna().sum():,} ({vs['Height'].notna().mean()*100:.1f}%)")
print(f"  Weight available: {vs['Weight'].notna().sum():,} ({vs['Weight'].notna().mean()*100:.1f}%)")
vs_diag = vs[vs['Vital_Signs_Delta'].abs() <= 7]
print(f"  Vital signs near diagnosis (|delta| <= 7): {len(vs_diag):,} "
      f"({vs_diag['subject_id'].nunique():,} subjects)")

# DEATH DATA
death = data['deathdata']
print(f"\n--- DEATH DATA ---")
print(f"  Total records: {len(death):,}  |  Unique subjects: {death['subject_id'].nunique():,}")
print(f"  Subject_Died distribution:")
print(f"    {death['Subject_Died'].value_counts().to_string()}")
print(f"  Death_Days available: {death['Death_Days'].notna().sum():,} "
      f"({death['Death_Days'].notna().mean()*100:.1f}%)")
if death['Death_Days'].notna().any():
    print(f"  Death_Days range: {death['Death_Days'].min():.0f} - {death['Death_Days'].max():.0f} days")
    print(f"  Death_Days mean: {death['Death_Days'].mean():.0f} days "
          f"(~{death['Death_Days'].mean()/30.44:.1f} months)")

# RILUZOLE
ril = data['riluzole']
print(f"\n--- RILUZOLE ---")
print(f"  Total records: {len(ril):,}  |  Unique subjects: {ril['subject_id'].nunique():,}")
print(f"  Usage distribution:")
print(f"    {ril['Subject_used_Riluzole'].value_counts().to_string()}")

# SVC & EL ESCORIAL (these will be excluded)
print(f"\n--- SVC (to be EXCLUDED — high missing) ---")
print(f"  Unique subjects: {data['svc']['subject_id'].nunique():,} "
      f"out of {len(all_unique):,} total ({data['svc']['subject_id'].nunique()/len(all_unique)*100:.1f}%)")

print(f"\n--- EL ESCORIAL (to be EXCLUDED — high missing) ---")
print(f"  Unique subjects: {data['elescorial']['subject_id'].nunique():,} "
      f"out of {len(all_unique):,} total ({data['elescorial']['subject_id'].nunique()/len(all_unique)*100:.1f}%)")

# ── 6. Save summary to CSV ────────────────────────────────────────────────
summary_df.to_csv(os.path.join(OUT_DIR, 'table_summary.csv'), index=False)

# Save detailed column info for each table
for name, df in data.items():
    col_info = pd.DataFrame({
        'column': df.columns,
        'dtype': df.dtypes.values,
        'non_null': df.notna().sum().values,
        'null_count': df.isnull().sum().values,
        'null_pct': (df.isnull().sum() / len(df) * 100).round(1).values,
        'n_unique': [df[c].nunique() for c in df.columns],
    })
    col_info.to_csv(os.path.join(OUT_DIR, f'columns_{name}.csv'), index=False)

elapsed = time.time() - start_time
print(f"\n{'=' * 70}")
print(f"Step 1 completed in {elapsed:.1f}s")
print(f"Summary files saved to: {os.path.abspath(OUT_DIR)}")
print(f"{'=' * 70}")
