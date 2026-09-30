"""
Step 3 — Cleaning, Survival Grouping & Codification
=====================================================
Takes the patient-level raw features from Step 2 and prepares the
final modelling dataset following Papaiz et al. (2024):

  1. Compute Survival_Group (Short / Non-Short / Censored)
  2. Remove Censored patients (alive < 24 months observation)
  3. Remove Site_of_Onset = Limb_and_Bulbar & Other (paper's binary coding)
  4. Complete case analysis — drop rows with ANY missing feature
  5. Codify features to ordinal integers (Table 1 scheme)

Scaling is deliberately deferred until after the train/test split in Step 4.

Current extract: 1,502 patients, 23 features, IR 7.6.
"""

import pandas as pd
import numpy as np
import os
import time

# ── Configuration ──────────────────────────────────────────────────────────
INTERIM_DIR = os.path.join(os.path.dirname(__file__), '..', '01_data', 'interim')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', '01_data', 'processed')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '03_outputs', 'step3')
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

start_time = time.time()

print("=" * 70)
print("STEP 3 — CLEANING, GROUPING & CODIFICATION")
print("Following Papaiz et al. (2024) methodology")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD Step 2 output
# ══════════════════════════════════════════════════════════════════════════
df = pd.read_csv(os.path.join(INTERIM_DIR, 'step2_patient_features.csv'))
print(f"\nLoaded dataset: {len(df):,} patients × {len(df.columns)} columns")


# ══════════════════════════════════════════════════════════════════════════
# 2. COMPUTE SURVIVAL GROUP
#    Short:     Died ≤ 24 months from symptom onset
#    Non-Short: Died > 24 months from onset OR alive ≥ 24 months observation
#    Censored:  Alive with < 24 months observation (indeterminate)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("COMPUTING SURVIVAL GROUPS...")

THRESHOLD_MONTHS = 24

def assign_group(row):
    dead = row['Event_Dead']
    time_dead = row['Event_Dead_Time_from_Onset']
    time_alive = row['Last_Visit_from_Onset']

    if dead and pd.notna(time_dead):
        return 'Short' if time_dead <= THRESHOLD_MONTHS else 'Non-Short'
    if not dead and pd.notna(time_alive):
        return 'Non-Short' if time_alive >= THRESHOLD_MONTHS else 'Censored'
    return np.nan

df['Survival_Group'] = df.apply(assign_group, axis=1)

vc = df['Survival_Group'].value_counts(dropna=False)
print(f"  Survival Group distribution:")
for g, c in vc.items():
    print(f"    {str(g):<12s}: {c:>6,d}")


# ══════════════════════════════════════════════════════════════════════════
# 3. REMOVE CENSORED + NaN groups
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("REMOVING Censored and unclassifiable patients...")

n_before = len(df)
df = df[df['Survival_Group'].isin(['Short', 'Non-Short'])].copy()
print(f"  Removed {n_before - len(df):,} patients → {len(df):,} remaining")


# ══════════════════════════════════════════════════════════════════════════
# 4. REMOVE SITE = Limb_and_Bulbar & Other
#    Paper codes Site_Onset as binary Limb/Spinal vs Bulbar;
#    Limb_and_Bulbar and Other patients are excluded
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("REMOVING Site_of_Onset = Limb_and_Bulbar / Other...")

site_counts = df['Site_of_Onset'].value_counts(dropna=False)
print(f"  Before: {site_counts.to_dict()}")

n_before = len(df)
df = df[~df['Site_of_Onset'].isin(['Limb_and_Bulbar', 'Other'])].copy()
print(f"  Removed {n_before - len(df):,} → {len(df):,} remaining")


# ══════════════════════════════════════════════════════════════════════════
# 5. DEFINE THE 23 FEATURES + COMPLETE CASE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("COMPLETE CASE ANALYSIS...")

# Map step2 column names → final paper names
feature_mapping = {
    'Diagnosis_Delay':                      'Diagnosis_Delay',
    'Age_at_Onset':                         'Age_at_Onset',
    'Sex':                                  'Sex_Male',
    'Site_of_Onset':                        'Site_Onset',
    'Riluzole':                             'Riluzole',
    'FVC_at_Diagnosis':                     'FVC_at_Diagnosis',
    'BMI_at_Diagnosis':                     'BMI_at_Diagnosis',
    'Patient_with_Gastrostomy_at_Diagnosis': 'Patient_with_Gastrostomy',
    'Qty_Regions_Involved_at_Diagnosis':    'Qty_Regions_Involved',
    'Region_Bulbar_at_Diagnosis':           'Region_Involved_Bulbar',
    'Region_Upper_Limb_at_Diagnosis':       'Region_Involved_Upper_Limb',
    'Region_Lower_Limb_at_Diagnosis':       'Region_Involved_Lower_Limb',
    'Region_Respiratory_at_Diagnosis':      'Region_Involved_Respiratory',
    'Slope_Q1_Speech_at_Diagnosis':         'Q1_Speech_slope',
    'Slope_Q2_Salivation_at_Diagnosis':     'Q2_Salivation_slope',
    'Slope_Q3_Swallowing_at_Diagnosis':     'Q3_Swallowing_slope',
    'Slope_Q4_Handwriting_at_Diagnosis':    'Q4_Handwriting_slope',
    'Slope_Q5_Cutting_at_Diagnosis':        'Q5_Cutting_slope',
    'Slope_Q6_Dressing_and_Hygiene_at_Diagnosis': 'Q6_Dressing_slope',
    'Slope_Q7_Turning_in_Bed_at_Diagnosis': 'Q7_Turning_slope',
    'Slope_Q8_Walking_at_Diagnosis':        'Q8_Walking_slope',
    'Slope_Q9_Climbing_Stairs_at_Diagnosis':'Q9_Climbing_slope',
    'Slope_Q10_Respiratory_at_Diagnosis':   'Q10_Respiratory_slope',
}

src_cols = list(feature_mapping.keys())

# Missing report before drop
missing = df[src_cols].isnull().sum().sort_values(ascending=False)
print(f"\n  Missing values per feature (before drop):")
for feat, count in missing.items():
    if count > 0:
        pct = count / len(df) * 100
        print(f"    {feat}: {count:,} ({pct:.1f}%)")

n_before = len(df)
df = df.dropna(subset=src_cols).copy()
print(f"\n  Dropped {n_before - len(df):,} incomplete rows → {len(df):,} complete cases")


# ══════════════════════════════════════════════════════════════════════════
# 6. CODIFY FEATURES (Table 1 ordinal scheme)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("CODIFYING FEATURES (Table 1)...")

coded = pd.DataFrame()
coded['subject_id'] = df['subject_id'].values

# Sex: Male=1, Female=0
coded['Sex_Male'] = (df['Sex'] == 'Male').astype(int).values

# Site_Onset: Limb/Spinal=1, Bulbar=0
coded['Site_Onset'] = (df['Site_of_Onset'] == 'Limb/Spinal').astype(int).values

# Age_at_Onset: ordinal 0-4 (0-39 → 0, 40-49 → 1, 50-59 → 2, 60-69 → 3, 70+ → 4)
coded['Age_at_Onset'] = pd.cut(
    df['Age_at_Onset'].values,
    bins=[-np.inf, 39, 49, 59, 69, np.inf],
    labels=[0, 1, 2, 3, 4]
).astype(int)

# Riluzole: Yes=1, No=0
coded['Riluzole'] = (df['Riluzole'] == 'Yes').astype(int).values

# Diagnosis_Delay: Short≤8=0, Average 9-18=1, Long≥19=2
coded['Diagnosis_Delay'] = pd.cut(
    df['Diagnosis_Delay'].values,
    bins=[-np.inf, 8, 18, np.inf],
    labels=[0, 1, 2]
).astype(int)

# FVC: Abnormal<80%=1, Normal≥80%=0
coded['FVC_at_Diagnosis'] = (df['FVC_at_Diagnosis'].values < 80).astype(int)

# BMI: Underweight<18.5=0, Normal 18.5-25=1, Overweight 25-30=2, Obesity≥30=3
coded['BMI_at_Diagnosis'] = pd.cut(
    df['BMI_at_Diagnosis'].values,
    bins=[-np.inf, 18.5, 25, 30, np.inf],
    labels=[0, 1, 2, 3]
).astype(int)

# Gastrostomy: already 0/1
coded['Patient_with_Gastrostomy'] = df['Patient_with_Gastrostomy_at_Diagnosis'].astype(int).values

# Regions: Qty already 0-4; individual regions already 0/1
coded['Qty_Regions_Involved'] = df['Qty_Regions_Involved_at_Diagnosis'].astype(int).values
coded['Region_Involved_Bulbar'] = df['Region_Bulbar_at_Diagnosis'].astype(int).values
coded['Region_Involved_Upper_Limb'] = df['Region_Upper_Limb_at_Diagnosis'].astype(int).values
coded['Region_Involved_Lower_Limb'] = df['Region_Lower_Limb_at_Diagnosis'].astype(int).values
coded['Region_Involved_Respiratory'] = df['Region_Respiratory_at_Diagnosis'].astype(int).values

# ALSFRS slopes: Slow<0.05=0, Average 0.05-0.13=1, Rapid≥0.14=2
slope_mapping = {
    'Slope_Q1_Speech_at_Diagnosis':             'Q1_Speech_slope',
    'Slope_Q2_Salivation_at_Diagnosis':         'Q2_Salivation_slope',
    'Slope_Q3_Swallowing_at_Diagnosis':         'Q3_Swallowing_slope',
    'Slope_Q4_Handwriting_at_Diagnosis':        'Q4_Handwriting_slope',
    'Slope_Q5_Cutting_at_Diagnosis':            'Q5_Cutting_slope',
    'Slope_Q6_Dressing_and_Hygiene_at_Diagnosis': 'Q6_Dressing_slope',
    'Slope_Q7_Turning_in_Bed_at_Diagnosis':     'Q7_Turning_slope',
    'Slope_Q8_Walking_at_Diagnosis':            'Q8_Walking_slope',
    'Slope_Q9_Climbing_Stairs_at_Diagnosis':    'Q9_Climbing_slope',
    'Slope_Q10_Respiratory_at_Diagnosis':       'Q10_Respiratory_slope',
}
for src_col, dst_col in slope_mapping.items():
    coded[dst_col] = pd.cut(
        df[src_col].values,
        bins=[-np.inf, 0.05, 0.14, np.inf],
        labels=[0, 1, 2],
        right=False   # [0, 0.05)=Slow, [0.05, 0.14)=Average, [0.14, inf)=Rapid
    ).astype(int)

# Target
coded['Survival_Group'] = (df['Survival_Group'] == 'Short').astype(int).values

feature_names = [c for c in coded.columns if c not in ['subject_id', 'Survival_Group']]
print(f"  Codified {len(feature_names)} features to ordinal integers")

# Print codification summary
for feat in feature_names:
    vc = coded[feat].value_counts().sort_index()
    print(f"    {feat}: {vc.to_dict()}")


# ══════════════════════════════════════════════════════════════════════════
# FINAL STATISTICS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL DATASET STATISTICS")
print("=" * 70)

n_total = len(coded)
n_short = coded['Survival_Group'].sum()
n_nonshort = n_total - n_short
ir = n_nonshort / n_short if n_short > 0 else float('inf')
pct_short = n_short / n_total * 100
pct_nonshort = n_nonshort / n_total * 100

print(f"\n  Total patients:     {n_total:,}")
print(f"  Short (≤24 months): {n_short:,} ({pct_short:.1f}%)")
print(f"  Non-Short:          {n_nonshort:,} ({pct_nonshort:.1f}%)")
print(f"  Imbalance Ratio:    {ir:.1f}")
print(f"  Features:           {len(feature_names)}")

# Compare with article: 1,967 patients, 13% Short, IR 6.9
print(f"\n  ┌─ Comparison with Papaiz et al. (2024) ─────────────────┐")
print(f"  │  Article:  1,967 patients  │  13% Short  │  IR = 6.9    │")
print(f"  │  Ours:     {n_total:>5,} patients  │  {pct_short:>4.1f}% Short  │  IR = {ir:.1f}{'':>4s}│")
print(f"  └──────────────────────────────────────────────────────────┘")


# ══════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SAVING OUTPUTS...")

# Save the complete coded dataset. Step 4 splits these rows before scaling.
out_path = os.path.join(PROCESSED_DIR, 'step3_final_dataset.csv')
coded.to_csv(out_path, index=False)
print(f"  Final dataset → {out_path}")

# Keep the explicitly named copy used for Table 1 and sensitivity analyses.
coded_unscaled = coded.copy()
coded_unscaled.to_csv(
    os.path.join(PROCESSED_DIR, 'step3_coded_unscaled.csv'), index=False)
print(f"  Coded (unscaled) → {os.path.join(PROCESSED_DIR, 'step3_coded_unscaled.csv')}")

# Save summary stats
summary = {
    'total_patients': n_total,
    'short_count': int(n_short),
    'nonshort_count': int(n_nonshort),
    'short_pct': round(pct_short, 1),
    'nonshort_pct': round(pct_nonshort, 1),
    'imbalance_ratio': round(ir, 1),
    'n_features': len(feature_names),
}
pd.Series(summary).to_csv(os.path.join(OUT_DIR, 'step3_summary.csv'))

# Save Table 1 distribution (from unscaled coded data)
table1_rows = []
for feat in feature_names:
    for val in sorted(coded_unscaled[feat].unique()):
        row = {'feature': feat, 'value': int(val)}
        for grp, gval in [('Short', 1), ('Non-Short', 0)]:
            mask = coded_unscaled['Survival_Group'] == gval
            n_grp = mask.sum()
            n_val = ((coded_unscaled[feat] == val) & mask).sum()
            row[f'{grp}_n'] = int(n_val)
            row[f'{grp}_pct'] = round(n_val / n_grp * 100, 1) if n_grp > 0 else 0
        n_val_all = (coded_unscaled[feat] == val).sum()
        row['All_n'] = int(n_val_all)
        row['All_pct'] = round(n_val_all / n_total * 100, 1)
        table1_rows.append(row)

table1_df = pd.DataFrame(table1_rows)
table1_df.to_csv(os.path.join(OUT_DIR, 'step3_table1_distribution.csv'), index=False)
print(f"  Table 1 distribution → {os.path.join(OUT_DIR, 'step3_table1_distribution.csv')}")

elapsed = time.time() - start_time
print(f"\n{'=' * 70}")
print(f"Step 3 completed in {elapsed:.1f}s")
print(f"{'=' * 70}")
