"""
Step 2 — Data Preprocessing & Feature Engineering
===================================================
Extracts features from PRO-ACT tables following the feature definitions
reported by Papaiz et al. (2024). Combines all sources into a patient-level
intermediate dataset with raw (uncategorized) features.

Output: patient-level dataset with raw features + survival metadata.
"""

import pandas as pd
import numpy as np
import os
import time

# ── Configuration ──────────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', '01_data', 'raw')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '03_outputs', 'step2')
INTERIM_DIR = os.path.join(os.path.dirname(__file__), '..', '01_data', 'interim')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(INTERIM_DIR, exist_ok=True)

start_time = time.time()

print("=" * 70)
print("STEP 2 — DATA PREPROCESSING & FEATURE ENGINEERING")
print("Following Papaiz et al. (2024) methodology")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════
# HELPER: Select record closest to patient's Diagnosis_Delta
# ══════════════════════════════════════════════════════════════════════════
def get_closest_to_diagnosis(
        temporal_df, delta_col, diag_deltas, id_col='subject_id'):
    """For each patient, select the temporal record closest to their
    Diagnosis_Delta. Falls back to delta=0 if Diagnosis_Delta is missing.
    Records with null delta are treated as delta=0 (enrollment baseline)."""
    temp = temporal_df.copy()
    # Treat null delta as enrollment baseline (delta=0)
    temp[delta_col] = temp[delta_col].fillna(0)
    temp = temp.merge(diag_deltas[[id_col, 'Diagnosis_Delta']], on=id_col, how='left')
    temp['_ref'] = temp['Diagnosis_Delta'].fillna(0)
    temp['_abs_diff'] = (temp[delta_col] - temp['_ref']).abs()
    idx = temp.groupby(id_col)['_abs_diff'].idxmin()
    result = temp.loc[idx].drop(columns=['_abs_diff', '_ref', 'Diagnosis_Delta'])
    return result.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════
# 1. DEMOGRAPHICS → Age, Sex
# ══════════════════════════════════════════════════════════════════════════
print("\n[1/8] Loading DEMOGRAPHICS...")
demo_raw = pd.read_csv(os.path.join(RAW_DIR, 'PROACT_DEMOGRAPHICS.csv'))
demo = demo_raw[['subject_id', 'Age', 'Sex']].drop_duplicates('subject_id').copy()
demo['Sex'] = demo['Sex'].str.strip().str.capitalize()
# Remove patients without Age or Sex (as paper does)
demo = demo.dropna(subset=['Age', 'Sex'])
print(f"  Patients with Age and Sex: {len(demo):,}")
print(f"  Sex distribution: {demo['Sex'].value_counts().to_dict()}")


# ══════════════════════════════════════════════════════════════════════════
# 2. ALS HISTORY → Site_of_Onset, Onset_Delta, Diagnosis_Delta
#    Paper combines BOTH text AND binary Site_of_Onset columns
# ══════════════════════════════════════════════════════════════════════════
print("\n[2/8] Loading ALS HISTORY...")
hist_raw = pd.read_csv(os.path.join(RAW_DIR, 'PROACT_ALSHISTORY.csv'))

# ── 2a. Site of Onset: combine text (priority) + binary columns (fallback) ──
def extract_site(row):
    """Extract Site_of_Onset combining text column and binary columns."""
    text = row.get('Site_of_Onset')
    if pd.notna(text):
        text = str(text).strip()
        if 'Bulbar' in text and 'Limb' not in text:
            return 'Bulbar'
        if text in ('Onset: Limb', 'Limb'):
            return 'Limb/Spinal'
        if text in ('Onset: Spine', 'Spine', 'Spinal'):
            return 'Limb/Spinal'
        if 'Limb_and_Bulbar' in text or ('Limb' in text and 'Bulbar' in text):
            return 'Limb_and_Bulbar'
        if 'Other' in text:
            return 'Other'
        return 'Limb/Spinal'  # default for unknown text patterns
    # Binary columns fallback (only when text column is NaN)
    binary_map = {
        'Site_of_Onset___Bulbar': 'Bulbar',
        'Site_of_Onset___Limb': 'Limb/Spinal',
        'Site_of_Onset___Spine': 'Limb/Spinal',
        'Site_of_Onset___Limb_and_Bulbar': 'Limb_and_Bulbar',
        'Site_of_Onset___Other': 'Other',
    }
    active = []
    for col, val in binary_map.items():
        if col in row.index and pd.notna(row[col]) and row[col] == 1:
            active.append(val)
    if len(active) == 1:
        return active[0]
    return np.nan

hist_raw['_site'] = hist_raw.apply(extract_site, axis=1)

# Per patient: first non-null site
site_df = (hist_raw.dropna(subset=['_site'])
           .drop_duplicates('subject_id')[['subject_id', '_site']])
site_df.rename(columns={'_site': 'Site_of_Onset'}, inplace=True)

# ── 2b. Onset_Delta (= Symptoms_Onset_Delta in paper) ──
onset_df = hist_raw.dropna(subset=['Onset_Delta'])[['subject_id', 'Onset_Delta']]
onset_df = onset_df.groupby('subject_id')['Onset_Delta'].min().reset_index()

# ── 2c. Diagnosis_Delta ──
diag_delta_df = hist_raw.dropna(subset=['Diagnosis_Delta'])[['subject_id', 'Diagnosis_Delta']]
diag_delta_df = diag_delta_df.drop_duplicates('subject_id')

# Merge site + onset + diagnosis deltas
als_hist = site_df.merge(onset_df, on='subject_id', how='outer')
als_hist = als_hist.merge(diag_delta_df, on='subject_id', how='outer')

print(f"  Patients with Site_of_Onset: {als_hist['Site_of_Onset'].notna().sum():,}")
print(f"  Site distribution: {als_hist['Site_of_Onset'].value_counts().to_dict()}")
print(f"  Patients with Onset_Delta: {als_hist['Onset_Delta'].notna().sum():,}")
print(f"  Patients with Diagnosis_Delta: {als_hist['Diagnosis_Delta'].notna().sum():,}")


# ══════════════════════════════════════════════════════════════════════════
# 3. MERGE DEMOGRAPHICS + ALS HISTORY → Diagnosis_Delay, Age_at_Onset
# ══════════════════════════════════════════════════════════════════════════
print("\n[3/8] Merging demographics + ALS history...")
df = demo.merge(als_hist, on='subject_id', how='left')

# Diagnosis_Delay (months from onset to diagnosis)
df['Diagnosis_Delay'] = (df['Diagnosis_Delta'] - df['Onset_Delta']) / 30.44
# Filter invalid delays
df.loc[df['Diagnosis_Delay'] <= 0, 'Diagnosis_Delay'] = np.nan

# Age at Onset
df['Age_at_Onset'] = df['Age'] + df['Onset_Delta'] / 365.25

print(f"  Total patients: {len(df):,}")
print(f"  Diagnosis_Delay available: {df['Diagnosis_Delay'].notna().sum():,}")
print(f"  Age_at_Onset available: {df['Age_at_Onset'].notna().sum():,}")


# ══════════════════════════════════════════════════════════════════════════
# 4. ALSFRS → slopes, regions, gastrostomy closest to diagnosis
#    Paper: slope = (4 - score) / months_from_onset AT EACH VISIT,
#    then pick the visit closest to Diagnosis_Delta.
#    Regions: Bulbar(Q1+Q2+Q3<12), UpperLimb(Q4+Q5<8),
#             LowerLimb(Q8+Q9<8), Respiratory(Q10<4)
# ══════════════════════════════════════════════════════════════════════════
print("\n[4/8] Processing ALSFRS...")
alsfrs_raw = pd.read_csv(os.path.join(RAW_DIR, 'PROACT_ALSFRS.csv'))

# Unify Q5: Q5a (without gastrostomy), fallback Q5b (with gastrostomy)
alsfrs_raw['Q5_Cutting'] = alsfrs_raw['Q5a_Cutting_without_Gastrostomy'].fillna(
    alsfrs_raw['Q5b_Cutting_with_Gastrostomy'])

# Gastrostomy flag: 1 if Q5b is filled (patient has gastrostomy)
alsfrs_raw['Patient_with_Gastrostomy'] = (
    alsfrs_raw['Q5b_Cutting_with_Gastrostomy'].notna().astype(float))

# Unify Q10: ALSFRS Q10_Respiratory + ALSFRS-R R_1_Dyspnea
alsfrs_raw['Q10'] = alsfrs_raw['Q10_Respiratory'].fillna(alsfrs_raw['R_1_Dyspnea'])

# Add Onset_Delta for slope computation (from ALL patients, not just demographics)
all_onset = hist_raw.dropna(subset=['Onset_Delta'])[['subject_id', 'Onset_Delta']]
all_onset = all_onset.groupby('subject_id')['Onset_Delta'].min().reset_index()
alsfrs_proc = alsfrs_raw.merge(all_onset, on='subject_id', how='left')

# Compute months from onset at each visit
alsfrs_proc['_months_from_onset'] = (
    (alsfrs_proc['ALSFRS_Delta'] - alsfrs_proc['Onset_Delta']) / 30.44)

# Compute slopes at each visit: (4 - score) / months_from_onset
q_map = {
    'Q1_Speech': 'Q1_Speech',
    'Q2_Salivation': 'Q2_Salivation',
    'Q3_Swallowing': 'Q3_Swallowing',
    'Q4_Handwriting': 'Q4_Handwriting',
    'Q5_Cutting': 'Q5_Cutting',
    'Q6_Dressing_and_Hygiene': 'Q6_Dressing_and_Hygiene',
    'Q7_Turning_in_Bed': 'Q7_Turning_in_Bed',
    'Q8_Walking': 'Q8_Walking',
    'Q9_Climbing_Stairs': 'Q9_Climbing_Stairs',
    'Q10': 'Q10_Respiratory',
}
for src_col, slope_name in q_map.items():
    slope_col = f'Slope_{slope_name}'
    alsfrs_proc[slope_col] = np.where(
        alsfrs_proc['_months_from_onset'] > 0,
        (4 - alsfrs_proc[src_col]) / alsfrs_proc['_months_from_onset'],
        np.nan)
    alsfrs_proc.loc[alsfrs_proc[slope_col] < 0, slope_col] = 0.0

# Compute Region flags at each visit (paper's exact thresholds)
q123 = alsfrs_proc['Q1_Speech'] + alsfrs_proc['Q2_Salivation'] + alsfrs_proc['Q3_Swallowing']
alsfrs_proc['Region_Bulbar'] = np.where(q123.notna(), (q123 < 12).astype(float), np.nan)

q45 = alsfrs_proc['Q4_Handwriting'] + alsfrs_proc['Q5_Cutting']
alsfrs_proc['Region_Upper_Limb'] = np.where(q45.notna(), (q45 < 8).astype(float), np.nan)

q89 = alsfrs_proc['Q8_Walking'] + alsfrs_proc['Q9_Climbing_Stairs']
alsfrs_proc['Region_Lower_Limb'] = np.where(q89.notna(), (q89 < 8).astype(float), np.nan)

alsfrs_proc['Region_Respiratory'] = np.where(
    alsfrs_proc['Q10'].notna(), (alsfrs_proc['Q10'] < 4).astype(float), np.nan)

alsfrs_proc['Qty_Regions_Involved'] = (
    alsfrs_proc['Region_Bulbar'] + alsfrs_proc['Region_Upper_Limb'] +
    alsfrs_proc['Region_Lower_Limb'] + alsfrs_proc['Region_Respiratory'])

# Columns retained from the visit closest to diagnosis
alsfrs_feature_cols = (
    [f'Slope_{n}' for n in q_map.values()] +
    ['Patient_with_Gastrostomy', 'Qty_Regions_Involved',
     'Region_Bulbar', 'Region_Upper_Limb', 'Region_Lower_Limb',
     'Region_Respiratory'])
alsfrs_to_pick = alsfrs_proc[['subject_id', 'ALSFRS_Delta'] + alsfrs_feature_cols].copy()

# Pick values closest to patient's Diagnosis_Delta
all_diag = hist_raw.dropna(subset=['Diagnosis_Delta'])[['subject_id', 'Diagnosis_Delta']]
all_diag = all_diag.drop_duplicates('subject_id')
alsfrs_at_diag = get_closest_to_diagnosis(
    alsfrs_to_pick, 'ALSFRS_Delta', all_diag)
alsfrs_at_diag.drop(columns=['ALSFRS_Delta'], inplace=True)

# Rename with _at_Diagnosis suffix (matching paper column names)
rename = {c: f'{c}_at_Diagnosis' for c in alsfrs_feature_cols}
alsfrs_at_diag.rename(columns=rename, inplace=True)

print(f"  ALSFRS closest to diagnosis: {len(alsfrs_at_diag):,} patients")


# ══════════════════════════════════════════════════════════════════════════
# 5. FVC → FVC_pct from the record closest to diagnosis
#    Paper: max of 3 trial pct columns; fallback: max liters/subject_normal*100
# ══════════════════════════════════════════════════════════════════════════
print("\n[5/8] Processing FVC...")
fvc_raw = pd.read_csv(os.path.join(RAW_DIR, 'PROACT_FVC.csv'))

# FVC_Pct = max of 3 trial pct columns
fvc_raw['FVC_Pct'] = fvc_raw[
    ['pct_of_Normal_Trial_1', 'pct_of_Normal_Trial_2', 'pct_of_Normal_Trial_3']
].max(axis=1)

# Fallback: max liters / subject_normal * 100
mask_no_pct = fvc_raw['FVC_Pct'].isna()
fvc_raw['_max_liters'] = fvc_raw[
    ['Subject_Liters_Trial_1', 'Subject_Liters_Trial_2', 'Subject_Liters_Trial_3']
].max(axis=1)
can_compute = (mask_no_pct & fvc_raw['_max_liters'].notna() &
               fvc_raw['subject_normal'].notna() & (fvc_raw['subject_normal'] > 0))
fvc_raw.loc[can_compute, 'FVC_Pct'] = (
    fvc_raw.loc[can_compute, '_max_liters'] /
    fvc_raw.loc[can_compute, 'subject_normal'] * 100)

fvc_to_pick = fvc_raw[['subject_id', 'Forced_Vital_Capacity_Delta', 'FVC_Pct']].dropna(
    subset=['FVC_Pct']).copy()
fvc_at_diag = get_closest_to_diagnosis(
    fvc_to_pick, 'Forced_Vital_Capacity_Delta', all_diag)
fvc_at_diag.drop(columns=['Forced_Vital_Capacity_Delta'], inplace=True)
fvc_at_diag.rename(columns={'FVC_Pct': 'FVC_at_Diagnosis'}, inplace=True)

print(f"  FVC closest to diagnosis: {len(fvc_at_diag):,} patients")
print(f"  FVC% stats: mean={fvc_at_diag['FVC_at_Diagnosis'].mean():.1f}, "
      f"median={fvc_at_diag['FVC_at_Diagnosis'].median():.1f}")


# ══════════════════════════════════════════════════════════════════════════
# 6. VITAL SIGNS → BMI using weight closest to diagnosis
#    Paper: Height inches→cm→m (delete ≤0.80m); Weight pounds→kg (delete <25kg)
#    Height is static for adults → use ANY available record.
#    Weight is temporal → pick closest to diagnosis.
#    BMI = weight (closest to diagnosis) / height² (any visit)
# ══════════════════════════════════════════════════════════════════════════
print("\n[6/8] Processing VITAL SIGNS (BMI)...")
vs_raw = pd.read_csv(os.path.join(RAW_DIR, 'PROACT_VITALSIGNS.csv'))

# Height: convert Inches → cm
vs_raw['Height_cm'] = vs_raw['Height'].copy()
is_inches = ((vs_raw['Height_Units'] == 'Inches') |
             (vs_raw['Height_Units'].isna() & vs_raw['Height'].notna() &
              (vs_raw['Height'] < 100)))
vs_raw.loc[is_inches, 'Height_cm'] = vs_raw.loc[is_inches, 'Height'] * 2.54

# Convert cm → meters; delete ≤ 0.80m (paper threshold)
vs_raw['Height_m'] = vs_raw['Height_cm'] / 100
vs_raw.loc[vs_raw['Height_m'] <= 0.80, 'Height_m'] = np.nan

# Height is static: use median height per patient from ALL available records
patient_height = (vs_raw.dropna(subset=['Height_m'])
                  .groupby('subject_id')['Height_m']
                  .median().reset_index())
print(f"  Patients with height: {len(patient_height):,}")

# Weight: convert Pounds → kg; delete < 25kg (paper threshold)
vs_raw['Weight_kg'] = vs_raw['Weight'].copy()
is_pounds = ((vs_raw['Weight_Units'] == 'Pounds') |
             (vs_raw['Weight_Units'].isna() & vs_raw['Weight'].notna() &
              (vs_raw['Weight'] > 200)))
vs_raw.loc[is_pounds, 'Weight_kg'] = vs_raw.loc[is_pounds, 'Weight'] * 0.453592
vs_raw.loc[vs_raw['Weight_kg'] < 25, 'Weight_kg'] = np.nan

# Weight: pick the closest-to-diagnosis record
wt_to_pick = (vs_raw[['subject_id', 'Vital_Signs_Delta', 'Weight_kg']]
              .dropna(subset=['Weight_kg']).copy())
wt_at_diag = get_closest_to_diagnosis(
    wt_to_pick, 'Vital_Signs_Delta', all_diag)
wt_at_diag.drop(columns=['Vital_Signs_Delta'], inplace=True)
print(f"  Patients with weight closest to diagnosis: {len(wt_at_diag):,}")

# Combine height (any visit) + weight (closest to diagnosis) → BMI
bmi_at_diag = wt_at_diag.merge(patient_height, on='subject_id', how='inner')
bmi_at_diag['BMI_at_Diagnosis'] = (
    bmi_at_diag['Weight_kg'] / (bmi_at_diag['Height_m'] ** 2))
bmi_at_diag = bmi_at_diag[['subject_id', 'BMI_at_Diagnosis']].copy()

print(f"  BMI from closest weight record: {len(bmi_at_diag):,} patients")
print(f"  BMI stats: mean={bmi_at_diag['BMI_at_Diagnosis'].mean():.1f}, "
      f"median={bmi_at_diag['BMI_at_Diagnosis'].median():.1f}")


# ══════════════════════════════════════════════════════════════════════════
# 7. RILUZOLE
#    Paper: NaN → No
# ══════════════════════════════════════════════════════════════════════════
print("\n[7/8] Processing RILUZOLE...")
ril_raw = pd.read_csv(os.path.join(RAW_DIR, 'PROACT_RILUZOLE.csv'))
# Per patient: Yes if any record says Yes, else No
ril = ril_raw.groupby('subject_id')['Subject_used_Riluzole'].apply(
    lambda x: 'Yes' if 'Yes' in x.values else 'No'
).reset_index()
ril.rename(columns={'Subject_used_Riluzole': 'Riluzole'}, inplace=True)
print(f"  Patients with riluzole info: {len(ril):,}")
print(f"  Distribution: {ril['Riluzole'].value_counts().to_dict()}")


# ══════════════════════════════════════════════════════════════════════════
# 8. LAST VISIT + DEATH DATA
#    Paper: Last Visit = max delta across ALL PRO-ACT tables
# ══════════════════════════════════════════════════════════════════════════
print("\n[8/8] Computing Last Visit & Death Data...")

# ── 8a. Last Visit: max delta from ALL available tables ──
delta_sources = [
    ('PROACT_ALSFRS.csv', 'ALSFRS_Delta'),
    ('PROACT_FVC.csv', 'Forced_Vital_Capacity_Delta'),
    ('PROACT_VITALSIGNS.csv', 'Vital_Signs_Delta'),
    ('PROACT_ALSHISTORY.csv', 'Subject_ALS_History_Delta'),
    ('PROACT_DEMOGRAPHICS.csv', 'Demographics_Delta'),
    ('PROACT_RILUZOLE.csv', 'Riluzole_use_Delta'),
    ('PROACT_DEATHDATA.csv', 'Death_Days'),
    ('PROACT_LABS.csv', 'Laboratory_Delta'),
    ('PROACT_SVC.csv', 'Slow_vital_Capacity_Delta'),
    ('PROACT_ELESCORIAL.csv', 'delta_days'),
]
max_deltas = []
for fname, dcol in delta_sources:
    fpath = os.path.join(RAW_DIR, fname)
    if not os.path.exists(fpath):
        continue
    try:
        tdf = pd.read_csv(fpath, usecols=['subject_id', dcol])
        md = tdf.groupby('subject_id')[dcol].max().reset_index()
        md.rename(columns={dcol: 'max_delta'}, inplace=True)
        max_deltas.append(md)
    except (ValueError, KeyError):
        pass

all_max = pd.concat(max_deltas)
last_visit = all_max.groupby('subject_id')['max_delta'].max().reset_index()
last_visit.rename(columns={'max_delta': 'Last_Visit_Delta'}, inplace=True)
print(f"  Last Visit computed for {len(last_visit):,} patients")

# ── 8b. Death Data ──
death_raw = pd.read_csv(os.path.join(RAW_DIR, 'PROACT_DEATHDATA.csv'))
death = death_raw[['subject_id', 'Subject_Died', 'Death_Days']].drop_duplicates(
    'subject_id').copy()
death.rename(columns={'Subject_Died': 'Event_Dead'}, inplace=True)
death['Event_Dead'] = death['Event_Dead'].map({'Yes': True, 'No': False})
print(f"  Patients with death records: {len(death):,}")
print(f"  Died=True: {death['Event_Dead'].sum():,}")


# ══════════════════════════════════════════════════════════════════════════
# MERGE ALL INTO PATIENT-LEVEL DATASET
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MERGING all sources into patient-level dataset...")

# df already has demographics + ALS history
df = df.merge(alsfrs_at_diag, on='subject_id', how='left')
df = df.merge(fvc_at_diag, on='subject_id', how='left')
df = df.merge(bmi_at_diag, on='subject_id', how='left')
df = df.merge(ril, on='subject_id', how='left')
df = df.merge(last_visit, on='subject_id', how='left')
df = df.merge(death, on='subject_id', how='left')

# Riluzole: NaN → No (patients not in riluzole table)
df['Riluzole'] = df['Riluzole'].fillna('No')
# Event_Dead: NaN → False
df['Event_Dead'] = df['Event_Dead'].fillna(False)

print(f"  Total patients after merge: {len(df):,}")


# ══════════════════════════════════════════════════════════════════════════
# COMPUTE SURVIVAL-RELATED FEATURES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("COMPUTING SURVIVAL-RELATED FEATURES...")

# Event_Dead_Time_from_Onset (months)
# Death_Days is days from enrollment; Onset_Delta is negative
df['Event_Dead_Time_from_Onset'] = np.where(
    df['Event_Dead'] & df['Death_Days'].notna() & df['Onset_Delta'].notna(),
    (df['Death_Days'] - df['Onset_Delta']) / 30.44,
    np.nan)

# Last_Visit_from_Onset (months)
df['Last_Visit_from_Onset'] = np.where(
    df['Last_Visit_Delta'].notna() & df['Onset_Delta'].notna(),
    (df['Last_Visit_Delta'] - df['Onset_Delta']) / 30.44,
    np.nan)

print(f"  Event_Dead_Time_from_Onset available: "
      f"{df['Event_Dead_Time_from_Onset'].notna().sum():,}")
print(f"  Last_Visit_from_Onset available: "
      f"{df['Last_Visit_from_Onset'].notna().sum():,}")


# ══════════════════════════════════════════════════════════════════════════
# FEATURE AVAILABILITY SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FEATURE AVAILABILITY SUMMARY")
print(f"  Total patients in dataset: {len(df):,}")

# The 23 features from the paper (before codification)
# Column names as they exist in our DataFrame
feature_cols = [
    'Diagnosis_Delay', 'Age_at_Onset', 'Sex', 'Site_of_Onset',
    'Riluzole', 'FVC_at_Diagnosis', 'BMI_at_Diagnosis',
    'Patient_with_Gastrostomy_at_Diagnosis',
    'Qty_Regions_Involved_at_Diagnosis',
    'Region_Bulbar_at_Diagnosis', 'Region_Upper_Limb_at_Diagnosis',
    'Region_Lower_Limb_at_Diagnosis', 'Region_Respiratory_at_Diagnosis',
    'Slope_Q1_Speech_at_Diagnosis', 'Slope_Q2_Salivation_at_Diagnosis',
    'Slope_Q3_Swallowing_at_Diagnosis', 'Slope_Q4_Handwriting_at_Diagnosis',
    'Slope_Q5_Cutting_at_Diagnosis',
    'Slope_Q6_Dressing_and_Hygiene_at_Diagnosis',
    'Slope_Q7_Turning_in_Bed_at_Diagnosis',
    'Slope_Q8_Walking_at_Diagnosis', 'Slope_Q9_Climbing_Stairs_at_Diagnosis',
    'Slope_Q10_Respiratory_at_Diagnosis',
]
survival_cols = [
    'Event_Dead', 'Event_Dead_Time_from_Onset', 'Last_Visit_from_Onset',
]

print(f"\n  {'Feature':<35s} {'Available':>10s} {'%':>8s}")
print("  " + "-" * 55)
for c in feature_cols + survival_cols:
    if c in df.columns:
        avail = df[c].notna().sum()
        pct = avail / len(df) * 100
        print(f"  {c:<35s} {avail:>10,d} {pct:>7.1f}%")
    else:
        print(f"  {c:<35s}  ** MISSING COLUMN **")


# ══════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SAVING OUTPUTS...")

# Save patient-level features (interim — step3 will clean/codify/scale)
out_path = os.path.join(INTERIM_DIR, 'step2_patient_features.csv')
df.to_csv(out_path, index=False)
print(f"  Saved {len(df):,} patients × {len(df.columns)} columns → {out_path}")

# Save summary stats
summary_stats = {
    'total_patients': len(df),
    'n_feature_cols': len(feature_cols),
    'n_columns_total': len(df.columns),
}
for c in feature_cols:
    if c in df.columns:
        summary_stats[f'avail_{c}'] = int(df[c].notna().sum())
pd.Series(summary_stats).to_csv(os.path.join(OUT_DIR, 'step2_summary.csv'))

elapsed = time.time() - start_time
print(f"\n{'=' * 70}")
print(f"Step 2 completed in {elapsed:.1f}s")
print(f"Interim data saved to: {os.path.abspath(INTERIM_DIR)}")
print(f"Summary saved to:      {os.path.abspath(OUT_DIR)}")
print(f"{'=' * 70}")
