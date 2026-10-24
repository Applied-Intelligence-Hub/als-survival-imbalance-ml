"""Build explicitly timed supplementary Study II cohorts without replacing Step 3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SLOPES = {
    "Q1_Speech": "Q1_Speech_slope", "Q2_Salivation": "Q2_Salivation_slope",
    "Q3_Swallowing": "Q3_Swallowing_slope", "Q4_Handwriting": "Q4_Handwriting_slope",
    "Q5_Cutting": "Q5_Cutting_slope", "Q6_Dressing_and_Hygiene": "Q6_Dressing_slope",
    "Q7_Turning_in_Bed": "Q7_Turning_slope", "Q8_Walking": "Q8_Walking_slope",
    "Q9_Climbing_Stairs": "Q9_Climbing_slope", "Q10": "Q10_Respiratory_slope",
}


def extract_site(row):
    text = row.get("Site_of_Onset")
    if pd.notna(text):
        text = str(text).strip()
        if "Bulbar" in text and "Limb" not in text:
            return "Bulbar"
        if text in ("Onset: Limb", "Limb", "Onset: Spine", "Spine", "Spinal"):
            return "Limb/Spinal"
        if "Limb_and_Bulbar" in text or ("Limb" in text and "Bulbar" in text):
            return "Limb_and_Bulbar"
        if "Other" in text:
            return "Other"
        return np.nan
    fields = {"Bulbar": "Bulbar", "Limb": "Limb/Spinal", "Spine": "Limb/Spinal",
              "Limb_and_Bulbar": "Limb_and_Bulbar", "Other": "Other"}
    active = []
    for key, value in fields.items():
        flag = row.get(f"Site_of_Onset___{key}")
        if pd.notna(flag) and flag == 1:
            active.append(value)
    return active[0] if len(active) == 1 else np.nan


def bmi_codes(values):
    return pd.cut(values, [-np.inf, 18.5, 25, 30, np.inf],
                  labels=[0, 1, 2, 3], right=False).astype(int)


def selected_before(records, date, cohort, columns):
    """Closest-to-diagnosis record among known dates at/before the landmark."""
    values = records[["subject_id", date] + columns].copy()
    values = values.merge(cohort[["subject_id", "Diagnosis_Delta", "landmark_delta"]],
                          on="subject_id", validate="many_to_one")
    values = values.loc[values[date].notna() & values[date].le(values.landmark_delta)]
    values["_distance"] = (values[date] - values.Diagnosis_Delta).abs()
    if values.empty:
        return pd.DataFrame(columns=["subject_id"] + columns)
    chosen = values.loc[values.groupby("subject_id")._distance.idxmin()]
    assert chosen[date].le(chosen.landmark_delta).all()
    return chosen[["subject_id"] + columns]


def landmark_features(base, raw_dir, month):
    cohort = base.copy()
    cohort["landmark_delta"] = cohort.Onset_Delta + month * 30.44
    eligible = ((-cohort.Onset_Delta / 30.44).between(0, month)
                & (~cohort.Event_Dead.astype(bool)
                   | cohort.Event_Dead_Time_from_Onset.gt(month))
                & cohort.Diagnosis_Delta.notna()
                & cohort.Diagnosis_Delta.le(cohort.landmark_delta))
    cohort = cohort.loc[eligible].copy()
    initial_n = len(cohort)
    history = pd.read_csv(raw_dir / "PROACT_ALSHISTORY.csv")
    history["_site"] = history.apply(extract_site, axis=1)
    site = history.dropna(subset=["_site"]).drop_duplicates("subject_id").set_index("subject_id")._site
    cohort["Site_of_Onset"] = cohort.subject_id.map(site)

    als = pd.read_csv(raw_dir / "PROACT_ALSFRS.csv")
    als["Q5_Cutting"] = als.Q5a_Cutting_without_Gastrostomy.fillna(als.Q5b_Cutting_with_Gastrostomy)
    als["Q10"] = als.Q10_Respiratory.fillna(als.R_1_Dyspnea)
    als["Patient_with_Gastrostomy"] = als.Q5b_Cutting_with_Gastrostomy.notna().astype(int)
    als = als.merge(cohort[["subject_id", "Onset_Delta"]], on="subject_id", validate="many_to_one")
    elapsed = (als.ALSFRS_Delta - als.Onset_Delta) / 30.44
    for score, name in SLOPES.items():
        als[name] = ((4 - als[score]) / elapsed).where(elapsed.gt(0)).clip(lower=0)
    regions = {"Region_Involved_Bulbar": (["Q1_Speech", "Q2_Salivation", "Q3_Swallowing"], 12),
               "Region_Involved_Upper_Limb": (["Q4_Handwriting", "Q5_Cutting"], 8),
               "Region_Involved_Lower_Limb": (["Q8_Walking", "Q9_Climbing_Stairs"], 8),
               "Region_Involved_Respiratory": (["Q10"], 4)}
    for name, (scores, threshold) in regions.items():
        total = als[scores].sum(axis=1, min_count=len(scores))
        als[name] = total.lt(threshold).astype(float).where(total.notna())
    als["Qty_Regions_Involved"] = als[list(regions)].sum(axis=1, min_count=4)
    dynamic = list(SLOPES.values()) + ["Patient_with_Gastrostomy", "Qty_Regions_Involved"] + list(regions)
    selected_als = selected_before(als, "ALSFRS_Delta", cohort, dynamic)

    fvc = pd.read_csv(raw_dir / "PROACT_FVC.csv")
    fvc["FVC"] = fvc[[f"pct_of_Normal_Trial_{i}" for i in [1, 2, 3]]].max(axis=1)
    litres = fvc[[f"Subject_Liters_Trial_{i}" for i in [1, 2, 3]]].max(axis=1)
    fallback = fvc.FVC.isna() & litres.notna() & fvc.subject_normal.gt(0)
    fvc.loc[fallback, "FVC"] = litres[fallback] / fvc.loc[fallback, "subject_normal"] * 100
    selected_fvc = selected_before(fvc.dropna(subset=["FVC"]),
                                   "Forced_Vital_Capacity_Delta", cohort, ["FVC"])

    vitals = pd.read_csv(raw_dir / "PROACT_VITALSIGNS.csv")
    vitals["height"] = vitals.Height
    inches = vitals.Height_Units.eq("Inches") | (vitals.Height_Units.isna() & vitals.Height.lt(100))
    vitals.loc[inches, "height"] *= 2.54
    vitals["height"] = (vitals.height / 100).where(vitals.height.gt(80))
    vitals["weight"] = vitals.Weight
    pounds = vitals.Weight_Units.eq("Pounds") | (vitals.Weight_Units.isna() & vitals.Weight.gt(200))
    vitals.loc[pounds, "weight"] *= .453592
    vitals.loc[vitals.weight.lt(25), "weight"] = np.nan
    selected_weight = selected_before(vitals.dropna(subset=["weight"]),
                                      "Vital_Signs_Delta", cohort, ["weight"])
    heights = vitals[["subject_id", "Vital_Signs_Delta", "height"]].merge(
        cohort[["subject_id", "landmark_delta"]], on="subject_id", validate="many_to_one")
    heights = heights.loc[heights.Vital_Signs_Delta.notna()
                          & heights.Vital_Signs_Delta.le(heights.landmark_delta)]
    height = heights.groupby("subject_id").height.median()

    riluzole = pd.read_csv(raw_dir / "PROACT_RILUZOLE.csv")
    riluzole = riluzole.merge(cohort[["subject_id", "landmark_delta"]], on="subject_id", validate="many_to_one")
    riluzole = riluzole.loc[riluzole.Riluzole_use_Delta.notna()
                            & riluzole.Riluzole_use_Delta.le(riluzole.landmark_delta)]
    known_use = riluzole.Subject_used_Riluzole.eq("Yes").groupby(riluzole.subject_id).any()
    result = cohort[["subject_id", "Sex", "Site_of_Onset", "Age_at_Onset", "Diagnosis_Delay", "Survival_Group"]].copy()
    result = result.merge(selected_als, on="subject_id", how="left", validate="one_to_one")
    result = result.merge(selected_fvc, on="subject_id", how="left", validate="one_to_one")
    result = result.merge(selected_weight, on="subject_id", how="left", validate="one_to_one")
    result["BMI"] = result.weight / result.subject_id.map(height) ** 2
    result["Riluzole"] = result.subject_id.map(known_use).fillna(False).astype(int)
    result = result.dropna()
    result = result.loc[result.Site_of_Onset.isin(["Bulbar", "Limb/Spinal"])].copy()
    coded = pd.DataFrame({"subject_id": result.subject_id, "Survival_Group": result.Survival_Group})
    coded["Sex_Male"] = result.Sex.eq("Male").astype(int)
    coded["Site_Onset"] = result.Site_of_Onset.eq("Limb/Spinal").astype(int)
    coded["Age_at_Onset"] = pd.cut(result.Age_at_Onset, [-np.inf, 39, 49, 59, 69, np.inf], labels=False)
    coded["Diagnosis_Delay"] = pd.cut(result.Diagnosis_Delay, [-np.inf, 8, 18, np.inf], labels=False)
    coded["FVC_at_Diagnosis"] = result.FVC.lt(80).astype(int)
    coded["BMI_at_Diagnosis"] = bmi_codes(result.BMI)
    for column in ["Riluzole", "Patient_with_Gastrostomy", "Qty_Regions_Involved"] + list(regions):
        coded[column] = result[column].astype(int)
    for column in SLOPES.values():
        coded[column] = pd.cut(result[column], [-np.inf, .05, .14, np.inf], right=False, labels=False)
    assert coded.notna().all().all()
    metadata = {"eligible_alive_and_observed": initial_n, "complete_timed_cases": len(coded),
                "dropped_for_missing_or_invalid_predictors": initial_n - len(coded),
                "no_known_riluzole_record": int((~cohort.subject_id.isin(known_use.index)).sum()),
                "prediction_month": month, "last_outcome_month": 24,
                "predictor_dates_bounded": True,
                "riluzole_definition": "Any recorded Yes at/before landmark; absence of a known record coded as no recorded use."}
    return coded, metadata


def build_cohorts(study_root):
    original = pd.read_csv(study_root / "01_data/processed/step3_coded_unscaled.csv")
    raw = pd.read_csv(study_root / "01_data/interim/step2_patient_features.csv")
    base = raw.merge(original[["subject_id", "Survival_Group"]], on="subject_id", validate="one_to_one")
    columns = original.columns.tolist()
    corrected = original.copy()
    bmi = base.set_index("subject_id").loc[original.subject_id, "BMI_at_Diagnosis"]
    corrected["BMI_at_Diagnosis"] = bmi_codes(bmi).to_numpy()
    cohorts = {"bmi_boundary_check": corrected}
    metadata = {"bmi_boundary_check": {"changed_bmi_codes": int((corrected.BMI_at_Diagnosis != original.BMI_at_Diagnosis).sum()),
                                        "same_patients_and_partitions": True}}
    for month in [18, 12]:
        ids = base.loc[(-base.Onset_Delta / 30.44).between(0, month), "subject_id"]
        cohorts[f"entry_{month}m"] = original.loc[original.subject_id.isin(ids)].copy()
        metadata[f"entry_{month}m"] = {"entry_proxy_max_month": month,
                                      "predictors_restricted_in_time": False,
                                      "interpretation": "Entry-only sensitivity; retrospective features retained."}
        timed, audit = landmark_features(base, study_root / "01_data/raw", month)
        cohorts[f"landmark_{month}m"] = timed[columns].copy()
        metadata[f"landmark_{month}m"] = audit
    return cohorts, metadata
