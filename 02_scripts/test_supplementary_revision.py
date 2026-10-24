"""Synthetic reviewer-regression fixtures; no study models are retrained.

Exported from the dissertation's test_reviewer_revisions.py.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
from reviewer_temporal_cohorts import bmi_codes, extract_site, selected_before


class TemporalEncodingTests(unittest.TestCase):
    def test_bmi_boundaries(self):
        values = pd.Series([18.49, 18.5, 24.999, 25, 29.999, 30])
        self.assertEqual(bmi_codes(values).tolist(), [0, 1, 1, 2, 2, 3])

    def test_unknown_onset_is_not_limb(self):
        self.assertTrue(pd.isna(extract_site(pd.Series({"Site_of_Onset": "unrecognised"}))))
        self.assertEqual(extract_site(pd.Series({"Site_of_Onset": "Onset: Limb"})), "Limb/Spinal")
        self.assertEqual(extract_site(pd.Series({"Site_of_Onset": "Onset: Bulbar"})), "Bulbar")
        self.assertTrue(pd.isna(extract_site(pd.Series({"Site_of_Onset": pd.NA,
                                                       "Site_of_Onset___Bulbar": pd.NA}))))

    def test_future_and_unknown_dates_excluded(self):
        records = pd.DataFrame({"subject_id": [1, 1, 1, 2],
                                "date": [-2, 6, np.nan, 5], "value": [10, 99, 88, 20]})
        cohort = pd.DataFrame({"subject_id": [1, 2], "Diagnosis_Delta": [-10, 0],
                               "landmark_delta": [5, 5]})
        selected = selected_before(records, "date", cohort, ["value"]).set_index("subject_id")
        self.assertEqual(selected.value.to_dict(), {1: 10, 2: 20})

class FailureAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("failure_test_modeling", ROOT / "02_scripts/step5_modeling.py")
        cls.module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(cls.module)
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(str(error))

    def test_resampling_failure_is_logged_and_raised(self):
        class BrokenSampler:
            def fit_resample(self, x, y):
                raise RuntimeError("synthetic resampling failure")
        module = self.module
        with tempfile.TemporaryDirectory(prefix="revision-tests-", dir=ROOT / "01_data/interim") as directory:
            self.assertEqual(Path(directory).resolve().parent, (ROOT / "01_data/interim").resolve())
            with patch.object(module, "OUT_DIR", directory), patch.object(module, "N_SPLITS", 2), \
                 patch.object(module, "N_REPEATS", 1), patch.object(module, "make_sampler", return_value=BrokenSampler()):
                with self.assertRaisesRegex(RuntimeError, "synthetic resampling failure"):
                    module.run_cv("Synthetic", {}, "Synthetic", np.arange(20).reshape(10, 2),
                                  np.array([0, 1] * 5), trial_number=7)
            records = [json.loads(line) for line in (Path(directory) / "step5_failures.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["phase"], "resampling")
            self.assertEqual(records[0]["trial"], 7)
            self.assertFalse((Path(directory) / "step5_fold_metrics.csv").exists())

    def test_success_ledger_has_context_and_metrics(self):
        with tempfile.TemporaryDirectory(prefix="revision-tests-", dir=ROOT / "01_data/interim") as directory, \
             patch.object(self.module, "OUT_DIR", directory):
            self.assertEqual(Path(directory).resolve().parent, (ROOT / "01_data/interim").resolve())
            self.module.record_fold("Synthetic", "None", 3, 2, "test", {"auc": .7, "f1": .4})
            frame = pd.read_csv(Path(directory) / "step5_fold_metrics.csv")
            self.assertEqual(frame.loc[0, "trial"], 3)
            self.assertEqual(frame.loc[0, "fold"], 2)
            self.assertEqual(frame.loc[0, "auc"], .7)

    def test_fit_failure_is_logged_and_raised(self):
        class BrokenModel:
            def fit(self, x, y):
                raise RuntimeError("synthetic fit failure")
        module = self.module
        with tempfile.TemporaryDirectory(prefix="revision-tests-", dir=ROOT / "01_data/interim") as directory:
            self.assertEqual(Path(directory).resolve().parent, (ROOT / "01_data/interim").resolve())
            with patch.object(module, "OUT_DIR", directory), patch.object(module, "N_SPLITS", 2), \
                 patch.object(module, "N_REPEATS", 1), patch.object(module, "make_sampler", return_value=None), \
                 patch.object(module, "make_classifier", return_value=BrokenModel()):
                with self.assertRaisesRegex(RuntimeError, "synthetic fit failure"):
                    module.run_cv("Synthetic", {}, "None", np.arange(20).reshape(10,2), np.array([0,1]*5))
            record = json.loads((Path(directory) / "step5_failures.jsonl").read_text())
            self.assertEqual(record["phase"], "fit_predict_score")

    def test_valid_metrics_unchanged(self):
        result = self.module.compute_fold_metrics(np.array([0,0,1,1]),
                                                  np.array([0,0,0,1]), np.array([.1,.4,.3,.9]))
        self.assertEqual(result['auc'], .75)
        self.assertEqual(result['recall'], .5)
        self.assertEqual(result['specificity'], 1.)

    def test_undefined_auc_has_no_half_score_fallback(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertRaises(ValueError):
                self.module.compute_fold_metrics(np.ones(3), np.ones(3), np.array([.1,.5,.9]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
