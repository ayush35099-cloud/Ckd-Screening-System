import unittest
import os
import sys
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.model import model_service, CKDModelService

class TestCKDModelService(unittest.TestCase):
    def setUp(self):
        self.service = model_service
        self.healthy_sample = {
            'age': 32, 'bp': 70, 'sg': '1.025', 'al': '0', 'su': '0',
            'bgr': 90, 'bu': 22, 'sc': '0.7', 'sod': 142, 'pot': 4.1,
            'hemo': 15.6, 'pcv': 46, 'wc': 6500, 'rc': 5.4,
            'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
            'htn': 'no', 'dm': 'no', 'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no'
        }
        self.ckd_sample = {
            'age': 67, 'bp': 95, 'sg': '1.010', 'al': '3', 'su': '2',
            'bgr': 240, 'bu': 90, 'sc': '3.8', 'sod': 128, 'pot': 5.6,
            'hemo': 8.8, 'pcv': 26, 'wc': 12000, 'rc': 2.9,
            'rbc': 'abnormal', 'pc': 'abnormal', 'pcc': 'present', 'ba': 'present',
            'htn': 'yes', 'dm': 'yes', 'cad': 'yes', 'appet': 'poor', 'pe': 'yes', 'ane': 'yes'
        }

    def test_artifacts_loaded(self):
        self.assertIsNotNone(self.service.rf_pipeline)
        self.assertIsNotNone(self.service.dt_pipeline)
        self.assertIsNotNone(self.service.metadata)
        self.assertIsNotNone(self.service.metrics)
        self.assertEqual(len(self.service.metadata['features']), 24)

    def test_ckd_model_pkl_compatibility(self):
        # Verify ckd_model.pkl exists in ml/models and root
        pkl_path = os.path.join(self.service.models_dir, 'ckd_model.pkl')
        self.assertTrue(os.path.exists(pkl_path))
        import joblib
        loaded_pkl = joblib.load(pkl_path)
        self.assertIsNotNone(loaded_pkl)
        self.assertTrue(hasattr(loaded_pkl, 'predict'))
        self.assertTrue(hasattr(loaded_pkl, 'predict_proba'))

    def test_healthy_prediction_rf(self):
        result = self.service.predict(self.healthy_sample, model_choice='Random Forest')
        self.assertFalse(result['is_flagged'])
        self.assertEqual(result['prediction_code'], 'notckd')
        self.assertIn("Not Detected", result['result_label'])
        self.assertLess(result['probability_ckd'], 50.0)

    def test_ckd_prediction_rf(self):
        result = self.service.predict(self.ckd_sample, model_choice='Random Forest')
        self.assertTrue(result['is_flagged'])
        self.assertEqual(result['prediction_code'], 'ckd')
        self.assertIn("Detected", result['result_label'])
        self.assertGreater(result['probability_ckd'], 50.0)
        self.assertGreater(len(result['key_findings']), 0)

    def test_healthy_prediction_dt(self):
        result = self.service.predict(self.healthy_sample, model_choice='Decision Tree')
        self.assertFalse(result['is_flagged'])
        self.assertEqual(result['prediction_code'], 'notckd')

    def test_ckd_prediction_dt(self):
        result = self.service.predict(self.ckd_sample, model_choice='Decision Tree')
        self.assertTrue(result['is_flagged'])
        self.assertEqual(result['prediction_code'], 'ckd')

    def test_missing_values_handled(self):
        # Partial sample with several missing parameters
        partial_sample = {
            'age': 55, 'bp': 80,
            'hemo': 10.1, 'sc': 2.4,
            'htn': 'yes'
        }
        result = self.service.predict(partial_sample, model_choice='Random Forest')
        self.assertIn('is_flagged', result)
        self.assertIn('probability_ckd', result)

    def test_invalid_numerical_inputs(self):
        # Pass non-numeric string values to numeric columns
        invalid_sample = {
            'age': 'not_a_number',
            'sc': 'unknown_text',
            'hemo': '---',
            'htn': 'no'
        }
        # Pipeline must impute invalid values rather than crashing
        result = self.service.predict(invalid_sample, model_choice='Random Forest')
        self.assertIn('is_flagged', result)
        self.assertIn('probability_ckd', result)

    def test_unknown_categorical_values(self):
        # Pass unexpected category values
        weird_sample = {
            'htn': 'maybe',
            'dm': 'borderline',
            'rbc': 'uncertain',
            'appet': 'neutral'
        }
        result = self.service.predict(weird_sample, model_choice='Random Forest')
        self.assertIn('is_flagged', result)
        self.assertIn('probability_ckd', result)

    def test_model_loading_missing_file_raises_error(self):
        # Testing constructor behavior when artifact paths do not exist
        instance = CKDModelService.__new__(CKDModelService)
        instance.models_dir = '/non_existent_directory_path'
        instance.rf_pipeline_path = '/non_existent_directory_path/ckd_pipeline.joblib'
        instance.rf_pkl_path = '/non_existent_directory_path/ckd_model.pkl'
        instance.root_pkl_path = '/non_existent_directory_path/root_ckd.pkl'
        instance.dt_pipeline_path = '/non_existent_directory_path/dt.joblib'
        instance.metrics_path = '/non_existent_directory_path/metrics.json'
        instance.meta_path = '/non_existent_directory_path/meta.json'
        with self.assertRaises(FileNotFoundError):
            instance._load_artifacts()

if __name__ == '__main__':
    unittest.main()
