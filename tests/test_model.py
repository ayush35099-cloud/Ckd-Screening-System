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
        self.assertIn("not detected", result['result_label'].lower())
        self.assertLess(result['probability_ckd'], 50.0)
        self.assertGreater(result['confidence'], 90.0)
        self.assertIn('missing_parameters', result)
        self.assertIn('probability_explanation', result)
        self.assertIn('comparison', result)
        self.assertTrue(result['comparison']['agreement'])

    def test_ckd_prediction_rf(self):
        result = self.service.predict(self.ckd_sample, model_choice='Random Forest')
        self.assertTrue(result['is_flagged'])
        self.assertEqual(result['prediction_code'], 'ckd')
        self.assertIn("detected", result['result_label'].lower())
        self.assertGreater(result['probability_ckd'], 50.0)
        self.assertGreater(result['confidence'], 90.0)
        self.assertGreater(len(result['key_findings']), 0)
        self.assertIn('missing_parameters', result)
        self.assertIn('probability_explanation', result)
        self.assertIn('comparison', result)
        self.assertTrue(result['comparison']['agreement'])

    def test_healthy_prediction_dt(self):
        result = self.service.predict(self.healthy_sample, model_choice='Decision Tree')
        self.assertFalse(result['is_flagged'])
        self.assertEqual(result['prediction_code'], 'notckd')
        self.assertEqual(result['confidence'], 100.0)
        self.assertEqual(result['probability_ckd'], 0.0)

    def test_ckd_prediction_dt(self):
        result = self.service.predict(self.ckd_sample, model_choice='Decision Tree')
        self.assertTrue(result['is_flagged'])
        self.assertEqual(result['prediction_code'], 'ckd')
        self.assertEqual(result['confidence'], 100.0)
        self.assertEqual(result['probability_ckd'], 100.0)

    def test_model_confidence_calculation_and_distinction(self):
        # Verify healthy prediction does NOT yield tiny confidence (e.g. 0.6%)
        res_healthy = self.service.predict(self.healthy_sample, model_choice='Random Forest')
        self.assertGreater(res_healthy['confidence'], 90.0)
        self.assertLess(res_healthy['probability_ckd'], 10.0)
        self.assertEqual(res_healthy['prediction_code'], 'notckd')

        # Verify CKD prediction yields high confidence and high probability_ckd
        res_ckd = self.service.predict(self.ckd_sample, model_choice='Random Forest')
        self.assertGreater(res_ckd['confidence'], 90.0)
        self.assertGreater(res_ckd['probability_ckd'], 90.0)
        self.assertEqual(res_ckd['prediction_code'], 'ckd')

        # Verify Decision Tree confidence
        res_dt_healthy = self.service.predict(self.healthy_sample, model_choice='Decision Tree')
        self.assertEqual(res_dt_healthy['confidence'], 100.0)
        self.assertEqual(res_dt_healthy['probability_ckd'], 0.0)

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
        self.assertIn('missing_parameters', result)
        self.assertGreater(len(result['missing_parameters']), 0)
        self.assertIn('bgr', result['missing_parameters'])
        self.assertIn('probability_explanation', result)

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

    def test_exact_24_features_schema(self):
        expected_features = [
            'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
            'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
        ]
        actual_features = list(self.service.metadata['features'].keys())
        self.assertEqual(actual_features, expected_features)
        self.assertEqual(len(actual_features), 24)

    def test_administrative_fields_ignored_by_model(self):
        # Administrative fields (patient_id, patient_name, sex) should not pollute the 24-feature ML matrix
        sample_with_admin = dict(self.healthy_sample)
        sample_with_admin['patient_id'] = 'PT-ADMIN-IGNORE'
        sample_with_admin['patient_name'] = 'Confidential Name'
        sample_with_admin['sex'] = 'Female'
        
        result = self.service.predict(sample_with_admin, model_choice='Random Forest')
        self.assertIn('evaluated_parameters', result)
        self.assertNotIn('patient_id', result['evaluated_parameters'])
        self.assertNotIn('patient_name', result['evaluated_parameters'])
        self.assertNotIn('sex', result['evaluated_parameters'])
        self.assertEqual(len(result['evaluated_parameters']), 24)

    def test_model_2_artifacts_and_schema(self):
        self.assertTrue(self.service.is_model_2_available())
        self.assertIsNotNone(self.service.nhanes_rf_pipeline)
        self.assertIsNotNone(self.service.nhanes_dt_pipeline)
        self.assertIsNotNone(self.service.nhanes_metadata)
        self.assertIsNotNone(self.service.nhanes_metrics)
        self.assertEqual(len(self.service.nhanes_metadata['features']), 19)

    def test_model_2_healthy_prediction(self):
        nhanes_healthy = {
            'age': 38, 'sex': 'female', 'bmi': 23.2, 'sbp': 116, 'dbp': 74,
            'bgr': 92, 'hba1c': 5.1, 'bu': 12.0, 'sc': 0.75, 'sod': 140, 'pot': 4.1,
            'tot_protein': 7.2, 'serum_alb': 4.4, 'uric_acid': 4.8, 'hemo': 14.0,
            'pcv': 42.0, 'wc': 6.2, 'htn': 'no', 'dm': 'no'
        }
        result = self.service.predict(nhanes_healthy, model_choice='Random Forest', model_id='model_2')
        self.assertEqual(result['model_id'], 'model_2')
        self.assertFalse(result['is_flagged'])
        self.assertEqual(result['prediction_code'], 'notckd')
        self.assertIn('CDC NHANES', result['model_used'])
        self.assertIn('CDC NHANES', result['dataset_source'])
        self.assertGreater(result['confidence'], 50.0)

    def test_model_2_flagged_prediction(self):
        nhanes_ckd = {
            'age': 71, 'sex': 'male', 'bmi': 32.5, 'sbp': 162, 'dbp': 94,
            'bgr': 185, 'hba1c': 8.5, 'bu': 42.0, 'sc': 2.8, 'sod': 133, 'pot': 5.2,
            'tot_protein': 6.1, 'serum_alb': 3.1, 'uric_acid': 8.9, 'hemo': 9.8,
            'pcv': 30.0, 'wc': 9.9, 'htn': 'yes', 'dm': 'yes'
        }
        result = self.service.predict(nhanes_ckd, model_choice='Random Forest', model_id='model_2')
        self.assertEqual(result['model_id'], 'model_2')
        self.assertTrue(result['is_flagged'])
        self.assertEqual(result['prediction_code'], 'ckd')
        self.assertGreater(result['probability_ckd'], 50.0)
        self.assertGreater(len(result['key_findings']), 0)

    def test_model_selection_validation_rejects_unknown(self):
        with self.assertRaises(ValueError):
            self.service.predict(self.healthy_sample, model_id='unknown_model_identifier_xyz')

    def test_model_2_graceful_fallback_when_unavailable(self):
        # Save current state
        orig_rf = self.service.nhanes_rf_pipeline
        orig_avail = self.service.model_2_available
        try:
            self.service.nhanes_rf_pipeline = None
            self.service.model_2_available = False

            # Model 1 must still work completely fine
            res1 = self.service.predict(self.healthy_sample, model_id='model_1')
            self.assertEqual(res1['model_id'], 'model_1')
            self.assertFalse(res1['is_flagged'])

            # Model 2 must raise clear RuntimeError
            with self.assertRaises(RuntimeError):
                self.service.predict(self.healthy_sample, model_id='model_2')
        finally:
            self.service.nhanes_rf_pipeline = orig_rf
            self.service.model_2_available = orig_avail

if __name__ == '__main__':
    unittest.main()
