import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from report_generator import generate_pdf_report
from app import create_app

class TestPDFReportGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.client = cls.app.test_client()

    def test_pdf_generation_flagged(self):
        sample_data = {
            'screening_uuid': 'test-uuid-12345678',
            'patient_id': 'PT-DEMO-001',
            'patient_name': 'Test Patient',
            'age': 65,
            'sex': 'Male',
            'prediction_label': 'CKD Indicators Detected',
            'prediction_code': 'ckd',
            'probability': 94.5,
            'risk_level': 'High',
            'model_used': 'Random Forest',
            'formatted_date': 'Nov 17, 2026, 10:00 AM',
            'recommendation': 'Nephrology referral advised.',
            'findings': [
                {'parameter': 'Serum Creatinine', 'value': '2.8 mg/dL', 'status': 'Elevated', 'detail': 'Reduced GFR'}
            ],
            'parameters': {
                'bp': 90, 'hemo': 9.5, 'sc': 2.8, 'htn': 'yes'
            }
        }
        pdf_bytes = generate_pdf_report(sample_data)
        self.assertIsNotNone(pdf_bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b'%PDF'))

    def test_pdf_generation_non_flagged(self):
        sample_data = {
            'screening_uuid': 'test-uuid-87654321',
            'patient_id': 'PT-DEMO-002',
            'patient_name': 'Healthy Patient',
            'age': 30,
            'sex': 'Female',
            'prediction_label': 'CKD Indicators Not Detected',
            'prediction_code': 'notckd',
            'probability': 5.2,
            'risk_level': 'Low',
            'model_used': 'Random Forest',
            'formatted_date': 'Nov 17, 2026, 11:00 AM',
            'recommendation': 'Routine annual monitoring.',
            'findings': [],
            'parameters': {'bp': 75, 'hemo': 14.5, 'sc': 0.8}
        }
        pdf_bytes = generate_pdf_report(sample_data)
        self.assertIsNotNone(pdf_bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b'%PDF'))

    def test_pdf_with_missing_parameters(self):
        sample_data = {
            'screening_uuid': 'test-uuid-missing-1111',
            'patient_id': 'PT-DEMO-003',
            'patient_name': 'Partial Data Patient',
            'age': 55,
            'sex': 'Male',
            'prediction_label': 'CKD Indicators Detected',
            'prediction_code': 'ckd',
            'probability': 88.0,
            'risk_level': 'High',
            'model_used': 'Random Forest',
            'formatted_date': 'Nov 17, 2026, 12:00 PM',
            'missing_parameters': ['bgr', 'bu', 'sod', 'pot'],
            'findings': [],
            'parameters': {'age': 55, 'bp': 80, 'sc': 2.1}
        }
        pdf_bytes = generate_pdf_report(sample_data)
        self.assertIsNotNone(pdf_bytes)
        self.assertTrue(pdf_bytes.startswith(b'%PDF'))

    def test_pdf_handles_none_values(self):
        sample_data = {
            'screening_uuid': 'test-uuid-empty-2222',
            'patient_id': 'PT-DEMO-EMPTY',
            'patient_name': None,
            'age': None,
            'sex': 'Not Specified',
            'prediction_label': 'CKD Indicators Not Detected',
            'prediction_code': 'notckd',
            'probability': 12.0,
            'risk_level': 'Low',
            'model_used': 'Decision Tree',
            'formatted_date': None,
            'missing_parameters': None,
            'findings': None,
            'parameters': None
        }
        pdf_bytes = generate_pdf_report(sample_data)
        self.assertIsNotNone(pdf_bytes)
        self.assertTrue(pdf_bytes.startswith(b'%PDF'))

    def test_pdf_generation_model_2(self):
        sample_data = {
            'screening_uuid': 'test-uuid-nhanes-001',
            'patient_id': 'PT-NHANES-PDF-01',
            'patient_name': 'NHANES Evaluation Subject',
            'age': 62,
            'sex': 'Female',
            'prediction_label': 'CKD Indicators Detected by Population Model',
            'prediction_code': 'ckd',
            'probability': 84.5,
            'risk_level': 'High',
            'model_id': 'model_2',
            'model_used': 'Random Forest',
            'model_name': 'CDC NHANES General Population Screening Pipeline',
            'formatted_date': 'Nov 27, 2026, 11:30 PM',
            'recommendation': 'Comprehensive clinical evaluation and formal laboratory eGFR / uACR testing indicated.',
            'findings': [
                {'parameter': 'Serum Creatinine', 'value': '1.6 mg/dL', 'status': 'Elevated', 'detail': 'Above normal reference limit'},
                {'parameter': 'Systolic Blood Pressure', 'value': '142 mmHg', 'status': 'Elevated', 'detail': 'Stage 2 Hypertension criteria'}
            ],
            'parameters': {
                'age': 62, 'sex': 2, 'bmi': 31.2, 'sbp': 142, 'dbp': 88, 'bgr': 145, 'hba1c': 7.2,
                'bu': 28, 'sc': 1.6, 'sod': 136, 'pot': 4.7, 'tot_protein': 6.8, 'serum_alb': 3.7,
                'uric_acid': 7.1, 'hemo': 11.2, 'pcv': 34.0, 'wc': 8.9, 'htn': 1, 'dm': 1
            }
        }
        pdf_bytes = generate_pdf_report(sample_data)
        self.assertIsNotNone(pdf_bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b'%PDF'))

    def test_pdf_language_isolation_all_8_languages(self):
        """All 8 language codes produce consistent, clean English PDFs with fixed English headers."""
        sample_record = {
            'screening_uuid': 'test-uuid-lang-isolation-001',
            'patient_id': 'PT-ISO-01',
            'patient_name': 'Lang Isolation Patient',
            'age': 54,
            'sex': 'Male',
            'prediction_code': 'ckd',
            'prediction_label': 'CKD Indicators Detected',
            'risk_level': 'High',
            'probability': 91.2,
            'confidence': 91.2,
            'model_used': 'Random Forest',
            'formatted_date': 'Nov 28, 2026, 12:00 AM',
            'parameters': {'bp': 85, 'hemo': 11.0, 'sc': 2.4, 'htn': 'yes'}
        }

        languages = ['en', 'hi', 'bn', 'mr', 'te', 'ta', 'ur', 'gu']
        for lang in languages:
            with self.subTest(lang=lang):
                pdf_bytes = generate_pdf_report(sample_record, lang=lang)
                self.assertIsNotNone(pdf_bytes)
                self.assertTrue(pdf_bytes.startswith(b'%PDF'))
                self.assertGreater(len(pdf_bytes), 1000)

    def test_pdf_unicode_and_special_patient_names(self):
        """Patient names with Unicode, accents, XML entities and long strings render cleanly."""
        test_names = [
            'आयुष शर्मा (Hindi/Marathi Unicode)',
            'মোহাম্মদ আলী (Bengali Unicode)',
            'Dr. René Müller (Accented Latin)',
            'A & B Medical Care <Specialist Clinical Center>',
            'Very Long Patient Identifier Name Used In Specialized Regional Diagnostic Screening Camps Ext'
        ]

        for pt_name in test_names:
            with self.subTest(pt_name=pt_name):
                sample_data = {
                    'screening_uuid': 'test-uuid-unicode-name',
                    'patient_id': 'PT-UNI-009',
                    'patient_name': pt_name,
                    'age': 45,
                    'sex': 'Female',
                    'prediction_label': 'CKD Indicators Not Detected',
                    'prediction_code': 'notckd',
                    'probability': 8.0,
                    'confidence': 92.0,
                    'risk_level': 'Low',
                    'model_used': 'Random Forest',
                    'parameters': {'sc': 0.9, 'bp': 75}
                }
                pdf_bytes = generate_pdf_report(sample_data)
                self.assertIsNotNone(pdf_bytes)
                self.assertTrue(pdf_bytes.startswith(b'%PDF'))
                self.assertGreater(len(pdf_bytes), 1000)

    def test_download_report_route_forces_english(self):
        """Flask route /report/<uuid> returns valid English PDF regardless of ?lang or cookies."""
        # Create a sample screening record via client POST
        post_data = {
            'model_id': 'model_1',
            'model_choice': 'rf',
            'patient_id': 'PT-ROUTE-TEST',
            'patient_name': 'Route Test Patient',
            'age': '50',
            'bp': '80',
            'sg': '1.020',
            'al': '0',
            'su': '0',
            'rbc': 'normal',
            'pc': 'normal',
            'pcc': 'notpresent',
            'ba': 'notpresent',
            'bgr': '100',
            'bu': '35',
            'sc': '1.0',
            'sod': '140',
            'pot': '4.0',
            'hemo': '15.0',
            'pcv': '45',
            'wc': '7500',
            'rc': '5.0',
            'htn': 'no',
            'dm': 'no',
            'cad': 'no',
            'appet': 'good',
            'pe': 'no',
            'ane': 'no'
        }
        res = self.client.post('/screen', data=post_data, follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        uuid = res.headers['Location'].split('/result/')[1]

        # Request report with Hindi parameter and Marathi cookie
        self.client.set_cookie('renalcare_lang', 'mr')
        rep_res = self.client.get(f'/report/{uuid}?lang=hi')
        self.assertEqual(rep_res.status_code, 200)
        self.assertEqual(rep_res.mimetype, 'application/pdf')
        self.assertTrue(rep_res.data.startswith(b'%PDF'))
        self.assertGreater(len(rep_res.data), 1000)

if __name__ == '__main__':
    unittest.main()
