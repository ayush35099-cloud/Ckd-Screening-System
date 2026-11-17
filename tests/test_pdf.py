import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from report_generator import generate_pdf_report

class TestPDFReportGenerator(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()
