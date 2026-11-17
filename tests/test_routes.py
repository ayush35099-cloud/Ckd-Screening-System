import unittest
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import create_app

class TestFlaskRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use an isolated test database
        test_db_path = os.path.join(os.path.dirname(__file__), 'test_renalcare.db')
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        os.environ['DATABASE_PATH'] = test_db_path
        os.environ['TESTING'] = '1'

        cls.app = create_app()
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        test_db_path = os.path.join(os.path.dirname(__file__), 'test_renalcare.db')
        if os.path.exists(test_db_path):
            try:
                os.remove(test_db_path)
            except Exception:
                pass

    def test_landing_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'RenalCare', response.data)
        self.assertIn(b'Better Kidney Health Starts With Early Awareness', response.data)

    def test_dashboard_empty_state(self):
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Primary Care Screening Dashboard', response.data)

    def test_screen_view(self):
        response = self.client.get('/screen')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Patient Screening Evaluation', response.data)
        self.assertIn(b'patient_id', response.data)
        self.assertIn(b'hemo', response.data)

    def test_health_api(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['models_loaded']['random_forest'])

    def test_api_predict(self):
        payload = {
            'patient_id': 'PT-TEST-API',
            'parameters': {
                'age': 55, 'bp': 85, 'hemo': 10.5, 'sc': 2.5, 'htn': 'yes'
            }
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('prediction', data)
        self.assertIn('is_flagged', data['prediction'])

    def test_screen_submission_and_report_download(self):
        form_data = {
            'patient_id': 'PT-FORM-001',
            'patient_name': 'Test Form Patient',
            'age': '48',
            'sex': 'Male',
            'model_choice': 'Random Forest',
            'bp': '80',
            'sg': '1.020',
            'al': '0',
            'su': '0',
            'sc': '0.9',
            'hemo': '14.5',
            'pcv': '44',
            'wc': '7000',
            'rc': '5.0',
            'htn': 'no',
            'dm': 'no',
            'cad': 'no',
            'appet': 'good',
            'pe': 'no',
            'ane': 'no'
        }
        # Post form -> should redirect
        post_resp = self.client.post('/screen', data=form_data, follow_redirects=False)
        self.assertEqual(post_resp.status_code, 302)
        redirect_url = post_resp.headers['Location']
        self.assertIn('/result/', redirect_url)

        # Follow redirect to result page
        result_resp = self.client.get(redirect_url)
        self.assertEqual(result_resp.status_code, 200)
        self.assertIn(b'PT-FORM-001', result_resp.data)

        # Extract uuid from redirect url
        screening_uuid = redirect_url.split('/result/')[-1]

        # Test PDF Report Download
        pdf_resp = self.client.get(f'/report/{screening_uuid}')
        self.assertEqual(pdf_resp.status_code, 200)
        self.assertEqual(pdf_resp.mimetype, 'application/pdf')
        self.assertTrue(pdf_resp.data.startswith(b'%PDF'))

    def test_history_view(self):
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Patient Screening Records', response.data)

    def test_research_view(self):
        response = self.client.get('/research')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Machine Learning Pipeline Architecture', response.data)

if __name__ == '__main__':
    unittest.main()
