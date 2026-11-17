import unittest
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import create_app
from database import get_screenings_history

class TestFlaskRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_db_path = os.path.join(os.path.dirname(__file__), 'test_renalcare.db')
        if os.path.exists(test_db_path):
            try:
                os.remove(test_db_path)
            except Exception:
                pass
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
        self.assertTrue(data['models_loaded']['decision_tree'])

    def test_api_predict_valid(self):
        payload = {
            'patient_id': 'PT-TEST-API-01',
            'patient_name': 'Test User',
            'parameters': {
                'age': 55, 'bp': 85, 'hemo': 10.5, 'sc': 2.5, 'htn': 'yes'
            }
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIsNotNone(data['screening_uuid'])
        self.assertIn('prediction', data)
        self.assertTrue(data['prediction']['is_flagged'])

    def test_api_predict_invalid_parameters(self):
        payload = {
            'patient_id': 'PT-INVALID',
            'parameters': "this should be a dictionary not string"
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

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

    def test_api_history_and_item(self):
        # Fetch history list
        resp = self.client.get('/api/history')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn('records', data)
        self.assertGreater(data['total_count'], 0)

        # Fetch single record
        first_uuid = data['records'][0]['screening_uuid']
        item_resp = self.client.get(f'/api/history/{first_uuid}')
        self.assertEqual(item_resp.status_code, 200)
        item_data = json.loads(item_resp.data)
        self.assertEqual(item_data['screening_uuid'], first_uuid)

    def test_api_history_item_not_found(self):
        resp = self.client.get('/api/history/non-existent-uuid-999')
        self.assertEqual(resp.status_code, 404)

    def test_api_dashboard(self):
        resp = self.client.get('/api/dashboard')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn('total', data)
        self.assertIn('flagged', data)

    def test_api_metrics(self):
        resp = self.client.get('/api/metrics')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn('champion_model', data)
        self.assertIn('models', data)

    def test_history_view(self):
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Patient Screening Records', response.data)

    def test_research_view(self):
        response = self.client.get('/research')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Machine Learning Pipeline Architecture', response.data)

    def test_404_handler(self):
        response = self.client.get('/non_existent_page_path')
        self.assertEqual(response.status_code, 404)

if __name__ == '__main__':
    unittest.main()
