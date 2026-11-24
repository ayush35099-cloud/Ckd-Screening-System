import unittest
import os
import sys
import json
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import create_app
from database import get_screenings_history

class TestFlaskRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_renalcare.db'))
        if os.path.exists(cls.test_db_path):
            try:
                os.remove(cls.test_db_path)
            except Exception:
                pass

        # Save existing env vars
        cls.orig_db_path = os.environ.get('DATABASE_PATH')
        cls.orig_db_url = os.environ.get('DATABASE_URL')

        # Enforce isolated test sqlite database
        os.environ['DATABASE_PATH'] = cls.test_db_path
        if 'DATABASE_URL' in os.environ:
            del os.environ['DATABASE_URL']
        os.environ['TESTING'] = '1'

        from database import init_db
        init_db(db_path=cls.test_db_path)

        cls.app = create_app()
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        # Restore environment variables
        if cls.orig_db_path is not None:
            os.environ['DATABASE_PATH'] = cls.orig_db_path
        else:
            os.environ.pop('DATABASE_PATH', None)

        if cls.orig_db_url is not None:
            os.environ['DATABASE_URL'] = cls.orig_db_url
        else:
            os.environ.pop('DATABASE_URL', None)

        os.environ.pop('TESTING', None)

        if hasattr(cls, 'test_db_path') and os.path.exists(cls.test_db_path):
            try:
                os.remove(cls.test_db_path)
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
        self.assertIn(b'Active ML Pipelines', response.data)
        self.assertIn(b'Comparative Model Evaluation Metrics', response.data)

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
        self.assertTrue(data['database_connected'])
        self.assertEqual(data['database_backend'], 'sqlite')
        self.assertEqual(data['database_persistence'], 'ephemeral_sqlite')
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
        self.assertIn('probability_explanation', data['prediction'])
        self.assertIn('missing_parameters', data['prediction'])

    def test_security_headers_present(self):
        response = self.client.get('/')
        self.assertEqual(response.headers.get('X-Content-Type-Options'), 'nosniff')
        self.assertEqual(response.headers.get('X-Frame-Options'), 'SAMEORIGIN')
        self.assertIn('Referrer-Policy', response.headers)

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
        self.assertIn(b'Model Confidence:', result_resp.data)
        self.assertIn(b'Dual-Pipeline Comparative Evaluation', result_resp.data)

        # Extract uuid from redirect url
        screening_uuid = redirect_url.split('/result/')[-1]

        # Test PDF Report Download
        pdf_resp = self.client.get(f'/report/{screening_uuid}')
        self.assertEqual(pdf_resp.status_code, 200)
        self.assertEqual(pdf_resp.mimetype, 'application/pdf')
        self.assertTrue(pdf_resp.data.startswith(b'%PDF'))

    def test_api_history_and_item(self):
        # Ensure at least one record exists in test database
        seed_payload = {
            'patient_id': 'PT-SEED-001',
            'patient_name': 'Seed Patient',
            'parameters': {'age': 55, 'bp': 80, 'sc': 1.1, 'hemo': 13.5}
        }
        self.client.post('/api/predict', json=seed_payload)

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
        self.assertIn('screening_trend', data)
        self.assertIn('algorithm_usage', data)

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

    def test_api_404_handler(self):
        response = self.client.get('/api/non_existent_api_endpoint')
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertEqual(data.get('status'), 404)
        self.assertIn('error', data)

    def test_invalid_screening_submission_missing_id(self):
        # Missing patient_id must trigger redirect with error
        form_data = {
            'patient_name': 'Anonymous',
            'age': '50',
            'bp': '80'
        }
        resp = self.client.post('/screen', data=form_data, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Patient Identifier is required', resp.data)

    def test_screen_submission_with_decision_tree(self):
        form_data = {
            'patient_id': 'PT-DT-TEST-001',
            'patient_name': 'Decision Tree Patient',
            'age': '52',
            'sex': 'Female',
            'model_choice': 'Decision Tree',
            'bp': '75',
            'sc': '0.8',
            'hemo': '14.0'
        }
        resp = self.client.post('/screen', data=form_data, follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        redirect_url = resp.headers['Location']
        self.assertIn('/result/', redirect_url)

        res_page = self.client.get(redirect_url)
        self.assertEqual(res_page.status_code, 200)
        self.assertIn(b'Decision Tree', res_page.data)

    def test_history_search_and_filters_query(self):
        # Seed record for search test
        seed_data = {
            'patient_id': 'PT-SEARCH-001',
            'patient_name': 'Search Patient',
            'age': '52',
            'sex': 'Female',
            'model_choice': 'Random Forest',
            'bp': '75',
            'sc': '0.8',
            'hemo': '14.0'
        }
        self.client.post('/screen', data=seed_data, follow_redirects=True)

        # Test searching by patient ID
        resp = self.client.get('/history?q=PT-SEARCH-001')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'PT-SEARCH-001', resp.data)

        # Test filtering by flagged
        resp_flagged = self.client.get('/history?status=flagged')
        self.assertEqual(resp_flagged.status_code, 200)

        # Test filtering by non_flagged
        resp_clear = self.client.get('/history?status=non_flagged')
        self.assertEqual(resp_clear.status_code, 200)

    # -----------------------------------------------------------------------
    # Phase 3 Tests: Batch Screening, CSV Export, Formula Injection & Warnings
    # -----------------------------------------------------------------------
    def test_batch_template_download(self):
        resp = self.client.get('/screen/batch/template')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('text/csv', resp.headers.get('Content-Type'))
        self.assertIn(b'patient_id,patient_name,age,sex,bp,sg,al,su', resp.data)
        self.assertIn(b'PT-CAMP-001', resp.data)

    def test_batch_screen_get_view(self):
        resp = self.client.get('/screen/batch')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Batch Patient Screening', resp.data)
        self.assertIn(b'Maximum 100 patient rows per upload', resp.data)

    def test_batch_screen_post_valid_csv(self):
        csv_content = (
            "patient_id,patient_name,age,sex,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,htn,dm,cad,appet,pe,ane\n"
            "PT-BATCH-VALID-1,Normal Patient,38,Female,75,1.025,0,0,normal,normal,notpresent,notpresent,100,28,0.8,140,4.2,15.0,45,6900,5.1,no,no,no,good,no,no\n"
            "PT-BATCH-VALID-2,HighRisk Patient,65,Male,95,1.010,3,2,abnormal,abnormal,present,present,220,85,3.2,130,5.5,9.2,28,11500,3.2,yes,yes,yes,poor,yes,yes\n"
        )
        data = {
            'file': (io.BytesIO(csv_content.encode('utf-8')), 'test_patients.csv'),
            'model_choice': 'Random Forest'
        }
        resp = self.client.post('/screen/batch', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Batch Evaluation Summary', resp.data)
        self.assertIn(b'PT-BATCH-VALID-1', resp.data)
        self.assertIn(b'PT-BATCH-VALID-2', resp.data)

    def test_batch_screen_post_invalid_headers(self):
        # CSV missing critical clinical columns like 'sc' and 'hemo'
        bad_csv = "patient_id,patient_name,age\nPT-01,Test,45\n"
        data = {
            'file': (io.BytesIO(bad_csv.encode('utf-8')), 'bad_headers.csv'),
            'model_choice': 'Random Forest'
        }
        resp = self.client.post('/screen/batch', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'missing required feature columns', resp.data)

    def test_batch_screen_post_over_100_rows(self):
        # Create CSV with 105 rows
        header = "patient_id,patient_name,age,sex,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,htn,dm,cad,appet,pe,ane\n"
        row_template = "PT-OVR-{0},Test,45,Male,80,1.020,0,0,normal,normal,notpresent,notpresent,100,30,0.9,138,4.2,14.5,44,7000,5.0,no,no,no,good,no,no\n"
        rows = [row_template.format(i) for i in range(105)]
        large_csv = header + "".join(rows)

        data = {
            'file': (io.BytesIO(large_csv.encode('utf-8')), 'large.csv'),
            'model_choice': 'Random Forest'
        }
        resp = self.client.post('/screen/batch', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Batch limit exceeded', resp.data)

    def test_batch_screen_post_decision_tree(self):
        csv_content = (
            "patient_id,patient_name,age,sex,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,htn,dm,cad,appet,pe,ane\n"
            "PT-BATCH-DT-1,DT Patient,50,Male,80,1.020,0,0,normal,normal,notpresent,notpresent,105,32,0.9,138,4.3,14.2,43,7100,4.9,no,no,no,good,no,no\n"
        )
        data = {
            'file': (io.BytesIO(csv_content.encode('utf-8')), 'dt_batch.csv'),
            'model_choice': 'Decision Tree'
        }
        resp = self.client.post('/screen/batch', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Decision Tree', resp.data)

    def test_batch_export_download(self):
        # First submit a batch to populate session
        csv_content = (
            "patient_id,patient_name,age,sex,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,htn,dm,cad,appet,pe,ane\n"
            "PT-EXP-01,Export Patient,55,Female,85,1.015,1,0,normal,normal,notpresent,notpresent,110,35,1.1,137,4.4,13.8,41,7400,4.7,no,no,no,good,no,no\n"
        )
        data = {
            'file': (io.BytesIO(csv_content.encode('utf-8')), 'export_test.csv'),
            'model_choice': 'Random Forest'
        }
        self.client.post('/screen/batch', data=data, content_type='multipart/form-data')

        # Now download batch export
        resp = self.client.get('/screen/batch/export')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('text/csv', resp.headers.get('Content-Type'))
        self.assertIn(b'PT-EXP-01', resp.data)

    def test_history_csv_export(self):
        resp = self.client.get('/history/export')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('text/csv', resp.headers.get('Content-Type'))
        self.assertIn(b'Screening UUID,Patient ID,Patient Name', resp.data)

    def test_history_csv_export_filters_preserved(self):
        resp = self.client.get('/history/export?status=flagged')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('text/csv', resp.headers.get('Content-Type'))

    def test_csv_formula_injection_protection(self):
        from app import sanitize_csv_field
        # Values starting with formula operators must be escaped with a leading single quote
        self.assertEqual(sanitize_csv_field('=SUM(1,2)'), "'=SUM(1,2)")
        self.assertEqual(sanitize_csv_field('+cmd|'), "'+cmd|")
        self.assertEqual(sanitize_csv_field('-10'), "'-10")
        self.assertEqual(sanitize_csv_field('@evil'), "'@evil")
        self.assertEqual(sanitize_csv_field('normal text'), 'normal text')
        self.assertEqual(sanitize_csv_field(None), '')

    def test_clinical_warnings_in_screen_view(self):
        resp = self.client.get('/screen')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'warning-sc', resp.data)
        self.assertIn(b'warning-bp', resp.data)
        self.assertIn(b'warning-pot', resp.data)
        self.assertIn(b'warning-hemo', resp.data)
        self.assertIn(b'Batch CSV Screening', resp.data)
        self.assertIn(b'CSV Template', resp.data)

if __name__ == '__main__':
    unittest.main()
