import os
import sys
import json
import sqlite3
import unittest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from database import (
    init_db,
    get_patient_profile,
    upsert_patient_profile,
    get_patient_profiles,
    get_email_diagnostics,
    get_latest_email_delivery,
    get_screening_by_uuid,
)
from email_service import get_email_service_diagnostics


def _mock_resend_ok(message_id="re_profile_12345"):
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({"id": message_id}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    return mock_resp


class TestPatientProfileAndEmailSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_profile_renalcare.db"))
        if os.path.exists(cls.test_db_path):
            try:
                os.remove(cls.test_db_path)
            except Exception:
                pass

        cls.orig_env = {
            "DATABASE_PATH": os.environ.get("DATABASE_PATH"),
            "DATABASE_URL": os.environ.get("DATABASE_URL"),
            "TESTING": os.environ.get("TESTING"),
            "RESEND_API_KEY": os.environ.get("RESEND_API_KEY"),
            "MAIL_FROM": os.environ.get("MAIL_FROM"),
            "APP_BASE_URL": os.environ.get("APP_BASE_URL"),
            "EMAIL_REPORT_ENABLED": os.environ.get("EMAIL_REPORT_ENABLED"),
        }

        os.environ["DATABASE_PATH"] = cls.test_db_path
        os.environ.pop("DATABASE_URL", None)
        os.environ["TESTING"] = "1"
        os.environ["RESEND_API_KEY"] = "re_test_key_profile"
        os.environ["MAIL_FROM"] = "RenalCare <onboarding@resend.dev>"
        os.environ["APP_BASE_URL"] = "https://ckd-screening-system.onrender.com"
        os.environ["EMAIL_REPORT_ENABLED"] = "true"

        init_db(db_path=cls.test_db_path)
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.app.config["ENFORCE_CSRF"] = False
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        for key, val in cls.orig_env.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)

        if os.path.exists(cls.test_db_path):
            try:
                os.remove(cls.test_db_path)
            except Exception:
                pass

    def setUp(self):
        self.app.config["ENFORCE_CSRF"] = False
        self.base_screening = {
            "patient_id": "PT-PROFILE-001",
            "patient_name": "Clinical Profile Test Patient",
            "age": "52",
            "sex": "Male",
            "model_choice": "Random Forest",
            "bp": "80",
            "sg": "1.020",
            "al": "0",
            "su": "0",
            "rbc": "normal",
            "pc": "normal",
            "pcc": "notpresent",
            "ba": "notpresent",
            "bgr": "100",
            "bu": "28",
            "sc": "0.9",
            "sod": "138",
            "pot": "4.1",
            "hemo": "14.8",
            "pcv": "44",
            "wc": "7100",
            "rc": "4.9",
            "htn": "no",
            "dm": "no",
            "cad": "no",
            "appet": "good",
            "pe": "no",
            "ane": "no",
        }

    # 1. Profile Creation & Retrieval via Database Functions
    def test_profile_creation_and_retrieval(self):
        p = upsert_patient_profile(
            patient_id="PT-PERSIST-101",
            patient_name="Persistent Patient A",
            email="patient.a@clinic.org",
            consent_recorded=True,
        )
        self.assertIsNotNone(p)
        self.assertEqual(p["patient_id"], "PT-PERSIST-101")
        self.assertEqual(p["patient_name"], "Persistent Patient A")
        self.assertEqual(p["email"], "patient.a@clinic.org")
        self.assertTrue(p["consent_recorded"])
        self.assertIsNotNone(p["consent_at"])
        self.assertIsNotNone(p["created_at"])
        self.assertIsNotNone(p["updated_at"])

        # Fetch independently by ID
        fetched = get_patient_profile("PT-PERSIST-101")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["patient_name"], "Persistent Patient A")

    # 2. Updating Existing Profile Attributes
    def test_profile_update_maintains_identity(self):
        upsert_patient_profile(
            patient_id="PT-PERSIST-102",
            patient_name="Original Name",
            email="old.email@clinic.org",
            consent_recorded=False,
        )

        # Update with new email and consent
        updated = upsert_patient_profile(
            patient_id="PT-PERSIST-102",
            patient_name="Updated Name",
            email="new.email@clinic.org",
            consent_recorded=True,
        )
        self.assertEqual(updated["patient_id"], "PT-PERSIST-102")
        self.assertEqual(updated["patient_name"], "Updated Name")
        self.assertEqual(updated["email"], "new.email@clinic.org")
        self.assertTrue(updated["consent_recorded"])

    # 3. Non-existent and Invalid Patient IDs
    def test_profile_non_existent_and_invalid_queries(self):
        self.assertIsNone(get_patient_profile("NON_EXISTENT_ID_9999"))
        self.assertIsNone(get_patient_profile(""))
        self.assertIsNone(get_patient_profile(None))
        with self.assertRaises(ValueError):
            upsert_patient_profile(patient_id="")

    # 4. Listing and Searching Patient Profiles
    def test_profile_listing_and_search(self):
        upsert_patient_profile("PT-SEARCH-A", "Alpha Patient", "alpha@test.org")
        upsert_patient_profile("PT-SEARCH-B", "Beta Patient", "beta@test.org")

        profiles, count = get_patient_profiles(limit=10, search="Alpha")
        self.assertGreaterEqual(count, 1)
        self.assertTrue(any(p["patient_id"] == "PT-SEARCH-A" for p in profiles))

        profiles_b, count_b = get_patient_profiles(limit=10, search="beta@test.org")
        self.assertGreaterEqual(count_b, 1)
        self.assertTrue(any(p["patient_id"] == "PT-SEARCH-B" for p in profiles_b))

    # 5. REST API: GET /api/patient/<patient_id>
    def test_api_get_patient_profile_found_and_not_found(self):
        upsert_patient_profile("PT-API-001", "API Test Subject", "api.test@clinic.org", consent_recorded=True)

        resp = self.client.get("/api/patient/PT-API-001")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["found"])
        self.assertEqual(data["profile"]["patient_id"], "PT-API-001")
        self.assertEqual(data["profile"]["patient_name"], "API Test Subject")
        self.assertEqual(data["profile"]["email"], "api.test@clinic.org")
        self.assertTrue(data["profile"]["consent_recorded"])

        resp_missing = self.client.get("/api/patient/UNKNOWN-PT-999")
        self.assertEqual(resp_missing.status_code, 200)
        data_missing = json.loads(resp_missing.data)
        self.assertFalse(data_missing["found"])

    # 6. Screening Submission Auto-Creates & Links Patient Profile
    @patch("email_service.urllib.request.urlopen")
    def test_screening_creates_and_links_profile(self, mock_urlopen):
        mock_urlopen.return_value = _mock_resend_ok("re_link_01")
        form = dict(
            self.base_screening,
            patient_id="PT-LINK-001",
            patient_name="Link Patient",
            patient_email="link.patient@clinic.org",
            email_consent="yes",
        )
        resp = self.client.post("/screen", data=form, follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        uuid_val = resp.headers["Location"].split("/result/")[-1]

        # Profile was persisted
        profile = get_patient_profile("PT-LINK-001")
        self.assertIsNotNone(profile)
        self.assertEqual(profile["patient_name"], "Link Patient")
        self.assertEqual(profile["email"], "link.patient@clinic.org")
        self.assertTrue(profile["consent_recorded"])

        # Email delivery was linked to patient_id
        delivery = get_latest_email_delivery(uuid_val)
        self.assertEqual(delivery["patient_id"], "PT-LINK-001")
        self.assertEqual(delivery["delivery_status"], "sent")

    # 7. Rescreening Reuses Stored Profile Email When Form Email Is Blank But Consent Is Given
    @patch("email_service.urllib.request.urlopen")
    def test_rescreening_reuses_stored_profile_email(self, mock_urlopen):
        mock_urlopen.return_value = _mock_resend_ok("re_reuse_01")
        # Pre-seed profile
        upsert_patient_profile("PT-REUSE-001", "Existing Name", "saved.profile@clinic.org", consent_recorded=True)

        # Clinician rescreens patient, leaves email blank, but confirms consent checkbox
        form = dict(
            self.base_screening,
            patient_id="PT-REUSE-001",
            patient_name="",
            patient_email="",  # Blank in form
            email_consent="yes",
        )
        resp = self.client.post("/screen", data=form, follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        uuid_val = resp.headers["Location"].split("/result/")[-1]

        # Email was dispatched to the saved profile address
        mock_urlopen.assert_called_once()
        sent_req = mock_urlopen.call_args[0][0]
        sent_payload = json.loads(sent_req.data.decode("utf-8"))
        self.assertEqual(sent_payload["to"], ["saved.profile@clinic.org"])

        # Check screening record preserved the patient name from profile
        screening = get_screening_by_uuid(uuid_val)
        self.assertEqual(screening["patient_name"], "Existing Name")

    # 8. Screening Without Email Does Not Send Email And Succeeds
    @patch("email_service.urllib.request.urlopen")
    def test_screening_without_email_succeeds(self, mock_urlopen):
        form = dict(
            self.base_screening,
            patient_id="PT-NOEMAIL-001",
            patient_email="",
            email_consent="",
        )
        resp = self.client.post("/screen", data=form, follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        uuid_val = resp.headers["Location"].split("/result/")[-1]

        mock_urlopen.assert_not_called()
        delivery = get_latest_email_delivery(uuid_val)
        self.assertEqual(delivery["delivery_status"], "not_provided")

    # 9. Screening Result Page Contains NO Unwanted Email Status Box
    def test_result_page_contains_no_email_box(self):
        form = dict(self.base_screening, patient_id="PT-CLEAN-UI-001")
        resp = self.client.post("/screen", data=form, follow_redirects=False)
        uuid_val = resp.headers["Location"].split("/result/")[-1]

        res_page = self.client.get(f"/result/{uuid_val}")
        self.assertEqual(res_page.status_code, 200)

        # Explicitly verify removed elements per Phase 1
        self.assertNotIn(b"Patient Email Report Delivery", res_page.data)
        self.assertNotIn(b"id=\"emailDeliveryCard\"", res_page.data)
        self.assertNotIn(b"id=\"emailDeliveryBadge\"", res_page.data)
        self.assertNotIn(b"id=\"emailDeliveryMessage\"", res_page.data)
        self.assertNotIn(b"Resend Report (New 24h Link)", res_page.data)
        self.assertNotIn(b"api/email-status", res_page.data)

        # Verify clinical outcome and direct PDF download button are preserved
        self.assertIn(b"Download Official PDF Report", res_page.data)
        self.assertIn(b"ckd indicators not detected", res_page.data.lower())
        self.assertIn(b"Model Confidence", res_page.data)
        self.assertIn(b"Low Statistical Indication", res_page.data)

    # 10. Developer Diagnostic Endpoint: /api/admin/email-diagnostics
    def test_email_diagnostics_endpoint_security_and_content(self):
        # 1. Unauthenticated client receives 403
        anon_client = self.app.test_client()
        unauth_resp = anon_client.get("/api/admin/email-diagnostics")
        self.assertEqual(unauth_resp.status_code, 403)

        # 2. Authenticated clinician receives 200 with diagnostics
        auth_resp = self.client.get("/api/admin/email-diagnostics")
        self.assertEqual(auth_resp.status_code, 200)
        data = json.loads(auth_resp.data)

        self.assertIn("provider_configured", data)
        self.assertIn("mail_from", data)
        self.assertIn("app_base_url", data)
        self.assertIn("sandbox_restriction_note", data)
        self.assertIn("recent_deliveries", data)
        self.assertIsInstance(data["recent_deliveries"], list)

        # Security check: Never leak raw API key or secrets
        diag_str = json.dumps(data)
        self.assertNotIn("re_test_key_profile", diag_str)
        self.assertNotIn("token_hash", diag_str)


if __name__ == "__main__":
    unittest.main()
