import os
import sys
import io
import json
import sqlite3
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from database import (
    init_db,
    get_latest_email_delivery,
    get_email_delivery_by_token_hash,
    get_screening_by_uuid,
    create_email_delivery,
)
from email_service import (
    validate_and_normalize_email,
    mask_email,
    generate_download_token,
    hash_download_token,
    build_https_download_url,
    ResendEmailProvider,
    deliver_email_with_retries,
    resend_screening_email,
)


def _mock_urlopen_success(message_id="re_msg_12345"):
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({"id": message_id}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    return mock_resp


class TestEmailDeliveryFeature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_email_renalcare.db"))
        if os.path.exists(cls.test_db_path):
            os.remove(cls.test_db_path)

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
        os.environ["RESEND_API_KEY"] = "re_test_dummy_key_000"
        os.environ["MAIL_FROM"] = "RenalCare <onboarding@resend.dev>"
        os.environ["APP_BASE_URL"] = "https://ckd-screening-system.onrender.com"
        os.environ["EMAIL_REPORT_ENABLED"] = "true"

        init_db(db_path=cls.test_db_path)
        cls.app = create_app()
        cls.app.config["TESTING"] = True
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
        # Reset email env defaults before each test
        os.environ["RESEND_API_KEY"] = "re_test_dummy_key_000"
        os.environ["MAIL_FROM"] = "RenalCare <onboarding@resend.dev>"
        os.environ["APP_BASE_URL"] = "https://ckd-screening-system.onrender.com"
        os.environ["EMAIL_REPORT_ENABLED"] = "true"
        self.app.config["ENFORCE_CSRF"] = False
        self.base_form = {
            "patient_id": "PT-EMAIL-TEST",
            "patient_name": "Ananya Rao",
            "age": "46",
            "sex": "Female",
            "model_choice": "Random Forest",
            "bp": "80",
            "sg": "1.020",
            "al": "0",
            "su": "0",
            "sc": "0.8",
            "hemo": "14.2",
            "pcv": "43",
            "wc": "7100",
            "rc": "4.9",
            "htn": "no",
            "dm": "no",
            "cad": "no",
            "appet": "good",
            "pe": "no",
            "ane": "no",
        }

    # 1. Screening without an email
    @patch("email_service.urllib.request.urlopen")
    def test_screening_without_email(self, mock_urlopen):
        form = dict(self.base_form, patient_id="PT-NO-EMAIL-01")
        resp = self.client.post("/screen", data=form, follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        uuid_val = resp.headers["Location"].split("/result/")[-1]

        mock_urlopen.assert_not_called()
        delivery = get_latest_email_delivery(uuid_val)
        self.assertIsNotNone(delivery)
        self.assertEqual(delivery["delivery_status"], "not_provided")

        res_page = self.client.get(f"/result/{uuid_val}")
        self.assertEqual(res_page.status_code, 200)
        self.assertNotIn(b"Patient Email Report Delivery", res_page.data)
        self.assertNotIn(b"emailDeliveryCard", res_page.data)
        self.assertIn(b"Download Official PDF Report", res_page.data)

    # 2. Screening with an email but no consent
    @patch("email_service.urllib.request.urlopen")
    def test_screening_email_without_consent(self, mock_urlopen):
        form = dict(self.base_form, patient_id="PT-NOCONSENT-01", patient_email="Patient.NoConsent@Example.COM")
        resp = self.client.post("/screen", data=form, follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        uuid_val = resp.headers["Location"].split("/result/")[-1]

        mock_urlopen.assert_not_called()
        delivery = get_latest_email_delivery(uuid_val)
        self.assertEqual(delivery["delivery_status"], "consent_not_provided")
        self.assertFalse(delivery["consent_recorded"])
        self.assertIsNone(delivery["token_hash"])

        res_page = self.client.get(f"/result/{uuid_val}")
        self.assertEqual(res_page.status_code, 200)
        self.assertNotIn(b"Patient Email Report Delivery", res_page.data)
        self.assertNotIn(b"emailDeliveryCard", res_page.data)

    # 3 & 6. Screening with valid email and consent + successful provider response + privacy check
    @patch("email_service.urllib.request.urlopen")
    def test_screening_with_valid_email_and_consent_success(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_success("re_msg_abc999")
        form = dict(
            self.base_form,
            patient_id="PT-EMAIL-OK-01",
            patient_email="  Patient.Valid@Example.COM ",
            email_consent="yes",
        )
        resp = self.client.post("/screen", data=form, follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        uuid_val = resp.headers["Location"].split("/result/")[-1]

        mock_urlopen.assert_called_once()
        sent_req = mock_urlopen.call_args[0][0]
        sent_payload = json.loads(sent_req.data.decode("utf-8"))

        # Verify normalization, headers, subject, privacy, and HTTPS download link
        self.assertEqual(sent_payload["to"], ["patient.valid@example.com"])
        self.assertEqual(sent_payload["subject"], "Your RenalCare Screening Report Is Ready")
        self.assertIn("Idempotency-key", {k.capitalize(): v for k, v in sent_req.headers.items()})
        self.assertIn("https://ckd-screening-system.onrender.com/report/download/", sent_payload["html"])
        self.assertIn("https://ckd-screening-system.onrender.com/report/download/", sent_payload["text"])
        self.assertIn("Download Your Report", sent_payload["html"])
        self.assertIn("24 hours", sent_payload["html"])
        # Privacy: no clinical findings, scores, or patient IDs in email or URL
        self.assertNotIn("PT-EMAIL-OK-01", sent_payload["html"])
        self.assertNotIn("creatinine", sent_payload["html"].lower())
        self.assertNotIn("notckd", sent_payload["html"].lower())

        delivery = get_latest_email_delivery(uuid_val)
        self.assertEqual(delivery["delivery_status"], "sent")
        self.assertEqual(delivery["provider_message_id"], "re_msg_abc999")
        self.assertTrue(delivery["consent_recorded"])
        self.assertEqual(delivery["attempts"], 1)

        # Result page preserves clean outcome view and PDF download button
        res_page = self.client.get(f"/result/{uuid_val}")
        self.assertEqual(res_page.status_code, 200)
        self.assertNotIn(b"Patient Email Report Delivery", res_page.data)
        self.assertNotIn(b"emailDeliveryCard", res_page.data)
        self.assertIn(b"Download Official PDF Report", res_page.data)

        # Privacy: Dashboard and History APIs must never expose recipient_email
        dash_json = json.loads(self.client.get("/api/dashboard").data)
        self.assertNotIn("patient.valid@example.com", json.dumps(dash_json))
        hist_json = json.loads(self.client.get("/api/history").data)
        self.assertNotIn("patient.valid@example.com", json.dumps(hist_json))

    # 4. Invalid email handling (does NOT block screening)
    @patch("email_service.urllib.request.urlopen")
    def test_invalid_email_does_not_block_screening(self, mock_urlopen):
        norm, err = validate_and_normalize_email("bad..email@invalid\n")
        self.assertIsNone(norm)
        self.assertEqual(err, "invalid_email")

        form = dict(self.base_form, patient_id="PT-BADEMAIL-01", patient_email="not-an-email", email_consent="yes")
        resp = self.client.post("/screen", data=form, follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        uuid_val = resp.headers["Location"].split("/result/")[-1]

        mock_urlopen.assert_not_called()
        delivery = get_latest_email_delivery(uuid_val)
        self.assertEqual(delivery["delivery_status"], "failed")
        self.assertEqual(delivery["error_category"], "invalid_email")
        self.assertIsNotNone(get_screening_by_uuid(uuid_val))

    # 5. Missing Resend API key handled gracefully
    @patch("email_service.urllib.request.urlopen")
    def test_missing_resend_api_key_graceful(self, mock_urlopen):
        os.environ["RESEND_API_KEY"] = ""
        form = dict(self.base_form, patient_id="PT-NOKEY-01", patient_email="user@example.com", email_consent="yes")
        resp = self.client.post("/screen", data=form, follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        uuid_val = resp.headers["Location"].split("/result/")[-1]

        mock_urlopen.assert_not_called()
        delivery = get_latest_email_delivery(uuid_val)
        self.assertEqual(delivery["delivery_status"], "not_configured")
        self.assertEqual(delivery["error_category"], "missing_api_key")

        # Direct PDF download still works normally
        pdf_resp = self.client.get(f"/report/{uuid_val}")
        self.assertEqual(pdf_resp.status_code, 200)
        self.assertTrue(pdf_resp.data.startswith(b"%PDF"))

        res_page = self.client.get(f"/result/{uuid_val}")
        self.assertEqual(res_page.status_code, 200)
        self.assertNotIn(b"Patient Email Report Delivery", res_page.data)
        self.assertNotIn(b"emailDeliveryCard", res_page.data)

    # 7. Provider API permanent failure (e.g. 403 domain/recipient restriction on free tier)
    @patch("email_service.urllib.request.urlopen")
    def test_provider_permanent_failure_no_retry(self, mock_urlopen):
        err_body = json.dumps({
            "statusCode": 403,
            "name": "validation_error",
            "message": "You can only send testing emails to your own email address."
        }).encode("utf-8")
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.resend.com/emails",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=io.BytesIO(err_body),
        )

        form = dict(self.base_form, patient_id="PT-403-01", patient_email="other@example.com", email_consent="yes")
        resp = self.client.post("/screen", data=form, follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        uuid_val = resp.headers["Location"].split("/result/")[-1]

        # Must NOT endlessly retry permanent 403 error
        self.assertEqual(mock_urlopen.call_count, 1)
        delivery = get_latest_email_delivery(uuid_val)
        self.assertEqual(delivery["delivery_status"], "failed")
        self.assertEqual(delivery["error_category"], "domain_verification_required")
        self.assertEqual(delivery["attempts"], 1)

    # 8. Temporary failure and retry behaviour
    @patch("email_service.urllib.request.urlopen")
    def test_temporary_failure_retries_and_succeeds(self, mock_urlopen):
        err_body = json.dumps({"statusCode": 429, "name": "rate_limit_exceeded", "message": "Slow down"}).encode("utf-8")
        http_429 = urllib.error.HTTPError(
            url="https://api.resend.com/emails",
            code=429,
            msg="Too Many Requests",
            hdrs={"retry-after": "0.01"},
            fp=io.BytesIO(err_body),
        )
        # Fail twice with temporary error, then succeed on 3rd attempt
        mock_urlopen.side_effect = [http_429, TimeoutError("timed out"), _mock_urlopen_success("re_retry_ok")]

        # Create screening first
        form = dict(self.base_form, patient_id="PT-RETRY-01")
        resp = self.client.post("/screen", data=form, follow_redirects=False)
        uuid_val = resp.headers["Location"].split("/result/")[-1]

        raw_tok, tok_hash, exp_dt = generate_download_token(24)
        delivery_id = create_email_delivery(
            screening_uuid=uuid_val,
            recipient_email="retry@example.com",
            consent_recorded=True,
            delivery_status="pending",
            idempotency_key="idem-retry-1",
            token_hash=tok_hash,
            token_expiry=exp_dt,
        )
        provider = ResendEmailProvider(api_key="re_test", mail_from="RenalCare <onboarding@resend.dev>")
        result = deliver_email_with_retries(
            delivery_id=delivery_id,
            screening_uuid=uuid_val,
            recipient_email="retry@example.com",
            subject="Your RenalCare Screening Report Is Ready",
            html_body="<p>test</p>",
            text_body="test",
            idempotency_key="idem-retry-1",
            provider=provider,
            max_attempts=3,
            retry_delays=(0.0, 0.0),
        )
        self.assertTrue(result.accepted)
        self.assertEqual(mock_urlopen.call_count, 3)

        delivery = get_latest_email_delivery(uuid_val)
        self.assertEqual(delivery["delivery_status"], "sent")
        self.assertEqual(delivery["attempts"], 3)
        self.assertEqual(delivery["provider_message_id"], "re_retry_ok")

    # 9. Duplicate email prevention
    @patch("email_service.urllib.request.urlopen")
    def test_duplicate_email_prevention(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_success("re_dup_guard")
        form = dict(self.base_form, patient_id="PT-DUP-01", patient_email="dup@example.com", email_consent="yes")
        resp = self.client.post("/screen", data=form, follow_redirects=False)
        uuid_val = resp.headers["Location"].split("/result/")[-1]
        delivery = get_latest_email_delivery(uuid_val)

        # Calling deliver_email_with_retries again on already-sent delivery_id must NOT re-call provider
        provider = ResendEmailProvider(api_key="re_test", mail_from="RenalCare <onboarding@resend.dev>")
        res2 = deliver_email_with_retries(
            delivery_id=delivery["id"],
            screening_uuid=uuid_val,
            recipient_email="dup@example.com",
            subject="Test",
            html_body="<p>test</p>",
            text_body="test",
            idempotency_key=delivery["idempotency_key"],
            provider=provider,
        )
        self.assertTrue(res2.accepted)
        self.assertEqual(mock_urlopen.call_count, 1)

        # Cooldown blocks immediate rapid duplicate resend
        record = get_screening_by_uuid(uuid_val)
        with self.app.app_context():
            ok, msg, meta = resend_screening_email(self.app, record, cooldown_seconds=30, background=False)
        self.assertFalse(ok)
        self.assertTrue(meta.get("duplicate_prevented"))
        self.assertEqual(mock_urlopen.call_count, 1)

    # 10, 11, 12, 13. Secure token generation, valid download, expired rejection, invalid rejection
    @patch("email_service.urllib.request.urlopen")
    def test_secure_token_download_and_expiry(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_success("re_tok_001")
        raw_token, tok_hash, exp_dt = generate_download_token(24)
        self.assertGreaterEqual(len(raw_token), 32)
        self.assertEqual(hash_download_token(raw_token), tok_hash)
        self.assertTrue(build_https_download_url(raw_token).startswith("https://"))

        # Submit screening & inspect issued link in mocked email
        form = dict(self.base_form, patient_id="PT-TOK-01", patient_email="tok@example.com", email_consent="yes")
        resp = self.client.post("/screen", data=form, follow_redirects=False)
        uuid_val = resp.headers["Location"].split("/result/")[-1]

        sent_req = mock_urlopen.call_args[0][0]
        sent_html = json.loads(sent_req.data.decode("utf-8"))["html"]
        token_prefix = "https://ckd-screening-system.onrender.com/report/download/"
        extracted_token = sent_html.split(token_prefix)[1].split('"')[0]

        # Confirm raw token is NOT stored in database (only hash!)
        conn = sqlite3.connect(self.test_db_path)
        raw_hits = conn.execute(
            "SELECT COUNT(*) FROM email_deliveries WHERE token_hash = ?;", (extracted_token,)
        ).fetchone()[0]
        hash_hits = conn.execute(
            "SELECT COUNT(*) FROM email_deliveries WHERE token_hash = ?;", (hash_download_token(extracted_token),)
        ).fetchone()[0]
        conn.close()
        self.assertEqual(raw_hits, 0)
        self.assertEqual(hash_hits, 1)

        # 11. Valid token downloads PDF successfully
        dl_resp = self.client.get(f"/report/download/{extracted_token}")
        self.assertEqual(dl_resp.status_code, 200)
        self.assertEqual(dl_resp.mimetype, "application/pdf")
        self.assertTrue(dl_resp.data.startswith(b"%PDF"))
        self.assertIn("no-store", dl_resp.headers.get("Cache-Control", ""))

        # 13. Invalid / tampered token is rejected safely
        bad_resp = self.client.get("/report/download/invalid_tampered_token_12345")
        self.assertEqual(bad_resp.status_code, 404)
        self.assertIn(b"Download Link Invalid or Unavailable", bad_resp.data)

        # 12. Expire the token in DB and verify 410 Gone + original screening record preserved
        past_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(self.test_db_path)
        conn.execute(
            "UPDATE email_deliveries SET token_expiry = ? WHERE screening_uuid = ?;",
            (past_ts, uuid_val),
        )
        conn.commit()
        conn.close()

        exp_resp = self.client.get(f"/report/download/{extracted_token}")
        self.assertEqual(exp_resp.status_code, 410)
        self.assertIn(b"This Report Download Link Has Expired", exp_resp.data)
        self.assertIsNotNone(get_screening_by_uuid(uuid_val))

    # 14. Unauthorized report resend & CSRF protection
    @patch("email_service.urllib.request.urlopen")
    def test_unauthorized_resend_and_csrf_rejected(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_success("re_auth_001")
        form = dict(self.base_form, patient_id="PT-AUTH-01", patient_email="auth@example.com", email_consent="yes")
        resp = self.client.post("/screen", data=form, follow_redirects=False)
        uuid_val = resp.headers["Location"].split("/result/")[-1]

        # Fresh anonymous client (like a patient opening an email link without clinic session)
        anon_client = self.app.test_client()
        unauth_resp = anon_client.post(f"/result/{uuid_val}/resend-email")
        self.assertEqual(unauth_resp.status_code, 403)

        # Authenticated clinic session WITHOUT valid CSRF token when ENFORCE_CSRF is enabled
        self.app.config["ENFORCE_CSRF"] = True
        try:
            self.client.get(f"/result/{uuid_val}")
            bad_csrf_resp = self.client.post(f"/result/{uuid_val}/resend-email", data={"csrf_token": "wrong-token"})
            self.assertEqual(bad_csrf_resp.status_code, 403)
        finally:
            self.app.config["ENFORCE_CSRF"] = False

    # 15. Authorized healthcare worker resends report with new link (invalidating old link)
    @patch("email_service.urllib.request.urlopen")
    def test_authorized_resend_issues_replacement_link(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_success("re_initial")
        form = dict(self.base_form, patient_id="PT-RESEND-01", patient_email="resend@example.com", email_consent="yes")
        resp = self.client.post("/screen", data=form, follow_redirects=False)
        uuid_val = resp.headers["Location"].split("/result/")[-1]

        first_html = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))["html"]
        token_prefix = "https://ckd-screening-system.onrender.com/report/download/"
        first_token = first_html.split(token_prefix)[1].split('"')[0]

        # Authorized clinician visits result page and submits resend with valid CSRF
        self.app.config["ENFORCE_CSRF"] = True
        try:
            res_page = self.client.get(f"/result/{uuid_val}")
            self.assertEqual(res_page.status_code, 200)
            with self.client.session_transaction() as sess:
                csrf_tok = sess["csrf_token"]

            mock_urlopen.return_value = _mock_urlopen_success("re_replacement")
            resend_resp = self.client.post(
                f"/result/{uuid_val}/resend-email",
                data={"csrf_token": csrf_tok},
                follow_redirects=False,
            )
            self.assertEqual(resend_resp.status_code, 302)
        finally:
            self.app.config["ENFORCE_CSRF"] = False

        second_html = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))["html"]
        second_token = second_html.split(token_prefix)[1].split('"')[0]
        self.assertNotEqual(first_token, second_token)

        # Old token must no longer work; new token downloads PDF cleanly
        self.assertEqual(self.client.get(f"/report/download/{first_token}").status_code, 404)
        new_dl = self.client.get(f"/report/download/{second_token}")
        self.assertEqual(new_dl.status_code, 200)
        self.assertTrue(new_dl.data.startswith(b"%PDF"))

    # 17. PostgreSQL dialect compatibility check (mocked psycopg2 connection)
    def test_postgres_dialect_compatibility(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [42]
        mock_conn.cursor.return_value = mock_cursor

        with patch("database.get_connection", return_value=(mock_conn, "postgres")):
            init_db(db_url="postgresql://user:pass@localhost:5432/db")
            executed_sql = " ".join(str(c[0][0]) for c in mock_cursor.execute.call_args_list)
            self.assertIn("CREATE TABLE IF NOT EXISTS email_deliveries", executed_sql)
            self.assertIn("SERIAL PRIMARY KEY", executed_sql)

            did = create_email_delivery(
                screening_uuid="pg-uuid-1",
                recipient_email="pg@example.com",
                consent_recorded=True,
                delivery_status="pending",
                token_hash="abc",
                token_expiry=datetime.now(timezone.utc) + timedelta(hours=24),
                db_url="postgresql://user:pass@localhost:5432/db",
            )
            self.assertEqual(did, 42)
            insert_sql = str(mock_cursor.execute.call_args[0][0])
            self.assertIn("RETURNING id", insert_sql)
            self.assertIn("%s", insert_sql)


if __name__ == "__main__":
    unittest.main()
