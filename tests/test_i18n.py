import os
import json
import unittest
from app import create_app
from report_generator import generate_pdf_report, LANG_NAMES

class TestMultilingualAccessibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.client = cls.app.test_client()
        cls.translations_dir = os.path.join(os.path.dirname(__file__), '..', 'translations')
        cls.expected_languages = ['en', 'hi', 'bn', 'mr', 'te', 'ta', 'ur', 'gu']

    def test_translation_files_exist_and_valid_json(self):
        """All 8 translation JSON files exist, parse cleanly, and have common clinical keys."""
        en_path = os.path.join(self.translations_dir, 'en.json')
        self.assertTrue(os.path.exists(en_path), "en.json missing")
        with open(en_path, 'r', encoding='utf-8') as f:
            en_data = json.load(f)

        en_keys = set(en_data.keys())
        self.assertGreater(len(en_keys), 100, "en.json should have 100+ keys")

        for lang in self.expected_languages:
            fpath = os.path.join(self.translations_dir, f"{lang}.json")
            self.assertTrue(os.path.exists(fpath), f"{lang}.json does not exist")
            with open(fpath, 'r', encoding='utf-8') as f:
                lang_data = json.load(f)
            self.assertIsInstance(lang_data, dict)
            # Ensure every key in en.json is present in the regional translation
            missing_keys = en_keys - set(lang_data.keys())
            self.assertEqual(len(missing_keys), 0, f"{lang}.json is missing keys: {missing_keys}")

    def test_urdu_rtl_and_metadata(self):
        """Verify language metadata has Urdu as rtl and others as ltr."""
        from ml.generate_translations import LANG_META
        self.assertEqual(LANG_META['ur']['dir'], 'rtl')
        for code, meta in LANG_META.items():
            if code != 'ur':
                self.assertEqual(meta['dir'], 'ltr')

    def test_client_assets_exist(self):
        """Verify client translation bundle and i18n script exist."""
        bundle_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'js', 'translations_data.js')
        i18n_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'js', 'i18n.js')
        self.assertTrue(os.path.exists(bundle_path), "translations_data.js missing")
        self.assertTrue(os.path.exists(i18n_path), "i18n.js missing")
        self.assertGreater(os.path.getsize(bundle_path), 5000)
        self.assertGreater(os.path.getsize(i18n_path), 2000)

    def test_navbar_about_ckd_removed_and_language_dropdown_added(self):
        """Verify About CKD is completely removed from nav links and replaced with Language dropdown."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # Nav links list must NOT contain About CKD
        self.assertNotIn('<li><a href="/" class="nav-link">About CKD</a></li>', html)
        self.assertNotIn('<li><a href="/#about-ckd" class="nav-link">About CKD</a></li>', html)

        # Nav links list must contain the Language dropdown button & items
        self.assertIn('id="navLangToggle"', html)
        self.assertIn('id="navLangDropdown"', html)
        self.assertIn('data-lang="hi"', html)
        self.assertIn('data-lang="bn"', html)
        self.assertIn('data-lang="mr"', html)
        self.assertIn('data-lang="te"', html)
        self.assertIn('data-lang="ta"', html)
        self.assertIn('data-lang="ur"', html)
        self.assertIn('data-lang="gu"', html)
        self.assertIn('data-lang="en"', html)

    def test_welcome_language_modal_present(self):
        """First-time visitor welcome modal with 8 cards and continue button is in base HTML."""
        res = self.client.get('/screen')
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        self.assertIn('id="welcomeLanguageModal"', html)
        self.assertIn('class="welcome-modal-card"', html)
        self.assertIn('id="btnWelcomeContinue"', html)
        self.assertIn('class="welcome-lang-grid"', html)

    def test_easy_mode_completely_removed(self):
        """Verify Easy Mode elements and toggles are completely removed from UI."""
        res = self.client.get('/screen')
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        self.assertNotIn('id="btnEasyModeToggle"', html)
        self.assertNotIn('class="btn-easy-toggle"', html)
        self.assertNotIn('id="easyModeBanner"', html)
        self.assertNotIn('easy-mode-helper', html)

        res_home = self.client.get('/')
        html_home = res_home.get_data(as_text=True)
        self.assertNotIn('id="btnEasyModeToggle"', html_home)
        self.assertNotIn('id="easyModeBanner"', html_home)

    def test_voice_tts_button_present(self):
        """Verify Web Speech Voice TTS button is present on screening & landing pages."""
        res_screen = self.client.get('/screen')
        html_screen = res_screen.get_data(as_text=True)
        self.assertIn('class="btn-tts"', html_screen)

        res_home = self.client.get('/')
        html_home = res_home.get_data(as_text=True)
        self.assertIn('class="btn-tts"', html_home)

    def test_pdf_report_all_8_languages(self):
        """PDF generation succeeds without error for each of the 8 supported languages."""
        sample_record = {
            'screening_uuid': 'test-uuid-multilingual-12345',
            'patient_id': 'PT-TEST-001',
            'patient_name': 'A. Test Patient',
            'age': 52,
            'sex': 'Male',
            'prediction_code': 'ckd',
            'prediction_label': 'CKD Indicators Detected',
            'risk_level': 'High',
            'probability': 88.5,
            'confidence': 88.5,
            'model_used': 'Random Forest',
            'formatted_date': 'Nov 27, 2026, 11:00 PM',
            'parameters': {'sc': 2.1, 'hemo': 10.5, 'bp': 90, 'al': 2, 'bgr': 160}
        }

        for lang in self.expected_languages:
            with self.subTest(lang=lang):
                pdf_bytes = generate_pdf_report(sample_record, lang=lang)
                self.assertIsNotNone(pdf_bytes)
                self.assertTrue(pdf_bytes.startswith(b'%PDF'), f"Failed to generate valid PDF for lang={lang}")
                self.assertGreater(len(pdf_bytes), 1000)

if __name__ == '__main__':
    unittest.main()
