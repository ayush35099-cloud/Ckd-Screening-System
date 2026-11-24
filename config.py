import os
import secrets

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or (
        secrets.token_hex(32) if (os.environ.get('RENDER') or os.environ.get('FLASK_ENV') == 'production')
        else 'renalcare-secure-session-key-dsce-aiml-2026'
    )
    DATABASE_PATH = os.environ.get('DATABASE_PATH') or os.path.join(os.path.dirname(__file__), 'renalcare.db')
    DATABASE_URL = os.environ.get('DATABASE_URL')
    DEBUG = os.environ.get('FLASK_DEBUG', '0') == '1'
    TESTING = os.environ.get('TESTING', '0') == '1'
    JSON_SORT_KEYS = False

    # Email Report Delivery Settings (Resend / Modular Provider)
    RESEND_API_KEY = os.environ.get('RESEND_API_KEY', '')
    MAIL_FROM = os.environ.get('MAIL_FROM', '')
    APP_BASE_URL = os.environ.get('APP_BASE_URL', 'https://ckd-screening-system.onrender.com')
    EMAIL_REPORT_ENABLED = os.environ.get('EMAIL_REPORT_ENABLED', 'true').lower() in ('1', 'true', 'yes', 'on')
    DOWNLOAD_TOKEN_EXPIRY_HOURS = int(os.environ.get('DOWNLOAD_TOKEN_EXPIRY_HOURS', '24'))
