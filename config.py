import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'renalcare-secure-session-key-dsce-aiml-2026')
    DATABASE_PATH = os.environ.get('DATABASE_PATH') or os.path.join(os.path.dirname(__file__), 'renalcare.db')
    DATABASE_URL = os.environ.get('DATABASE_URL')
    DEBUG = os.environ.get('FLASK_DEBUG', '0') == '1'
    JSON_SORT_KEYS = False
