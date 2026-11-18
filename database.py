import os
import json
import sqlite3
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def get_database_path():
    """
    Dynamically resolves the SQLite database path at connection time.
    Prefers os.environ['DATABASE_PATH'] for clean test isolation.
    """
    return os.environ.get('DATABASE_PATH') or os.path.join(os.path.dirname(__file__), 'renalcare.db')

def get_database_url():
    """
    Dynamically resolves PostgreSQL DATABASE_URL at connection time.
    """
    return os.environ.get('DATABASE_URL')

# Backwards-compatible module-level references
DATABASE_PATH = get_database_path()
DATABASE_URL = get_database_url()

def get_connection(db_path=None, db_url=None):
    """
    Returns (connection, db_type).
    Tries PostgreSQL if DATABASE_URL is set and starts with postgres:// or postgresql://;
    falls back cleanly to local SQLite.
    Resolves configuration dynamically at call time for clean test isolation and runtime flexibility.
    """
    target_url = db_url if db_url is not None else get_database_url()
    if target_url and target_url.strip().startswith(('postgres://', 'postgresql://')):
        pg_url = target_url.strip()
        if pg_url.startswith('postgres://'):
            pg_url = pg_url.replace('postgres://', 'postgresql://', 1)
        try:
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(pg_url, connect_timeout=5)
            return conn, 'postgres'
        except Exception as e:
            host_info = pg_url.split('@')[-1] if '@' in pg_url else 'target host'
            logger.warning(f"Could not connect to PostgreSQL at {host_info} ({e}). Falling back to local SQLite.")

    # SQLite fallback
    target_path = db_path if db_path is not None else get_database_path()
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    return conn, 'sqlite'

def init_db(db_path=None, db_url=None):
    conn, db_type = get_connection(db_path=db_path, db_url=db_url)
    try:
        cursor = conn.cursor()

        if db_type == 'postgres':
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS screenings (
                    id SERIAL PRIMARY KEY,
                    screening_uuid VARCHAR(64) UNIQUE NOT NULL,
                    patient_id VARCHAR(64) NOT NULL,
                    patient_name VARCHAR(128),
                    age INTEGER,
                    sex VARCHAR(16),
                    parameters_json TEXT NOT NULL,
                    prediction_label VARCHAR(128) NOT NULL,
                    prediction_code VARCHAR(32) NOT NULL,
                    probability REAL NOT NULL,
                    risk_level VARCHAR(64) NOT NULL,
                    model_used VARCHAR(64) NOT NULL,
                    missing_params_json TEXT,
                    findings_json TEXT,
                    recommendation TEXT,
                    probability_explanation TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_uuid ON screenings(screening_uuid);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_patient ON screenings(patient_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_code ON screenings(prediction_code);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_created ON screenings(created_at);")
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS screenings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    screening_uuid TEXT UNIQUE NOT NULL,
                    patient_id TEXT NOT NULL,
                    patient_name TEXT,
                    age INTEGER,
                    sex TEXT,
                    parameters_json TEXT NOT NULL,
                    prediction_label TEXT NOT NULL,
                    prediction_code TEXT NOT NULL,
                    probability REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    missing_params_json TEXT,
                    findings_json TEXT,
                    recommendation TEXT,
                    probability_explanation TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Indexes for fast querying & security
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_uuid ON screenings(screening_uuid);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_patient ON screenings(patient_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_code ON screenings(prediction_code);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_created ON screenings(created_at);")

        # Ensure schema migrations for existing databases
        migration_columns = [
            ("missing_params_json", "TEXT"),
            ("findings_json", "TEXT"),
            ("recommendation", "TEXT"),
            ("probability_explanation", "TEXT"),
        ]
        for col_name, col_type in migration_columns:
            try:
                cursor.execute(f"ALTER TABLE screenings ADD COLUMN {col_name} {col_type};")
                conn.commit()
            except Exception:
                pass  # Column already exists

        conn.commit()
    finally:
        conn.close()

def save_screening(patient_data, prediction_result, db_path=None, db_url=None):
    screening_uuid = str(uuid.uuid4())
    conn, db_type = get_connection(db_path=db_path, db_url=db_url)
    try:
        cursor = conn.cursor()

        patient_id = patient_data.get('patient_id') or f"PT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        raw_name = patient_data.get('patient_name')
        patient_name = raw_name.strip() if (raw_name and isinstance(raw_name, str)) else None
        age = None
        try:
            if patient_data.get('age'):
                age = int(patient_data['age'])
        except (ValueError, TypeError):
            pass

        sex = patient_data.get('sex', 'Not Specified')
        params_json = json.dumps(patient_data.get('parameters', {}))
        missing_json = json.dumps(prediction_result.get('missing_parameters', []))
        findings_json = json.dumps(prediction_result.get('key_findings', []))
        prob_explanation = prediction_result.get('probability_explanation', '')

        if db_type == 'postgres':
            query = """
                INSERT INTO screenings (
                    screening_uuid, patient_id, patient_name, age, sex,
                    parameters_json, prediction_label, prediction_code,
                    probability, risk_level, model_used, missing_params_json,
                    findings_json, recommendation, probability_explanation
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """
            cursor.execute(query, (
                screening_uuid, patient_id, patient_name, age, sex,
                params_json, prediction_result['result_label'], prediction_result['prediction_code'],
                prediction_result['probability_ckd'], prediction_result['risk_level'],
                prediction_result['model_used'], missing_json, findings_json,
                prediction_result['recommendation'], prob_explanation
            ))
            row = cursor.fetchone()
            screening_id = row[0]
        else:
            query = """
                INSERT INTO screenings (
                    screening_uuid, patient_id, patient_name, age, sex,
                    parameters_json, prediction_label, prediction_code,
                    probability, risk_level, model_used, missing_params_json,
                    findings_json, recommendation, probability_explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            cursor.execute(query, (
                screening_uuid, patient_id, patient_name, age, sex,
                params_json, prediction_result['result_label'], prediction_result['prediction_code'],
                prediction_result['probability_ckd'], prediction_result['risk_level'],
                prediction_result['model_used'], missing_json, findings_json,
                prediction_result['recommendation'], prob_explanation
            ))
            screening_id = cursor.lastrowid

        conn.commit()
    finally:
        conn.close()

    return {
        'id': screening_id,
        'screening_uuid': screening_uuid,
        'patient_id': patient_id
    }

def get_screening_by_uuid(screening_uuid, db_path=None, db_url=None):
    conn, db_type = get_connection(db_path=db_path, db_url=db_url)
    try:
        cursor = conn.cursor()

        if db_type == 'postgres':
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("SELECT * FROM screenings WHERE screening_uuid = %s;", (screening_uuid,))
            row = cursor.fetchone()
            result = dict(row) if row else None
        else:
            cursor.execute("SELECT * FROM screenings WHERE screening_uuid = ?;", (screening_uuid,))
            row = cursor.fetchone()
            result = dict(row) if row else None
    finally:
        conn.close()

    if result:
        result['parameters'] = json.loads(result['parameters_json']) if result.get('parameters_json') else {}
        result['findings'] = json.loads(result['findings_json']) if result.get('findings_json') else []
        result['missing_parameters'] = json.loads(result['missing_params_json']) if result.get('missing_params_json') else []
        result['missing_count'] = len(result['missing_parameters'])
        if isinstance(result['created_at'], datetime):
            result['formatted_date'] = result['created_at'].strftime('%b %d, %Y, %I:%M %p')
        else:
            result['formatted_date'] = str(result['created_at'])
    return result

def get_screenings_history(search='', filter_status='all', limit=50, offset=0, db_path=None, db_url=None):
    conn, db_type = get_connection(db_path=db_path, db_url=db_url)
    try:
        cursor = conn.cursor()

        where_clauses = []
        params = []

        if search:
            search_pattern = f"%{search.strip()}%"
            if db_type == 'postgres':
                where_clauses.append("(patient_id ILIKE %s OR screening_uuid ILIKE %s)")
            else:
                where_clauses.append("(patient_id LIKE ? OR screening_uuid LIKE ?)")
            params.extend([search_pattern, search_pattern])

        if filter_status == 'flagged':
            where_clauses.append("prediction_code = %s" if db_type == 'postgres' else "prediction_code = ?")
            params.append('ckd')
        elif filter_status == 'non_flagged':
            where_clauses.append("prediction_code = %s" if db_type == 'postgres' else "prediction_code = ?")
            params.append('notckd')

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        # Count total
        count_sql = f"SELECT COUNT(*) FROM screenings{where_sql};"
        cursor.execute(count_sql, tuple(params))
        total_count = cursor.fetchone()[0]

        # Select records (excluding sensitive parameters from list views)
        if db_type == 'postgres':
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            select_sql = f"""
                SELECT id, screening_uuid, patient_id, patient_name, age, sex,
                       prediction_label, prediction_code, probability, risk_level,
                       model_used, created_at
                FROM screenings
                {where_sql}
                ORDER BY id DESC
                LIMIT %s OFFSET %s;
            """
            cursor.execute(select_sql, tuple(params + [limit, offset]))
            rows = [dict(r) for r in cursor.fetchall()]
        else:
            select_sql = f"""
                SELECT id, screening_uuid, patient_id, patient_name, age, sex,
                       prediction_label, prediction_code, probability, risk_level,
                       model_used, created_at
                FROM screenings
                {where_sql}
                ORDER BY id DESC
                LIMIT ? OFFSET ?;
            """
            cursor.execute(select_sql, tuple(params + [limit, offset]))
            rows = [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()

    for r in rows:
        if isinstance(r['created_at'], datetime):
            r['formatted_date'] = r['created_at'].strftime('%b %d, %Y, %I:%M %p')
        else:
            r['formatted_date'] = str(r['created_at'])

    return rows, total_count

def get_dashboard_summary(db_path=None, db_url=None):
    """
    Computes real operational statistics from the database.
    Separates actual patient screenings from model benchmark figures.
    """
    conn, db_type = get_connection(db_path=db_path, db_url=db_url)
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM screenings;")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM screenings WHERE prediction_code = 'ckd';")
        flagged = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM screenings WHERE prediction_code = 'notckd';")
        non_flagged = cursor.fetchone()[0]

        # Algorithm usage breakdown from database
        cursor.execute("SELECT model_used, COUNT(*) FROM screenings GROUP BY model_used;")
        algo_rows = cursor.fetchall()
        algorithm_usage = {}
        for row in algo_rows:
            algorithm_usage[row[0]] = int(row[1])

        # Screening activity trends (last 7 recorded days)
        if db_type == 'postgres':
            trend_sql = """
                SELECT TO_CHAR(created_at, 'YYYY-MM-DD') AS day, COUNT(*) AS count
                FROM screenings
                GROUP BY day
                ORDER BY day DESC
                LIMIT 7;
            """
        else:
            trend_sql = """
                SELECT SUBSTR(created_at, 1, 10) AS day, COUNT(*) AS count
                FROM screenings
                GROUP BY day
                ORDER BY day DESC
                LIMIT 7;
            """
        cursor.execute(trend_sql)
        trend_rows = cursor.fetchall()
        screening_trend = [{'date': r[0], 'count': int(r[1])} for r in reversed(trend_rows)]

        # Recent 5 screenings
        if db_type == 'postgres':
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("""
                SELECT id, screening_uuid, patient_id, patient_name, age, sex,
                       prediction_label, prediction_code, probability, risk_level,
                       model_used, created_at
                FROM screenings
                ORDER BY id DESC
                LIMIT 5;
            """)
            recent = [dict(r) for r in cursor.fetchall()]
        else:
            cursor.execute("""
                SELECT id, screening_uuid, patient_id, patient_name, age, sex,
                       prediction_label, prediction_code, probability, risk_level,
                       model_used, created_at
                FROM screenings
                ORDER BY id DESC
                LIMIT 5;
            """)
            recent = [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()

    for r in recent:
        if isinstance(r['created_at'], datetime):
            r['formatted_date'] = r['created_at'].strftime('%b %d, %Y, %I:%M %p')
        else:
            r['formatted_date'] = str(r['created_at'])

    return {
        'total': total,
        'flagged': flagged,
        'non_flagged': non_flagged,
        'flagged_percentage': round((flagged / total * 100), 1) if total > 0 else 0,
        'algorithm_usage': algorithm_usage,
        'screening_trend': screening_trend,
        'recent_screenings': recent
    }
