import os
import json
import sqlite3
import uuid
from datetime import datetime

DATABASE_PATH = os.environ.get('DATABASE_PATH') or os.path.join(os.path.dirname(__file__), 'renalcare.db')
DATABASE_URL = os.environ.get('DATABASE_URL')

def get_connection():
    # If DATABASE_URL is set and starts with postgres, use psycopg2
    if DATABASE_URL and DATABASE_URL.startswith(('postgres://', 'postgresql://')):
        import psycopg2
        import psycopg2.extras
        # Handle Render postgres URI schema if needed (postgres:// -> postgresql://)
        pg_url = DATABASE_URL
        if pg_url.startswith('postgres://'):
            pg_url = pg_url.replace('postgres://', 'postgresql://', 1)
        conn = psycopg2.connect(pg_url)
        return conn, 'postgres'
    else:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        return conn, 'sqlite'

def init_db():
    conn, db_type = get_connection()
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
                prediction_label VARCHAR(64) NOT NULL,
                prediction_code VARCHAR(32) NOT NULL,
                probability REAL NOT NULL,
                risk_level VARCHAR(32) NOT NULL,
                model_used VARCHAR(64) NOT NULL,
                findings_json TEXT,
                recommendation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
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
                findings_json TEXT,
                recommendation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Index on screening_uuid and patient_id for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_uuid ON screenings(screening_uuid);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_patient ON screenings(patient_id);")

    conn.commit()
    conn.close()

def save_screening(patient_data, prediction_result):
    screening_uuid = str(uuid.uuid4())
    conn, db_type = get_connection()
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
    findings_json = json.dumps(prediction_result.get('key_findings', []))

    if db_type == 'postgres':
        query = """
            INSERT INTO screenings (
                screening_uuid, patient_id, patient_name, age, sex,
                parameters_json, prediction_label, prediction_code,
                probability, risk_level, model_used, findings_json, recommendation
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """
        cursor.execute(query, (
            screening_uuid, patient_id, patient_name, age, sex,
            params_json, prediction_result['result_label'], prediction_result['prediction_code'],
            prediction_result['probability_ckd'], prediction_result['risk_level'],
            prediction_result['model_used'], findings_json, prediction_result['recommendation']
        ))
        row = cursor.fetchone()
        screening_id = row[0]
    else:
        query = """
            INSERT INTO screenings (
                screening_uuid, patient_id, patient_name, age, sex,
                parameters_json, prediction_label, prediction_code,
                probability, risk_level, model_used, findings_json, recommendation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        cursor.execute(query, (
            screening_uuid, patient_id, patient_name, age, sex,
            params_json, prediction_result['result_label'], prediction_result['prediction_code'],
            prediction_result['probability_ckd'], prediction_result['risk_level'],
            prediction_result['model_used'], findings_json, prediction_result['recommendation']
        ))
        screening_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return {
        'id': screening_id,
        'screening_uuid': screening_uuid,
        'patient_id': patient_id
    }

def get_screening_by_uuid(screening_uuid):
    conn, db_type = get_connection()
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

    conn.close()

    if result:
        result['parameters'] = json.loads(result['parameters_json']) if result.get('parameters_json') else {}
        result['findings'] = json.loads(result['findings_json']) if result.get('findings_json') else []
        if isinstance(result['created_at'], datetime):
            result['formatted_date'] = result['created_at'].strftime('%b %d, %Y, %I:%M %p')
        else:
            result['formatted_date'] = str(result['created_at'])
    return result

def get_screenings_history(search='', filter_status='all', limit=50, offset=0):
    conn, db_type = get_connection()
    cursor = conn.cursor()

    where_clauses = []
    params = []

    if search:
        search_pattern = f"%{search.strip()}%"
        if db_type == 'postgres':
            where_clauses.append("(patient_id ILIKE %s OR patient_name ILIKE %s OR screening_uuid ILIKE %s)")
        else:
            where_clauses.append("(patient_id LIKE ? OR patient_name LIKE ? OR screening_uuid LIKE ?)")
        params.extend([search_pattern, search_pattern, search_pattern])

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

    # Select records
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

    conn.close()

    for r in rows:
        if isinstance(r['created_at'], datetime):
            r['formatted_date'] = r['created_at'].strftime('%b %d, %Y, %I:%M %p')
        else:
            r['formatted_date'] = str(r['created_at'])

    return rows, total_count

def get_dashboard_summary():
    conn, db_type = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM screenings;")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM screenings WHERE prediction_code = 'ckd';")
    flagged = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM screenings WHERE prediction_code = 'notckd';")
    non_flagged = cursor.fetchone()[0]

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
        'recent_screenings': recent
    }
