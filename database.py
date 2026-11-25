import os
import json
import sqlite3
import uuid
import logging
from datetime import datetime, timezone

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
                    confidence REAL,
                    risk_level VARCHAR(64) NOT NULL,
                    model_used VARCHAR(64) NOT NULL,
                    missing_params_json TEXT,
                    findings_json TEXT,
                    recommendation TEXT,
                    probability_explanation TEXT,
                    comparison_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_uuid ON screenings(screening_uuid);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_patient ON screenings(patient_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_code ON screenings(prediction_code);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_created ON screenings(created_at);")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS email_deliveries (
                    id SERIAL PRIMARY KEY,
                    screening_uuid VARCHAR(64) NOT NULL,
                    patient_id VARCHAR(64),
                    recipient_email VARCHAR(254),
                    consent_recorded BOOLEAN NOT NULL DEFAULT FALSE,
                    delivery_status VARCHAR(32) NOT NULL,
                    provider_name VARCHAR(32) DEFAULT 'resend',
                    provider_message_id VARCHAR(128),
                    idempotency_key VARCHAR(128),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sent_at TIMESTAMP,
                    last_attempt_at TIMESTAMP,
                    error_category VARCHAR(64),
                    error_detail TEXT,
                    token_hash VARCHAR(128),
                    token_expiry TIMESTAMP,
                    download_count INTEGER NOT NULL DEFAULT 0,
                    last_downloaded_at TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_email_deliveries_screening ON email_deliveries(screening_uuid);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_email_deliveries_token_hash ON email_deliveries(token_hash);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_email_deliveries_status ON email_deliveries(delivery_status);")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patient_profiles (
                    id SERIAL PRIMARY KEY,
                    patient_id VARCHAR(64) UNIQUE NOT NULL,
                    patient_name VARCHAR(128),
                    email VARCHAR(254),
                    email_verified BOOLEAN DEFAULT FALSE,
                    consent_recorded BOOLEAN DEFAULT FALSE,
                    consent_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patient_profiles_id ON patient_profiles(patient_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patient_profiles_email ON patient_profiles(email);")
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
                    confidence REAL,
                    risk_level TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    missing_params_json TEXT,
                    findings_json TEXT,
                    recommendation TEXT,
                    probability_explanation TEXT,
                    comparison_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Indexes for fast querying & security
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_uuid ON screenings(screening_uuid);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_patient ON screenings(patient_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_code ON screenings(prediction_code);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenings_created ON screenings(created_at);")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS email_deliveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    screening_uuid TEXT NOT NULL,
                    patient_id TEXT,
                    recipient_email TEXT,
                    consent_recorded INTEGER NOT NULL DEFAULT 0,
                    delivery_status TEXT NOT NULL,
                    provider_name TEXT DEFAULT 'resend',
                    provider_message_id TEXT,
                    idempotency_key TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sent_at TIMESTAMP,
                    last_attempt_at TIMESTAMP,
                    error_category TEXT,
                    error_detail TEXT,
                    token_hash TEXT,
                    token_expiry TIMESTAMP,
                    download_count INTEGER NOT NULL DEFAULT 0,
                    last_downloaded_at TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_email_deliveries_screening ON email_deliveries(screening_uuid);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_email_deliveries_token_hash ON email_deliveries(token_hash);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_email_deliveries_status ON email_deliveries(delivery_status);")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patient_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT UNIQUE NOT NULL,
                    patient_name TEXT,
                    email TEXT,
                    email_verified INTEGER DEFAULT 0,
                    consent_recorded INTEGER DEFAULT 0,
                    consent_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patient_profiles_id ON patient_profiles(patient_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patient_profiles_email ON patient_profiles(email);")

        # Ensure schema migrations for existing databases
        migration_columns = [
            ("missing_params_json", "TEXT"),
            ("findings_json", "TEXT"),
            ("recommendation", "TEXT"),
            ("probability_explanation", "TEXT"),
            ("confidence", "REAL"),
            ("comparison_json", "TEXT"),
        ]
        for col_name, col_type in migration_columns:
            try:
                cursor.execute(f"ALTER TABLE screenings ADD COLUMN {col_name} {col_type};")
                conn.commit()
            except Exception:
                pass  # Column already exists

        # Ensure patient_id migration in email_deliveries
        try:
            pid_col_type = "VARCHAR(64)" if db_type == 'postgres' else "TEXT"
            cursor.execute(f"ALTER TABLE email_deliveries ADD COLUMN patient_id {pid_col_type};")
            conn.commit()
        except Exception:
            pass
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_email_deliveries_patient ON email_deliveries(patient_id);")
            conn.commit()
        except Exception:
            pass

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
        comparison_json = json.dumps(prediction_result.get('comparison', {}))

        confidence_val = prediction_result.get('confidence')
        if confidence_val is None:
            prob_ckd = float(prediction_result.get('probability_ckd', prediction_result.get('probability', 0.0)))
            is_flagged = bool(prediction_result.get('is_flagged', prediction_result.get('prediction_code') == 'ckd'))
            confidence_val = prob_ckd if is_flagged else round(100.0 - prob_ckd, 1)

        if db_type == 'postgres':
            query = """
                INSERT INTO screenings (
                    screening_uuid, patient_id, patient_name, age, sex,
                    parameters_json, prediction_label, prediction_code,
                    probability, confidence, risk_level, model_used, missing_params_json,
                    findings_json, recommendation, probability_explanation, comparison_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """
            cursor.execute(query, (
                screening_uuid, patient_id, patient_name, age, sex,
                params_json, prediction_result['result_label'], prediction_result['prediction_code'],
                prediction_result['probability_ckd'], confidence_val, prediction_result['risk_level'],
                prediction_result['model_used'], missing_json, findings_json,
                prediction_result['recommendation'], prob_explanation, comparison_json
            ))
            row = cursor.fetchone()
            screening_id = row[0]
        else:
            query = """
                INSERT INTO screenings (
                    screening_uuid, patient_id, patient_name, age, sex,
                    parameters_json, prediction_label, prediction_code,
                    probability, confidence, risk_level, model_used, missing_params_json,
                    findings_json, recommendation, probability_explanation, comparison_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            cursor.execute(query, (
                screening_uuid, patient_id, patient_name, age, sex,
                params_json, prediction_result['result_label'], prediction_result['prediction_code'],
                prediction_result['probability_ckd'], confidence_val, prediction_result['risk_level'],
                prediction_result['model_used'], missing_json, findings_json,
                prediction_result['recommendation'], prob_explanation, comparison_json
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
        result['comparison'] = json.loads(result['comparison_json']) if result.get('comparison_json') else {}

        prob_val = float(result.get('probability', 0.0))
        result['probability_ckd'] = round(prob_val, 1)
        result['probability_notckd'] = round(100.0 - prob_val, 1)

        if result.get('confidence') is not None:
            result['confidence'] = round(float(result['confidence']), 1)
        else:
            if result.get('prediction_code') == 'ckd':
                result['confidence'] = round(prob_val, 1)
            else:
                result['confidence'] = round(100.0 - prob_val, 1)

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
        algorithm_usage = {'Random Forest': 0, 'Decision Tree': 0}
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

        # Recent 5 screenings with confidence and probability
        query_recent = """
            SELECT id, screening_uuid, patient_id, patient_name, age, sex,
                   prediction_label, prediction_code, probability, confidence, risk_level,
                   model_used, created_at
            FROM screenings
            ORDER BY id DESC
            LIMIT 5;
        """
        if db_type == 'postgres':
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute(query_recent)
            recent = [dict(r) for r in cursor.fetchall()]
        else:
            cursor.execute(query_recent)
            recent = [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()

    for r in recent:
        prob = float(r.get('probability', 0.0))
        r['probability_ckd'] = round(prob, 1)
        if r.get('confidence') is not None:
            r['confidence'] = round(float(r['confidence']), 1)
        else:
            r['confidence'] = round(prob if r.get('prediction_code') == 'ckd' else (100.0 - prob), 1)

        if isinstance(r['created_at'], datetime):
            r['formatted_date'] = r['created_at'].strftime('%b %d, %Y, %I:%M %p')
        else:
            r['formatted_date'] = str(r['created_at'])

    # Fetch serialized ML metrics for comparative dashboard benchmark display
    model_metrics = None
    try:
        from ml.model import model_service
        model_metrics = model_service.get_metrics()
    except Exception as e:
        logger.warning(f"Could not load ML metrics in dashboard summary: {e}")

    return {
        'total': total,
        'flagged': flagged,
        'non_flagged': non_flagged,
        'flagged_percentage': round((flagged / total * 100), 1) if total > 0 else 0,
        'algorithm_usage': algorithm_usage,
        'model_metrics': model_metrics,
        'screening_trend': screening_trend,
        'recent_screenings': recent
    }

# ---------------------------------------------------------------------------
# Email Delivery & Secure Token Persistence Helpers
# ---------------------------------------------------------------------------
def _to_db_timestamp(dt_val, db_type):
    if dt_val is None:
        return None
    if isinstance(dt_val, str):
        return dt_val
    if dt_val.tzinfo is not None:
        dt_val = dt_val.astimezone(timezone.utc).replace(tzinfo=None)
    return dt_val if db_type == 'postgres' else dt_val.strftime('%Y-%m-%d %H:%M:%S')

def _parse_utc_timestamp(val):
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
    s = str(val).strip()
    if s.endswith('Z'):
        s = s[:-1]
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S.%f'):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None

def _normalize_delivery_row(row_dict):
    if not row_dict:
        return None
    d = dict(row_dict)
    d['consent_recorded'] = bool(d.get('consent_recorded'))
    for ts_field in ('created_at', 'sent_at', 'last_attempt_at', 'token_expiry', 'last_downloaded_at'):
        parsed = _parse_utc_timestamp(d.get(ts_field))
        if parsed is not None:
            d[f'{ts_field}_dt'] = parsed
            d[ts_field] = parsed.strftime('%Y-%m-%d %H:%M:%S UTC')
        else:
            d[f'{ts_field}_dt'] = None
    return d

def create_email_delivery(
    screening_uuid,
    recipient_email=None,
    consent_recorded=False,
    delivery_status='not_provided',
    provider_name='resend',
    idempotency_key=None,
    token_hash=None,
    token_expiry=None,
    error_category=None,
    error_detail=None,
    patient_id=None,
    db_path=None,
    db_url=None
):
    conn, db_type = get_connection(db_path=db_path, db_url=db_url)
    try:
        cursor = conn.cursor()
        consent_db = bool(consent_recorded) if db_type == 'postgres' else (1 if consent_recorded else 0)
        expiry_db = _to_db_timestamp(token_expiry, db_type)

        if db_type == 'postgres':
            cursor.execute("""
                INSERT INTO email_deliveries (
                    screening_uuid, recipient_email, consent_recorded, delivery_status,
                    provider_name, idempotency_key, attempts, error_category,
                    error_detail, token_hash, token_expiry, patient_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                screening_uuid, recipient_email, consent_db, delivery_status,
                provider_name, idempotency_key, 0, error_category,
                error_detail, token_hash, expiry_db, patient_id
            ))
            delivery_id = cursor.fetchone()[0]
        else:
            cursor.execute("""
                INSERT INTO email_deliveries (
                    screening_uuid, recipient_email, consent_recorded, delivery_status,
                    provider_name, idempotency_key, attempts, error_category,
                    error_detail, token_hash, token_expiry, patient_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                screening_uuid, recipient_email, consent_db, delivery_status,
                provider_name, idempotency_key, 0, error_category,
                error_detail, token_hash, expiry_db, patient_id
            ))
            delivery_id = cursor.lastrowid

        conn.commit()
        return delivery_id
    finally:
        conn.close()

def invalidate_old_tokens_for_screening(screening_uuid, exclude_delivery_id=None, db_path=None, db_url=None):
    """Clears older token hashes for a screening when a replacement link is issued."""
    conn, db_type = get_connection(db_path=db_path, db_url=db_url)
    try:
        cursor = conn.cursor()
        ph = "%s" if db_type == 'postgres' else "?"
        if exclude_delivery_id is not None:
            cursor.execute(
                f"UPDATE email_deliveries SET token_hash = NULL WHERE screening_uuid = {ph} AND id != {ph};",
                (screening_uuid, exclude_delivery_id)
            )
        else:
            cursor.execute(
                f"UPDATE email_deliveries SET token_hash = NULL WHERE screening_uuid = {ph};",
                (screening_uuid,)
            )
        conn.commit()
    finally:
        conn.close()

def update_email_delivery_status(
    delivery_id,
    delivery_status,
    provider_message_id=None,
    attempts=None,
    sent_at=None,
    last_attempt_at=None,
    error_category=None,
    error_detail=None,
    db_path=None,
    db_url=None
):
    conn, db_type = get_connection(db_path=db_path, db_url=db_url)
    try:
        cursor = conn.cursor()
        ph = "%s" if db_type == 'postgres' else "?"
        fields = [f"delivery_status = {ph}"]
        values = [delivery_status]

        if provider_message_id is not None:
            fields.append(f"provider_message_id = {ph}")
            values.append(provider_message_id)
        if attempts is not None:
            fields.append(f"attempts = {ph}")
            values.append(int(attempts))
        if sent_at is not None:
            fields.append(f"sent_at = {ph}")
            values.append(_to_db_timestamp(sent_at, db_type))
        if last_attempt_at is not None:
            fields.append(f"last_attempt_at = {ph}")
            values.append(_to_db_timestamp(last_attempt_at, db_type))
        if error_category is not None:
            fields.append(f"error_category = {ph}")
            values.append(error_category)
        if error_detail is not None:
            fields.append(f"error_detail = {ph}")
            values.append(error_detail)

        values.append(delivery_id)
        cursor.execute(f"UPDATE email_deliveries SET {', '.join(fields)} WHERE id = {ph};", tuple(values))
        conn.commit()
    finally:
        conn.close()

def get_latest_email_delivery(screening_uuid, db_path=None, db_url=None):
    conn, db_type = get_connection(db_path=db_path, db_url=db_url)
    try:
        if db_type == 'postgres':
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute(
                "SELECT * FROM email_deliveries WHERE screening_uuid = %s ORDER BY id DESC LIMIT 1;",
                (screening_uuid,)
            )
            row = cursor.fetchone()
        else:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM email_deliveries WHERE screening_uuid = ? ORDER BY id DESC LIMIT 1;",
                (screening_uuid,)
            )
            row = cursor.fetchone()
        return _normalize_delivery_row(row) if row else None
    finally:
        conn.close()

def get_email_delivery_by_token_hash(token_hash, db_path=None, db_url=None):
    if not token_hash:
        return None
    conn, db_type = get_connection(db_path=db_path, db_url=db_url)
    try:
        if db_type == 'postgres':
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute(
                "SELECT * FROM email_deliveries WHERE token_hash = %s ORDER BY id DESC LIMIT 1;",
                (token_hash,)
            )
            row = cursor.fetchone()
        else:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM email_deliveries WHERE token_hash = ? ORDER BY id DESC LIMIT 1;",
                (token_hash,)
            )
            row = cursor.fetchone()
        return _normalize_delivery_row(row) if row else None
    finally:
        conn.close()

def increment_token_download(delivery_id, db_path=None, db_url=None):
    conn, db_type = get_connection(db_path=db_path, db_url=db_url)
    try:
        cursor = conn.cursor()
        ph = "%s" if db_type == 'postgres' else "?"
        now_db = _to_db_timestamp(datetime.now(timezone.utc), db_type)
        cursor.execute(
            f"UPDATE email_deliveries SET download_count = download_count + 1, last_downloaded_at = {ph} WHERE id = {ph};",
            (now_db, delivery_id)
        )
        conn.commit()
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Patient Profile Persistence (Phase 2)
# ---------------------------------------------------------------------------

def _normalize_patient_profile_row(row_dict):
    if not row_dict:
        return None
    d = dict(row_dict)
    d['email_verified'] = bool(d.get('email_verified'))
    d['consent_recorded'] = bool(d.get('consent_recorded'))
    for ts_field in ('consent_at', 'created_at', 'updated_at'):
        parsed = _parse_utc_timestamp(d.get(ts_field))
        if parsed is not None:
            d[f'{ts_field}_dt'] = parsed
            d[ts_field] = parsed.strftime('%Y-%m-%d %H:%M:%S UTC')
        else:
            d[f'{ts_field}_dt'] = None
    return d

def get_patient_profile(patient_id, db_path=None, db_url=None):
    """Retrieves a persistent patient profile by patient_id."""
    if not patient_id or not str(patient_id).strip():
        return None
    pid = str(patient_id).strip()
    conn, db_type = get_connection(db_path=db_path, db_url=db_url)
    try:
        ph = "%s" if db_type == 'postgres' else "?"
        if db_type == 'postgres':
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        else:
            cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM patient_profiles WHERE patient_id = {ph} LIMIT 1;", (pid,))
        row = cursor.fetchone()
        return _normalize_patient_profile_row(row) if row else None
    finally:
        conn.close()

def upsert_patient_profile(
    patient_id,
    patient_name=None,
    email=None,
    consent_recorded=None,
    email_verified=None,
    db_path=None,
    db_url=None
):
    """
    Inserts or updates a persistent patient profile.
    Guarantees that patient_id remains unique across all records.
    """
    if not patient_id or not str(patient_id).strip():
        raise ValueError("patient_id is required for patient profile records")

    pid = str(patient_id).strip()
    clean_name = str(patient_name).strip() if patient_name is not None and str(patient_name).strip() else None
    clean_email = str(email).strip().lower() if email is not None and str(email).strip() else None

    conn, db_type = get_connection(db_path=db_path, db_url=db_url)
    try:
        cursor = conn.cursor()
        ph = "%s" if db_type == 'postgres' else "?"
        now_dt = datetime.now(timezone.utc)
        now_db = _to_db_timestamp(now_dt, db_type)

        cursor.execute(f"SELECT id, patient_name, email, email_verified, consent_recorded, consent_at FROM patient_profiles WHERE patient_id = {ph} LIMIT 1;", (pid,))
        existing = cursor.fetchone()

        if existing:
            fields = []
            vals = []
            if patient_name is not None:
                fields.append(f"patient_name = {ph}")
                vals.append(clean_name)
            if email is not None:
                fields.append(f"email = {ph}")
                vals.append(clean_email)
            if consent_recorded is not None:
                c_val = bool(consent_recorded) if db_type == 'postgres' else (1 if consent_recorded else 0)
                fields.append(f"consent_recorded = {ph}")
                vals.append(c_val)
                if consent_recorded:
                    fields.append(f"consent_at = {ph}")
                    vals.append(now_db)
            if email_verified is not None:
                v_val = bool(email_verified) if db_type == 'postgres' else (1 if email_verified else 0)
                fields.append(f"email_verified = {ph}")
                vals.append(v_val)

            fields.append(f"updated_at = {ph}")
            vals.append(now_db)

            vals.append(pid)
            cursor.execute(f"UPDATE patient_profiles SET {', '.join(fields)} WHERE patient_id = {ph};", tuple(vals))
        else:
            c_val = bool(consent_recorded) if db_type == 'postgres' else (1 if consent_recorded else 0)
            v_val = bool(email_verified) if db_type == 'postgres' else (1 if email_verified else 0)
            c_at = now_db if consent_recorded else None

            if db_type == 'postgres':
                cursor.execute("""
                    INSERT INTO patient_profiles (
                        patient_id, patient_name, email, email_verified,
                        consent_recorded, consent_at, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                """, (pid, clean_name, clean_email, v_val, c_val, c_at, now_db, now_db))
            else:
                cursor.execute("""
                    INSERT INTO patient_profiles (
                        patient_id, patient_name, email, email_verified,
                        consent_recorded, consent_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, (pid, clean_name, clean_email, v_val, c_val, c_at, now_db, now_db))

        conn.commit()
    finally:
        conn.close()

    return get_patient_profile(pid, db_path=db_path, db_url=db_url)

def get_patient_profiles(limit=50, offset=0, search=None, db_path=None, db_url=None):
    """Lists patient profiles with optional search and pagination."""
    conn, db_type = get_connection(db_path=db_path, db_url=db_url)
    try:
        limit_val = max(1, min(int(limit), 200))
        offset_val = max(0, int(offset))
        ph = "%s" if db_type == 'postgres' else "?"

        where_clause = ""
        params = []
        if search and str(search).strip():
            s = f"%{str(search).strip()}%"
            where_clause = f"WHERE patient_id LIKE {ph} OR patient_name LIKE {ph} OR email LIKE {ph}"
            params = [s, s, s]

        count_query = f"SELECT COUNT(*) FROM patient_profiles {where_clause};"
        cursor = conn.cursor()
        cursor.execute(count_query, tuple(params))
        total_count = cursor.fetchone()[0]

        select_query = f"SELECT * FROM patient_profiles {where_clause} ORDER BY updated_at DESC LIMIT {ph} OFFSET {ph};"
        select_params = list(params) + [limit_val, offset_val]

        if db_type == 'postgres':
            import psycopg2.extras
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        else:
            cur = conn.cursor()
        cur.execute(select_query, tuple(select_params))
        rows = cur.fetchall()

        return [_normalize_patient_profile_row(r) for r in rows], total_count
    finally:
        conn.close()

def get_email_diagnostics(limit=50, db_path=None, db_url=None):
    """
    Returns sanitized summary of recent email deliveries for authorized clinician troubleshooting.
    Masks recipient emails, omits secret tokens, and details failure categories.
    """
    conn, db_type = get_connection(db_path=db_path, db_url=db_url)
    try:
        limit_val = max(1, min(int(limit), 200))
        ph = "%s" if db_type == 'postgres' else "?"
        if db_type == 'postgres':
            import psycopg2.extras
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        else:
            cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM email_deliveries ORDER BY id DESC LIMIT {ph};", (limit_val,))
        rows = cursor.fetchall()

        results = []
        for r in rows:
            norm = _normalize_delivery_row(r)
            raw_rec = norm.get('recipient_email') or ''
            if raw_rec and '@' in raw_rec:
                l, d = raw_rec.split('@', 1)
                masked = (l[0] + '***' + l[-1] if len(l) > 2 else '*@' + d) + '@' + d
            else:
                masked = 'Not provided'

            results.append({
                'id': norm.get('id'),
                'screening_uuid': norm.get('screening_uuid'),
                'patient_id': norm.get('patient_id'),
                'masked_recipient': masked,
                'delivery_status': norm.get('delivery_status'),
                'provider_name': norm.get('provider_name'),
                'provider_message_id': norm.get('provider_message_id'),
                'attempts': norm.get('attempts', 0),
                'error_category': norm.get('error_category'),
                'error_detail': norm.get('error_detail'),
                'created_at': norm.get('created_at'),
                'sent_at': norm.get('sent_at'),
                'last_attempt_at': norm.get('last_attempt_at'),
                'token_expiry': norm.get('token_expiry'),
                'download_count': norm.get('download_count', 0),
            })
        return results
    finally:
        conn.close()
