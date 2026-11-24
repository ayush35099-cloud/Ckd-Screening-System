import os
import json
import logging
import csv
import hmac
import secrets
import time
import threading
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, abort, Response, session
)
import io

from config import Config
from database import (
    init_db, save_screening, get_screening_by_uuid,
    get_screenings_history, get_dashboard_summary, get_connection,
    get_latest_email_delivery, increment_token_download
)
from ml.model import model_service
from report_generator import generate_pdf_report
from email_service import (
    initiate_screening_email_workflow,
    resend_screening_email,
    verify_download_token,
    format_delivery_status_summary,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_STORE = {}

def _is_rate_limited(bucket_key, max_calls=20, window_seconds=60):
    """Lightweight thread-safe sliding-window rate limiter compatible with single/free workers."""
    now = time.monotonic()
    with _RATE_LIMIT_LOCK:
        timestamps = [t for t in _RATE_LIMIT_STORE.get(bucket_key, []) if (now - t) < window_seconds]
        if len(timestamps) >= max_calls:
            _RATE_LIMIT_STORE[bucket_key] = timestamps
            return True
        timestamps.append(now)
        _RATE_LIMIT_STORE[bucket_key] = timestamps
        return False

def sanitize_csv_field(val):
    """
    Mitigates CSV formula injection by prepending a single quote
    if a string starts with standard spreadsheet formula trigger characters.
    """
    if val is None:
        return ""
    s = str(val).strip()
    if s and s[0] in ('=', '+', '-', '@', '\t', '\r'):
        return "'" + s
    return s

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    if os.environ.get('TESTING') == '1':
        app.config['TESTING'] = True

    # Initialize database tables & indexes safely
    with app.app_context():
        try:
            init_db()
        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    # Security Headers Middleware
    @app.after_request
    def set_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        return response

    def _get_or_create_csrf_token():
        token = session.get('csrf_token')
        if not token:
            token = secrets.token_urlsafe(32)
            session['csrf_token'] = token
        return token

    @app.context_processor
    def inject_security_helpers():
        return {'csrf_token': _get_or_create_csrf_token}

    def _validate_csrf():
        expected = session.get('csrf_token') or ''
        provided = (request.form.get('csrf_token') or request.headers.get('X-CSRF-Token') or '').strip()
        return bool(expected and provided and hmac.compare_digest(expected, provided))

    def _grant_clinician_session():
        session['clinician_authorized'] = True
        _get_or_create_csrf_token()

    def _is_clinician_authorized():
        if session.get('clinician_authorized') is True:
            return True
        configured_token = (os.environ.get('CLINICIAN_ACCESS_TOKEN') or '').strip()
        provided_token = (request.headers.get('X-Clinician-Token') or '').strip()
        if configured_token and provided_token and hmac.compare_digest(configured_token, provided_token):
            return True
        return False

    # -----------------------------------------------------------------------
    # Web Page Routes
    # -----------------------------------------------------------------------
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/dashboard')
    def dashboard_view():
        _grant_clinician_session()
        summary = get_dashboard_summary()
        metrics = model_service.get_metrics()
        return render_template('dashboard.html', summary=summary, metrics=metrics)

    @app.route('/screen', methods=['GET'])
    def screen_view():
        _grant_clinician_session()
        meta = model_service.get_feature_metadata()
        return render_template('screen.html', feature_meta=meta)

    def _is_csrf_enforced():
        if app.config.get('ENFORCE_CSRF'):
            return True
        is_testing = bool(app.config.get('TESTING') or os.environ.get('TESTING') == '1')
        return not is_testing

    @app.route('/screen', methods=['POST'])
    def screen_submit():
        if _is_csrf_enforced() and session.get('csrf_token') and not _validate_csrf():
            abort(403, description="Invalid or missing CSRF security token")

        _grant_clinician_session()
        form_data = request.form.to_dict()

        patient_id = form_data.get('patient_id', '').strip()
        if not patient_id:
            flash("Patient Identifier is required for clinical record keeping.", "danger")
            return redirect(url_for('screen_view'))

        model_choice = form_data.get('model_choice', 'Random Forest')

        # The 24 UCI dataset features strictly separated from patient administrative metadata
        param_keys = [
            'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
            'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
        ]
        input_params = {}
        for key in param_keys:
            val = form_data.get(key, '').strip()
            input_params[key] = val if val != '' else None

        # Execute machine learning model prediction
        try:
            pred_result = model_service.predict(input_params, model_choice=model_choice)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            flash(f"Screening evaluation error: {str(e)}", "danger")
            return redirect(url_for('screen_view'))

        # Prepare database record
        patient_record = {
            'patient_id': patient_id,
            'patient_name': form_data.get('patient_name', '').strip() or None,
            'age': form_data.get('age', '').strip() or None,
            'sex': form_data.get('sex', 'Not Specified'),
            'parameters': input_params
        }

        saved = save_screening(patient_record, pred_result)
        saved_record = get_screening_by_uuid(saved['screening_uuid'])

        # Generate PDF report BEFORE initiating automatic email delivery
        pdf_ready = False
        try:
            pdf_bytes = generate_pdf_report(saved_record)
            pdf_ready = bool(pdf_bytes and pdf_bytes.startswith(b'%PDF'))
        except Exception as pdf_err:
            logger.error(f"PDF pre-generation check failed for {saved['screening_uuid']}: {pdf_err}")

        patient_email = form_data.get('patient_email', '').strip()
        consent_raw = form_data.get('email_consent', '').strip().lower()
        email_consent = consent_raw in ('on', 'true', 'yes', '1')

        if pdf_ready:
            try:
                initiate_screening_email_workflow(
                    app=app,
                    screening_record=saved_record,
                    raw_email=patient_email,
                    consent_given=email_consent,
                    background=not app.config.get('TESTING'),
                )
            except Exception as email_err:
                logger.error(f"Non-blocking email workflow error for {saved['screening_uuid']}: {email_err}")

        return redirect(url_for('result_view', screening_uuid=saved['screening_uuid']))

    # -----------------------------------------------------------------------
    # Batch CSV Screening Routes (Feature 1 & Feature 2)
    # -----------------------------------------------------------------------
    @app.route('/screen/batch', methods=['GET'])
    def batch_screen_view():
        return render_template('batch_screen.html', summary=None, results=None, failed=None)

    @app.route('/screen/batch/template', methods=['GET'])
    def batch_template():
        output = io.StringIO()
        writer = csv.writer(output)

        headers = [
            'patient_id', 'patient_name', 'age', 'sex',
            'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
            'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
        ]
        writer.writerow(headers)

        # Representative baseline demo patient
        sample_row_1 = [
            'PT-CAMP-001', 'Sample Baseline Patient', '45', 'Male',
            '80', '1.020', '0', '0', 'normal', 'normal', 'notpresent', 'notpresent',
            '105', '30', '0.9', '138', '4.2', '14.5', '44', '7200', '5.0',
            'no', 'no', 'no', 'good', 'no', 'no'
        ]
        # Representative high-risk demo patient
        sample_row_2 = [
            'PT-CAMP-002', 'Sample High-Risk Patient', '62', 'Female',
            '95', '1.010', '3', '2', 'abnormal', 'abnormal', 'present', 'present',
            '220', '85', '3.1', '130', '5.5', '9.5', '28', '11500', '3.1',
            'yes', 'yes', 'yes', 'poor', 'yes', 'yes'
        ]
        writer.writerow(sample_row_1)
        writer.writerow(sample_row_2)

        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename="renalcare_batch_template.csv"'}
        )

    @app.route('/screen/batch', methods=['POST'])
    def batch_screen_submit():
        if 'file' not in request.files:
            flash("No file was uploaded. Please select a CSV file.", "danger")
            return redirect(url_for('batch_screen_view'))

        uploaded_file = request.files['file']
        if not uploaded_file or uploaded_file.filename == '':
            flash("No file selected. Please choose a valid .csv file.", "danger")
            return redirect(url_for('batch_screen_view'))

        if not uploaded_file.filename.lower().endswith('.csv'):
            flash("Invalid file format. Only CSV files (.csv) are supported.", "danger")
            return redirect(url_for('batch_screen_view'))

        model_choice = request.form.get('model_choice', 'Random Forest')

        try:
            content = uploaded_file.read()
            if len(content) > 2 * 1024 * 1024:
                flash("File size exceeds 2MB limit. Please upload a smaller file.", "danger")
                return redirect(url_for('batch_screen_view'))

            try:
                text_stream = io.StringIO(content.decode('utf-8'))
            except UnicodeDecodeError:
                text_stream = io.StringIO(content.decode('utf-8-sig', errors='replace'))

            reader = csv.DictReader(text_stream)
            if not reader.fieldnames:
                flash("CSV file is empty or missing headers.", "danger")
                return redirect(url_for('batch_screen_view'))

            # Normalize headers to lowercase stripped
            normalized_header_map = {col.strip().lower(): col for col in reader.fieldnames if col}

            REQUIRED_FEATURES = [
                'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
                'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
                'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
            ]

            missing_features = [f for f in REQUIRED_FEATURES if f not in normalized_header_map]
            if missing_features:
                flash(f"CSV file is missing required feature columns: {', '.join(missing_features)}. Please download the template.", "danger")
                return redirect(url_for('batch_screen_view'))

            raw_rows = list(reader)
            if len(raw_rows) == 0:
                flash("Uploaded CSV contains no patient records.", "danger")
                return redirect(url_for('batch_screen_view'))

            if len(raw_rows) > 100:
                flash(f"Batch limit exceeded: file contains {len(raw_rows)} rows (maximum allowed per batch is 100 rows).", "danger")
                return redirect(url_for('batch_screen_view'))

            successful_screenings = []
            failed_rows = []
            flagged_count = 0
            non_flagged_count = 0

            for idx, row in enumerate(raw_rows, start=1):
                try:
                    row_clean = {k.strip().lower(): (v.strip() if v is not None else '') for k, v in row.items() if k}

                    patient_id = row_clean.get('patient_id') or f"PT-BATCH-{idx}-{datetime.now().strftime('%M%S')}"
                    patient_name = row_clean.get('patient_name') or None
                    age_val = row_clean.get('age') or None
                    sex_val = row_clean.get('sex') or 'Not Specified'

                    input_params = {}
                    for feat in REQUIRED_FEATURES:
                        val = row_clean.get(feat, '')
                        input_params[feat] = val if val != '' else None

                    pred_result = model_service.predict(input_params, model_choice=model_choice)

                    patient_record = {
                        'patient_id': patient_id,
                        'patient_name': patient_name,
                        'age': age_val,
                        'sex': sex_val,
                        'parameters': input_params
                    }
                    saved = save_screening(patient_record, pred_result)

                    if pred_result['is_flagged']:
                        flagged_count += 1
                    else:
                        non_flagged_count += 1

                    successful_screenings.append({
                        'row': idx,
                        'screening_uuid': saved['screening_uuid'],
                        'patient_id': patient_id,
                        'patient_name': patient_name,
                        'age': age_val,
                        'sex': sex_val,
                        'prediction_label': pred_result['result_label'],
                        'prediction_code': pred_result['prediction_code'],
                        'probability_ckd': pred_result['probability_ckd'],
                        'risk_level': pred_result['risk_level'],
                        'model_used': pred_result['model_used']
                    })

                except Exception as row_err:
                    failed_rows.append({
                        'row': idx,
                        'patient_id': row.get('patient_id', f'Row {idx}'),
                        'error': str(row_err)
                    })

            # Save batch results in session for export
            session['last_batch_results'] = successful_screenings

            summary = {
                'total': len(raw_rows),
                'success': len(successful_screenings),
                'failed': len(failed_rows),
                'flagged': flagged_count,
                'non_flagged': non_flagged_count,
                'model_used': model_choice,
                'filename': uploaded_file.filename
            }

            return render_template(
                'batch_screen.html',
                summary=summary,
                results=successful_screenings,
                failed=failed_rows
            )

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            flash(f"Batch processing error: {str(e)}", "danger")
            return redirect(url_for('batch_screen_view'))

    @app.route('/screen/batch/export', methods=['GET'])
    def batch_export():
        batch_results = session.get('last_batch_results', [])
        if not batch_results:
            flash("No recent batch screening results available to export.", "warning")
            return redirect(url_for('batch_screen_view'))

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            'Row', 'Screening UUID', 'Patient ID', 'Patient Name', 'Age', 'Sex',
            'Screening Outcome', 'Prediction Code', 'Risk Tier', 'Probability %', 'Model Used'
        ])
        for r in batch_results:
            writer.writerow([
                sanitize_csv_field(r.get('row')),
                sanitize_csv_field(r.get('screening_uuid')),
                sanitize_csv_field(r.get('patient_id')),
                sanitize_csv_field(r.get('patient_name') or 'N/A'),
                sanitize_csv_field(r.get('age') or 'N/A'),
                sanitize_csv_field(r.get('sex') or 'N/A'),
                sanitize_csv_field(r.get('prediction_label')),
                sanitize_csv_field(r.get('prediction_code')),
                sanitize_csv_field(r.get('risk_level')),
                sanitize_csv_field(r.get('probability_ckd')),
                sanitize_csv_field(r.get('model_used'))
            ])

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"RenalCare_Batch_Results_{timestamp}.csv"

        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )

    @app.route('/result/<screening_uuid>')
    def result_view(screening_uuid):
        _grant_clinician_session()
        record = get_screening_by_uuid(screening_uuid)
        if not record:
            abort(404, description="Screening record not found")
        latest_delivery = get_latest_email_delivery(screening_uuid)
        email_status = format_delivery_status_summary(latest_delivery)
        return render_template('result.html', result=record, email_status=email_status)

    @app.route('/result/<screening_uuid>/resend-email', methods=['POST'])
    def resend_report_email(screening_uuid):
        if not _is_clinician_authorized():
            abort(403, description="Healthcare worker authorization is required to resend report links.")

        if _is_csrf_enforced() and not _validate_csrf():
            abort(403, description="Invalid or missing CSRF security token.")

        client_ip = request.remote_addr or 'unknown'
        if not app.config.get('TESTING') and _is_rate_limited(f"resend:{screening_uuid}:{client_ip}", max_calls=5, window_seconds=300):
            flash("Too many resend attempts. Please wait a few minutes before retrying.", "danger")
            return redirect(url_for('result_view', screening_uuid=screening_uuid)), 429

        record = get_screening_by_uuid(screening_uuid)
        if not record:
            abort(404, description="Screening record not found")

        # Verify PDF generates cleanly before issuing replacement link or sending email
        try:
            pdf_bytes = generate_pdf_report(record)
            if not pdf_bytes or not pdf_bytes.startswith(b'%PDF'):
                raise RuntimeError("Empty PDF buffer")
        except Exception as pdf_err:
            logger.error(f"PDF generation failed on resend for {screening_uuid}: {pdf_err}")
            flash("Unable to generate PDF report for email delivery.", "danger")
            return redirect(url_for('result_view', screening_uuid=screening_uuid))

        cooldown = 0 if app.config.get('TESTING') else 15
        ok, msg, _ = resend_screening_email(
            app=app,
            screening_record=record,
            cooldown_seconds=cooldown,
            background=not app.config.get('TESTING'),
        )
        flash(msg, "success" if ok else "danger")
        return redirect(url_for('result_view', screening_uuid=screening_uuid))

    @app.route('/history')
    def history_view():
        _grant_clinician_session()
        search_query = request.args.get('q', '').strip()
        filter_status = request.args.get('status', 'all')
        page = max(1, int(request.args.get('page', 1)))
        per_page = 50
        offset = (page - 1) * per_page

        records, total_count = get_screenings_history(
            search=search_query,
            filter_status=filter_status,
            limit=per_page,
            offset=offset
        )

        total_pages = max(1, (total_count + per_page - 1) // per_page)

        return render_template(
            'history.html',
            records=records,
            total_count=total_count,
            search_query=search_query,
            filter_status=filter_status,
            page=page,
            total_pages=total_pages
        )

    @app.route('/history/export', methods=['GET'])
    def history_export():
        search_query = request.args.get('q', '').strip()
        filter_status = request.args.get('status', 'all')

        records, _ = get_screenings_history(
            search=search_query,
            filter_status=filter_status,
            limit=10000,
            offset=0
        )

        output = io.StringIO()
        writer = csv.writer(output)

        headers = [
            'Screening UUID', 'Patient ID', 'Patient Name', 'Age', 'Sex',
            'Screening Outcome', 'Prediction Code', 'Risk Tier',
            'Confidence Percentage', 'Model Used', 'Screening Date'
        ]
        writer.writerow(headers)

        for r in records:
            writer.writerow([
                sanitize_csv_field(r.get('screening_uuid')),
                sanitize_csv_field(r.get('patient_id')),
                sanitize_csv_field(r.get('patient_name') or 'N/A'),
                sanitize_csv_field(r.get('age') or 'N/A'),
                sanitize_csv_field(r.get('sex') or 'N/A'),
                sanitize_csv_field(r.get('prediction_label')),
                sanitize_csv_field(r.get('prediction_code')),
                sanitize_csv_field(r.get('risk_level')),
                sanitize_csv_field(r.get('probability')),
                sanitize_csv_field(r.get('model_used')),
                sanitize_csv_field(r.get('formatted_date'))
            ])

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"RenalCare_Screenings_Export_{timestamp}.csv"

        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )

    @app.route('/research')
    def research_view():
        metrics = model_service.get_metrics()
        meta = model_service.get_feature_metadata()
        return render_template('research.html', metrics=metrics, metadata=meta)

    @app.route('/report/<screening_uuid>')
    def download_report(screening_uuid):
        record = get_screening_by_uuid(screening_uuid)
        if not record:
            abort(404, description="Screening record not found")

        pdf_bytes = generate_pdf_report(record)
        filename = f"RenalCare_Screening_{record['patient_id']}_{screening_uuid[:8]}.pdf"

        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )

    @app.route('/report/download/<token>')
    def download_report_by_token(token):
        client_ip = request.remote_addr or 'unknown'
        if not app.config.get('TESTING') and _is_rate_limited(f"token_dl:{client_ip}", max_calls=30, window_seconds=60):
            abort(429, description="Too many download requests. Please wait and try again.")

        is_valid, status_code, delivery, record = verify_download_token(token)
        if status_code == 'expired':
            return render_template('report_link_error.html', error_type='expired'), 410
        if not is_valid or not record:
            return render_template('report_link_error.html', error_type='invalid'), 404

        increment_token_download(delivery['id'])

        pdf_bytes = generate_pdf_report(record)
        filename = f"RenalCare_Screening_{record['screening_uuid'][:8]}.pdf"

        response = send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        return response

    @app.route('/api/email-status/<screening_uuid>')
    def api_email_status(screening_uuid):
        record = get_screening_by_uuid(screening_uuid)
        if not record:
            return jsonify({'error': 'Record not found'}), 404
        latest = get_latest_email_delivery(screening_uuid)
        return jsonify(format_delivery_status_summary(latest)), 200

    # -----------------------------------------------------------------------
    # REST API Endpoints
    # -----------------------------------------------------------------------
    @app.route('/api/health')
    def health_check():
        db_type = 'unknown'
        db_connected = False
        conn = None
        try:
            conn, db_type = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            db_connected = True
        except Exception as e:
            logger.error(f"Health check database query error: {e}")
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

        persistence_mode = 'persistent' if db_type == 'postgres' else 'ephemeral_sqlite'

        return jsonify({
            'status': 'healthy' if db_connected else 'degraded',
            'application': 'RenalCare CKD Screening System',
            'version': '1.1.0',
            'database_backend': db_type,
            'database_connected': db_connected,
            'database_persistence': persistence_mode,
            'models_loaded': {
                'random_forest': model_service.rf_pipeline is not None,
                'decision_tree': model_service.dt_pipeline is not None
            }
        }), 200

    @app.route('/api/dashboard')
    def api_dashboard():
        summary = get_dashboard_summary()
        return jsonify(summary), 200

    @app.route('/api/predict', methods=['POST'])
    def api_predict():
        data = request.get_json(silent=True) or {}
        parameters = data.get('parameters', data)
        model_choice = data.get('model', 'Random Forest')

        if not isinstance(parameters, dict):
            return jsonify({'error': 'Invalid request: parameters must be a dictionary'}), 400

        try:
            prediction = model_service.predict(parameters, model_choice=model_choice)
            
            saved_uuid = None
            if data.get('patient_id'):
                patient_record = {
                    'patient_id': str(data.get('patient_id')).strip(),
                    'patient_name': str(data.get('patient_name', '')).strip() or None,
                    'age': data.get('age'),
                    'sex': data.get('sex', 'Not Specified'),
                    'parameters': parameters
                }
                saved = save_screening(patient_record, prediction)
                saved_uuid = saved['screening_uuid']

            return jsonify({
                'success': True,
                'screening_uuid': saved_uuid,
                'prediction': prediction
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/history')
    def api_history():
        search_query = request.args.get('q', '').strip()
        filter_status = request.args.get('status', 'all')
        records, total_count = get_screenings_history(
            search=search_query,
            filter_status=filter_status,
            limit=100
        )
        return jsonify({
            'total_count': total_count,
            'records': records
        }), 200

    @app.route('/api/history/<screening_uuid>')
    def api_history_item(screening_uuid):
        record = get_screening_by_uuid(screening_uuid)
        if not record:
            return jsonify({'error': 'Record not found'}), 404
        return jsonify(record), 200

    @app.route('/api/report/<screening_uuid>')
    def api_report(screening_uuid):
        return download_report(screening_uuid)

    @app.route('/api/metrics')
    def api_metrics():
        metrics = model_service.get_metrics()
        return jsonify(metrics), 200

    # -----------------------------------------------------------------------
    # Error Handlers
    # -----------------------------------------------------------------------
    @app.errorhandler(404)
    def page_not_found(e):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Endpoint not found', 'status': 404}), 404
        return render_template('base.html', content="<div class='container-wide' style='padding: 4rem 1.5rem; text-align: center;'><h2>404 — Page Not Found</h2><p style='margin: 1rem 0;'>The requested clinical resource could not be found.</p><a href='/' class='btn-renal btn-renal-primary'>Return to Overview</a></div>"), 404

    @app.errorhandler(500)
    def internal_error(e):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Internal server error', 'status': 500}), 500
        return render_template('base.html', content="<div class='container-wide' style='padding: 4rem 1.5rem; text-align: center;'><h2>500 — Application Error</h2><p style='margin: 1rem 0;'>An unexpected error occurred while processing the request.</p><a href='/' class='btn-renal btn-renal-primary'>Return to Overview</a></div>"), 500

    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=Config.DEBUG)
