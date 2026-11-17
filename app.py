import os
import json
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, Response, abort
)
import io

from config import Config
from database import (
    init_db, save_screening, get_screening_by_uuid,
    get_screenings_history, get_dashboard_summary
)
from ml.model import model_service
from report_generator import generate_pdf_report

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize database
    with app.app_context():
        init_db()

    # -----------------------------------------------------------------------
    # Web Page Routes
    # -----------------------------------------------------------------------
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/dashboard')
    def dashboard_view():
        summary = get_dashboard_summary()
        return render_template('dashboard.html', summary=summary)

    @app.route('/screen', methods=['GET'])
    def screen_view():
        meta = model_service.get_feature_metadata()
        return render_template('screen.html', feature_meta=meta)

    @app.route('/screen', methods=['POST'])
    def screen_submit():
        form_data = request.form.to_dict()

        patient_id = form_data.get('patient_id', '').strip()
        if not patient_id:
            flash("Patient ID is required for screening record tracking.", "danger")
            return redirect(url_for('screen_view'))

        model_choice = form_data.get('model_choice', 'Random Forest')

        # Medical parameter dictionary
        param_keys = [
            'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
            'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
        ]
        input_params = {}
        for key in param_keys:
            val = form_data.get(key, '').strip()
            input_params[key] = val if val else None

        # Execute machine learning model prediction
        try:
            pred_result = model_service.predict(input_params, model_choice=model_choice)
        except Exception as e:
            app.logger.error(f"Prediction failed: {e}")
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
        return redirect(url_for('result_view', screening_uuid=saved['screening_uuid']))

    @app.route('/result/<screening_uuid>')
    def result_view(screening_uuid):
        record = get_screening_by_uuid(screening_uuid)
        if not record:
            abort(404, description="Screening record not found")
        return render_template('result.html', result=record)

    @app.route('/history')
    def history_view():
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

        return render_template(
            'history.html',
            records=records,
            total_count=total_count,
            search_query=search_query,
            filter_status=filter_status,
            page=page
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

    # -----------------------------------------------------------------------
    # REST API Endpoints
    # -----------------------------------------------------------------------
    @app.route('/api/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'application': 'RenalCare CKD Screening System',
            'version': '1.0.0',
            'champion_model': 'Random Forest Classifier',
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
            
            # Save if patient_id is provided
            saved_uuid = None
            if data.get('patient_id'):
                patient_record = {
                    'patient_id': data.get('patient_id'),
                    'patient_name': data.get('patient_name'),
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
