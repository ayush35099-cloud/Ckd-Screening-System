import os
import json
import joblib
import pandas as pd
import numpy as np

class CKDModelService:
    _instance = None

    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.rf_pipeline_path = os.path.join(self.models_dir, 'ckd_pipeline.joblib')
        self.dt_pipeline_path = os.path.join(self.models_dir, 'ckd_dt_pipeline.joblib')
        self.metrics_path = os.path.join(self.models_dir, 'metrics.json')
        self.meta_path = os.path.join(self.models_dir, 'feature_meta.json')

        self.rf_pipeline = None
        self.dt_pipeline = None
        self.metrics = None
        self.metadata = None

        self._load_artifacts()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_artifacts(self):
        if os.path.exists(self.rf_pipeline_path):
            self.rf_pipeline = joblib.load(self.rf_pipeline_path)
        else:
            raise FileNotFoundError(f"Pipeline not found at {self.rf_pipeline_path}. Run ml/train.py first.")

        if os.path.exists(self.dt_pipeline_path):
            self.dt_pipeline = joblib.load(self.dt_pipeline_path)

        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r', encoding='utf-8') as f:
                self.metrics = json.load(f)

        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

    def get_feature_metadata(self):
        return self.metadata

    def get_metrics(self):
        return self.metrics

    def predict(self, input_dict, model_choice='Random Forest'):
        """
        Takes raw dictionary of patient parameters.
        Returns:
            dict containing prediction ('ckd' or 'notckd'),
            confidence / probability,
            risk_level ('Low', 'Moderate', 'High'),
            model_used,
            key_findings list,
            recommendation string.
        """
        if model_choice.lower() in ['decision tree', 'decisiontree', 'dt'] and self.dt_pipeline is not None:
            pipeline = self.dt_pipeline
            model_name = 'Decision Tree'
        else:
            pipeline = self.rf_pipeline
            model_name = 'Random Forest'

        features_def = self.metadata['features']
        row_data = {}

        # Parse and sanitize each feature according to schema
        for feat_name, meta in features_def.items():
            val = input_dict.get(feat_name, None)
            if val is None or val == '' or val == '?':
                row_data[feat_name] = np.nan
            else:
                if meta['type'] == 'numerical':
                    try:
                        row_data[feat_name] = float(val)
                    except (ValueError, TypeError):
                        row_data[feat_name] = np.nan
                else:
                    # Categorical: lowercase stripped string
                    val_str = str(val).strip().lower()
                    if val_str in meta['options']:
                        row_data[feat_name] = val_str
                    else:
                        row_data[feat_name] = np.nan

        # Convert to single-row DataFrame with expected columns
        input_df = pd.DataFrame([row_data])

        # Prediction and probabilities
        pred_label_idx = pipeline.predict(input_df)[0] # 1 = ckd, 0 = notckd
        probabilities = pipeline.predict_proba(input_df)[0] # [prob_notckd, prob_ckd]
        prob_ckd = float(probabilities[1])
        prob_notckd = float(probabilities[0])

        is_flagged = bool(pred_label_idx == 1)
        result_label = "CKD Indicators Detected" if is_flagged else "CKD Indicators Not Detected"

        # Risk stratification based on model probability
        if prob_ckd >= 0.70:
            risk_level = "High"
            risk_badge = "risk-high"
        elif prob_ckd >= 0.35:
            risk_level = "Moderate"
            risk_badge = "risk-moderate"
        else:
            risk_level = "Low"
            risk_badge = "risk-low"

        # Analyze key contributing abnormal flags from entered laboratory parameters
        key_findings = []
        sc_val = row_data.get('sc', np.nan)
        if not np.isnan(sc_val) and sc_val > 1.2:
            key_findings.append({
                'parameter': 'Serum Creatinine',
                'value': f"{sc_val} mg/dL",
                'status': 'Elevated',
                'detail': 'Elevated creatinine suggests decreased glomerular filtration rate (GFR).'
            })

        hemo_val = row_data.get('hemo', np.nan)
        if not np.isnan(hemo_val) and hemo_val < 12.0:
            key_findings.append({
                'parameter': 'Hemoglobin',
                'value': f"{hemo_val} g/dL",
                'status': 'Low (Anemia)',
                'detail': 'Decreased hemoglobin is a frequent complication of impaired renal erythropoietin production.'
            })

        al_val = row_data.get('al', np.nan)
        if not np.isnan(al_val) and al_val > 0:
            key_findings.append({
                'parameter': 'Urine Albumin (Protein)',
                'value': f"Grade {int(al_val)}",
                'status': 'Proteinuria Detected',
                'detail': 'Significant protein leakage in urine signals glomerular barrier damage.'
            })

        sg_val = row_data.get('sg', np.nan)
        if not np.isnan(sg_val) and sg_val < 1.015:
            key_findings.append({
                'parameter': 'Specific Gravity',
                'value': f"{sg_val:.3f}",
                'status': 'Low Urine Osmolality',
                'detail': 'Indicates impaired renal tubular concentrating capacity.'
            })

        if str(row_data.get('htn', 'no')) == 'yes':
            key_findings.append({
                'parameter': 'Hypertension',
                'value': 'Present',
                'status': 'Vascular Comorbidity',
                'detail': 'Sustained systemic blood pressure elevates glomerular pressure, accelerating nephron sclerosis.'
            })

        if str(row_data.get('dm', 'no')) == 'yes':
            key_findings.append({
                'parameter': 'Diabetes Mellitus',
                'value': 'Present',
                'status': 'Metabolic Comorbidity',
                'detail': 'Diabetic microvascular disease is the leading global precursor to end-stage renal disease.'
            })

        bu_val = row_data.get('bu', np.nan)
        if not np.isnan(bu_val) and bu_val > 45.0:
            key_findings.append({
                'parameter': 'Blood Urea',
                'value': f"{bu_val} mg/dL",
                'status': 'Elevated',
                'detail': 'Retention of nitrogenous metabolic waste in circulating plasma.'
            })

        if str(row_data.get('pe', 'no')) == 'yes':
            key_findings.append({
                'parameter': 'Pedal Edema',
                'value': 'Present',
                'status': 'Fluid Retention',
                'detail': 'Dependent peripheral edema indicates fluid overload or hypoalbuminemia.'
            })

        # Clinical Guidance / Action Recommendation
        if is_flagged:
            recommendation = (
                "The machine learning screening pipeline has detected markers indicative of Chronic Kidney Disease. "
                "Urgent clinical correlation is advised: arrange formal laboratory evaluation including eGFR, "
                "24-hour urinary protein quantification, serum electrolytes, and renal ultrasonography with a qualified healthcare professional. "
                "Note: This tool is an automated educational screening and decision-support aid, not a clinical diagnosis."
            )
        else:
            recommendation = (
                "Routine screening indicators show values within non-CKD baseline ranges. "
                "Routine health maintenance and annual wellness monitoring are advised. If the patient has persistent symptoms "
                "or risk factors such as longstanding hypertension or diabetes, ongoing clinical oversight is recommended."
            )

        return {
            'is_flagged': is_flagged,
            'result_label': result_label,
            'prediction_code': 'ckd' if is_flagged else 'notckd',
            'probability_ckd': round(prob_ckd * 100, 1),
            'probability_notckd': round(prob_notckd * 100, 1),
            'probability_raw': round(prob_ckd, 4),
            'risk_level': risk_level,
            'risk_badge': risk_badge,
            'model_used': model_name,
            'key_findings': key_findings,
            'recommendation': recommendation,
            'evaluated_parameters': row_data
        }

model_service = CKDModelService()
