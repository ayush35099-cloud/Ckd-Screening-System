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
        self.rf_pkl_path = os.path.join(self.models_dir, 'ckd_model.pkl')
        self.root_pkl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ckd_model.pkl')
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
        # Support ckd_pipeline.joblib, ml/models/ckd_model.pkl, or root ckd_model.pkl
        if os.path.exists(self.rf_pipeline_path):
            self.rf_pipeline = joblib.load(self.rf_pipeline_path)
        elif os.path.exists(self.rf_pkl_path):
            self.rf_pipeline = joblib.load(self.rf_pkl_path)
        elif os.path.exists(self.root_pkl_path):
            self.rf_pipeline = joblib.load(self.root_pkl_path)
        else:
            raise FileNotFoundError(f"Champion pipeline not found at {self.rf_pipeline_path} or {self.rf_pkl_path}. Run ml/train.py first.")

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

    def _evaluate_single_pipeline(self, pipeline, input_df, model_name):
        """
        Evaluates a single scikit-learn Pipeline on prepared input_df.
        Returns normalized predictions, class probabilities, and true model confidence.
        """
        pred_label_idx = int(pipeline.predict(input_df)[0])  # 1 = ckd, 0 = notckd
        is_flagged = bool(pred_label_idx == 1)

        # Determine class probabilities
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(input_df)[0]
            # classes_ are [0, 1] where 0=notckd, 1=ckd
            prob_notckd = float(probabilities[0])
            prob_ckd = float(probabilities[1])
        else:
            prob_ckd = 1.0 if is_flagged else 0.0
            prob_notckd = 0.0 if is_flagged else 1.0

        # Model confidence is the probability of the predicted class:
        # For notckd: confidence = prob_notckd (e.g. 99.5%)
        # For ckd:    confidence = prob_ckd    (e.g. 98.2%)
        conf_ratio = prob_ckd if is_flagged else prob_notckd
        confidence = round(conf_ratio * 100, 1)
        prob_ckd_pct = round(prob_ckd * 100, 1)
        prob_notckd_pct = round(prob_notckd * 100, 1)

        if is_flagged:
            result_label = "CKD indicators detected – further clinical evaluation recommended."
        else:
            result_label = "CKD indicators not detected by this model – this result does not rule out CKD."

        # Risk tier stratification based on statistical probability of CKD markers
        if prob_ckd >= 0.70:
            risk_level = "High Statistical Indication"
            risk_badge = "risk-high"
        elif prob_ckd >= 0.35:
            risk_level = "Moderate Statistical Indication"
            risk_badge = "risk-moderate"
        else:
            risk_level = "Low Statistical Indication"
            risk_badge = "risk-low"

        return {
            'model_name': model_name,
            'is_flagged': is_flagged,
            'prediction_code': 'ckd' if is_flagged else 'notckd',
            'result_label': result_label,
            'confidence': confidence,
            'probability_ckd': prob_ckd_pct,
            'probability_notckd': prob_notckd_pct,
            'probability_raw': round(prob_ckd, 4),
            'risk_level': risk_level,
            'risk_badge': risk_badge
        }

    def predict(self, input_dict, model_choice='Random Forest'):
        """
        Takes raw dictionary of patient parameters.
        Evaluates both primary chosen model and comparative dual pipeline.
        Returns:
            dict containing primary prediction, true model confidence,
            statistical CKD probability, risk_level, key findings,
            recommendation, and dual-model comparative evaluation.
        """
        if model_choice.lower() in ['decision tree', 'decisiontree', 'dt'] and self.dt_pipeline is not None:
            primary_pipeline = self.dt_pipeline
            primary_name = 'Decision Tree'
            secondary_pipeline = self.rf_pipeline
            secondary_name = 'Random Forest'
        else:
            primary_pipeline = self.rf_pipeline
            primary_name = 'Random Forest'
            secondary_pipeline = self.dt_pipeline
            secondary_name = 'Decision Tree'

        features_def = self.metadata['features']
        row_data = {}
        missing_params = []

        # Parse and sanitize strictly the 24 ML features (ignore demographics like patient_name, patient_id, sex)
        for feat_name, meta in features_def.items():
            val = input_dict.get(feat_name, None)
            if val is None or val == '' or str(val).strip() == '?' or str(val).strip().lower() == 'nan':
                row_data[feat_name] = np.nan
                missing_params.append(feat_name)
            else:
                if meta['type'] == 'numerical':
                    try:
                        row_data[feat_name] = float(val)
                    except (ValueError, TypeError):
                        row_data[feat_name] = np.nan
                        missing_params.append(feat_name)
                else:
                    # Categorical: lowercase stripped string
                    val_str = str(val).strip().lower()
                    if val_str in meta['options']:
                        row_data[feat_name] = val_str
                    else:
                        row_data[feat_name] = np.nan
                        missing_params.append(feat_name)

        # Convert to single-row DataFrame with expected column order
        input_df = pd.DataFrame([row_data])

        # Execute primary model evaluation
        primary_eval = self._evaluate_single_pipeline(primary_pipeline, input_df, primary_name)

        # Execute secondary model evaluation for dual-model comparative cross-validation
        if secondary_pipeline is not None:
            secondary_eval = self._evaluate_single_pipeline(secondary_pipeline, input_df, secondary_name)
        else:
            secondary_eval = None

        # Build dual-model comparison structure
        if primary_name == 'Random Forest':
            rf_summary = primary_eval
            dt_summary = secondary_eval
        else:
            rf_summary = secondary_eval
            dt_summary = primary_eval

        agreement = bool(rf_summary['prediction_code'] == dt_summary['prediction_code']) if (rf_summary and dt_summary) else True
        consensus_label = "Complete Consensus (Both Models Agree)" if agreement else "Model Divergence (Clinical Follow-Up Recommended)"

        comparison = {
            'random_forest': rf_summary,
            'decision_tree': dt_summary,
            'agreement': agreement,
            'consensus_label': consensus_label
        }

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
        if primary_eval['is_flagged']:
            recommendation = (
                "The machine learning screening pipeline has identified laboratory indicators consistent with Chronic Kidney Disease. "
                "Further clinical correlation by a qualified healthcare professional or nephrologist is recommended, "
                "including formal diagnostic tests (e.g. measured eGFR, 24-hour urinary protein quantification, serum electrolytes, and renal ultrasonography). "
                "This tool provides decision-support screening and does not constitute a definitive medical diagnosis."
            )
        else:
            recommendation = (
                "Routine screening indicators show values within non-CKD baseline ranges for this model. "
                "Note that this result does not rule out Chronic Kidney Disease or other renal disorders. "
                "Routine wellness monitoring is advised, and any persistent symptoms (such as unexplained fatigue, changes in urination, or edema) warrant clinical evaluation."
            )

        probability_explanation = (
            f"The {primary_name} model predicts '{primary_eval['result_label'].split('–')[0].strip()}' with "
            f"{primary_eval['confidence']}% model confidence ({primary_eval['probability_ckd']}% statistical probability of CKD markers vs "
            f"{primary_eval['probability_notckd']}% baseline). This score represents mathematical class probability on the "
            f"retrospective UCI training distribution and does NOT represent an individual patient's "
            f"medically validated diagnostic certainty."
        )

        return {
            'is_flagged': primary_eval['is_flagged'],
            'result_label': primary_eval['result_label'],
            'prediction_code': primary_eval['prediction_code'],
            'confidence': primary_eval['confidence'],
            'probability': primary_eval['probability_ckd'],
            'probability_ckd': primary_eval['probability_ckd'],
            'probability_notckd': primary_eval['probability_notckd'],
            'probability_raw': primary_eval['probability_raw'],
            'risk_level': primary_eval['risk_level'],
            'risk_badge': primary_eval['risk_badge'],
            'model_used': primary_name,
            'missing_parameters': missing_params,
            'missing_count': len(missing_params),
            'key_findings': key_findings,
            'recommendation': recommendation,
            'probability_explanation': probability_explanation,
            'evaluated_parameters': row_data,
            'comparison': comparison
        }

model_service = CKDModelService()
