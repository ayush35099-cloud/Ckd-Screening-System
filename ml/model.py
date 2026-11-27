import os
import json
import logging
import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class CKDModelService:
    _instance = None

    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), 'models')

        # Model 1 (UCI Standard Model) paths
        self.rf_pipeline_path = os.path.join(self.models_dir, 'ckd_pipeline.joblib')
        self.rf_pkl_path = os.path.join(self.models_dir, 'ckd_model.pkl')
        self.root_pkl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ckd_model.pkl')
        self.dt_pipeline_path = os.path.join(self.models_dir, 'ckd_dt_pipeline.joblib')
        self.metrics_path = os.path.join(self.models_dir, 'metrics.json')
        self.meta_path = os.path.join(self.models_dir, 'feature_meta.json')

        # Model 2 (CDC NHANES Expanded Model) paths
        self.nhanes_rf_path = os.path.join(self.models_dir, 'nhanes_pipeline.joblib')
        self.nhanes_dt_path = os.path.join(self.models_dir, 'nhanes_dt_pipeline.joblib')
        self.nhanes_metrics_path = os.path.join(self.models_dir, 'nhanes_metrics.json')
        self.nhanes_meta_path = os.path.join(self.models_dir, 'nhanes_feature_meta.json')

        # Model 1 artifacts
        self.rf_pipeline = None
        self.dt_pipeline = None
        self.metrics = None
        self.metadata = None

        # Model 2 artifacts
        self.nhanes_rf_pipeline = None
        self.nhanes_dt_pipeline = None
        self.nhanes_metrics = None
        self.nhanes_metadata = None
        self.model_2_available = False

        self._load_artifacts()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_artifacts(self):
        # 1. Load Model 1 (UCI CKD)
        if os.path.exists(self.rf_pipeline_path):
            self.rf_pipeline = joblib.load(self.rf_pipeline_path)
        elif os.path.exists(self.rf_pkl_path):
            self.rf_pipeline = joblib.load(self.rf_pkl_path)
        elif os.path.exists(self.root_pkl_path):
            self.rf_pipeline = joblib.load(self.root_pkl_path)
        else:
            raise FileNotFoundError(f"Model 1 champion pipeline not found at {self.rf_pipeline_path} or {self.rf_pkl_path}. Run ml/train.py first.")

        if os.path.exists(self.dt_pipeline_path):
            self.dt_pipeline = joblib.load(self.dt_pipeline_path)

        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r', encoding='utf-8') as f:
                self.metrics = json.load(f)

        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

        # 2. Load Model 2 (CDC NHANES) independently with graceful fallback
        try:
            if (os.path.exists(self.nhanes_rf_path) and
                os.path.exists(self.nhanes_dt_path) and
                os.path.exists(self.nhanes_metrics_path) and
                os.path.exists(self.nhanes_meta_path)):

                self.nhanes_rf_pipeline = joblib.load(self.nhanes_rf_path)
                self.nhanes_dt_pipeline = joblib.load(self.nhanes_dt_path)

                with open(self.nhanes_metrics_path, 'r', encoding='utf-8') as f:
                    self.nhanes_metrics = json.load(f)

                with open(self.nhanes_meta_path, 'r', encoding='utf-8') as f:
                    self.nhanes_metadata = json.load(f)

                self.model_2_available = True
                logger.info("Model 2 (Expanded CDC NHANES) artifacts successfully loaded.")
            else:
                self.model_2_available = False
                logger.warning("Model 2 artifacts not found. Model 1 remains fully operational.")
        except Exception as e:
            self.model_2_available = False
            logger.error(f"Failed to load Model 2 artifacts ({e}). Model 1 remains fully operational.")

    def is_model_2_available(self):
        return bool(self.model_2_available and self.nhanes_rf_pipeline is not None)

    def get_available_models(self):
        """Returns structured metadata about all registered screening models."""
        return {
            'model_1': {
                'id': 'model_1',
                'name': 'Standard CKD Model',
                'dataset_name': 'UCI Chronic Kidney Disease Dataset',
                'sample_count': 400,
                'features_count': 24,
                'badge': 'Standard Clinical',
                'description': 'Trained using the standardized 24-feature UCI clinical dataset including urinalysis and blood chemistry.',
                'available': bool(self.rf_pipeline is not None)
            },
            'model_2': {
                'id': 'model_2',
                'name': 'Expanded CKD Model',
                'dataset_name': 'CDC NHANES 2017-2018 Survey',
                'sample_count': 5063,
                'features_count': 19,
                'badge': 'Population Survey',
                'description': 'Trained using official CDC NHANES examination, vitals, metabolic and CBC laboratory data.',
                'available': self.is_model_2_available()
            }
        }

    def get_feature_metadata(self, model_id='model_1'):
        m_id = str(model_id or 'model_1').strip().lower()
        if m_id in ['model_2', 'nhanes', 'expanded', '2', 'model2']:
            return self.nhanes_metadata or {}
        return self.metadata or {}

    def get_metrics(self, model_id='model_1'):
        m_id = str(model_id or 'model_1').strip().lower()
        if m_id in ['model_2', 'nhanes', 'expanded', '2', 'model2']:
            return self.nhanes_metrics or {}
        return self.metrics or {}

    def get_all_metrics(self):
        """Returns comparison dictionary of all model metrics for dashboard reporting."""
        return {
            'model_1': self.metrics,
            'model_2': self.nhanes_metrics,
            'model_2_available': self.is_model_2_available()
        }

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

    def predict(self, input_dict, model_choice='Random Forest', model_id='model_1'):
        """
        Unified prediction entrypoint.
        Routes prediction to Model 1 (UCI) or Model 2 (CDC NHANES).
        Validates model_id, rejects unknown identifiers, and applies separate feature schemas.
        """
        m_id = str(model_id or 'model_1').strip().lower()

        if m_id in ['model_1', 'uci', 'standard', '1', 'model1']:
            return self._predict_model_1(input_dict, model_choice=model_choice)
        elif m_id in ['model_2', 'nhanes', 'expanded', '2', 'model2']:
            if not self.is_model_2_available():
                raise RuntimeError(
                    "Expanded CKD Model (CDC NHANES) is currently unavailable on this instance. "
                    "Please select the Standard CKD Model."
                )
            return self._predict_model_2(input_dict, model_choice=model_choice)
        else:
            raise ValueError(
                f"Unknown or unsupported model identifier: '{model_id}'. "
                f"Expected 'model_1' (Standard UCI CKD) or 'model_2' (Expanded CDC NHANES)."
            )

    def _predict_model_1(self, input_dict, model_choice='Random Forest'):
        """Model 1 (UCI CKD 24-feature pipeline) evaluation."""
        if str(model_choice).strip().lower() in ['decision tree', 'decisiontree', 'dt'] and self.dt_pipeline is not None:
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

        # Parse and sanitize strictly the 24 ML features
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
                    val_str = str(val).strip().lower()
                    if val_str in meta['options']:
                        row_data[feat_name] = val_str
                    else:
                        row_data[feat_name] = np.nan
                        missing_params.append(feat_name)

        input_df = pd.DataFrame([row_data])

        primary_eval = self._evaluate_single_pipeline(primary_pipeline, input_df, primary_name)

        if secondary_pipeline is not None:
            secondary_eval = self._evaluate_single_pipeline(secondary_pipeline, input_df, secondary_name)
        else:
            secondary_eval = None

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

        # Model 1 Clinical Findings
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
            'model_id': 'model_1',
            'model_title': 'Standard CKD Model (UCI CKD)',
            'dataset_source': 'UCI Machine Learning Repository',
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

    def _predict_model_2(self, input_dict, model_choice='Random Forest'):
        """Model 2 (CDC NHANES 19-feature pipeline) evaluation."""
        if str(model_choice).strip().lower() in ['decision tree', 'decisiontree', 'dt'] and self.nhanes_dt_pipeline is not None:
            primary_pipeline = self.nhanes_dt_pipeline
            primary_name = 'Decision Tree'
            secondary_pipeline = self.nhanes_rf_pipeline
            secondary_name = 'Random Forest'
        else:
            primary_pipeline = self.nhanes_rf_pipeline
            primary_name = 'Random Forest'
            secondary_pipeline = self.nhanes_dt_pipeline
            secondary_name = 'Decision Tree'

        features_def = self.nhanes_metadata['features']
        row_data = {}
        missing_params = []

        # Parse and sanitize strictly the 19 NHANES features
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
                    val_str = str(val).strip().lower()
                    if val_str in meta['options']:
                        row_data[feat_name] = val_str
                    else:
                        row_data[feat_name] = np.nan
                        missing_params.append(feat_name)

        input_df = pd.DataFrame([row_data])

        primary_eval = self._evaluate_single_pipeline(primary_pipeline, input_df, primary_name)

        if secondary_pipeline is not None:
            secondary_eval = self._evaluate_single_pipeline(secondary_pipeline, input_df, secondary_name)
        else:
            secondary_eval = None

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

        # Model 2 Clinical Findings (tailored to NHANES features)
        key_findings = []
        sc_val = row_data.get('sc', np.nan)
        if not np.isnan(sc_val) and sc_val > 1.2:
            key_findings.append({
                'parameter': 'Serum Creatinine',
                'value': f"{sc_val} mg/dL",
                'status': 'Elevated',
                'detail': 'Elevated serum creatinine signals reduced glomerular filtration capacity.'
            })

        bu_val = row_data.get('bu', np.nan)
        if not np.isnan(bu_val) and bu_val > 20.0:
            key_findings.append({
                'parameter': 'Blood Urea Nitrogen (BUN)',
                'value': f"{bu_val} mg/dL",
                'status': 'Elevated',
                'detail': 'Elevated circulating BUN reflects decreased renal clearance of nitrogenous end-products.'
            })

        sbp_val = row_data.get('sbp', np.nan)
        dbp_val = row_data.get('dbp', np.nan)
        if (not np.isnan(sbp_val) and sbp_val >= 130.0) or (not np.isnan(dbp_val) and dbp_val >= 80.0):
            sbp_str = f"{int(sbp_val)}" if not np.isnan(sbp_val) else "N/A"
            dbp_str = f"{int(dbp_val)}" if not np.isnan(dbp_val) else "N/A"
            key_findings.append({
                'parameter': 'Blood Pressure',
                'value': f"{sbp_str}/{dbp_str} mmHg",
                'status': 'Hypertension Range',
                'detail': 'Systemic blood pressure elevation drives progressive nephron sclerosis.'
            })

        hba1c_val = row_data.get('hba1c', np.nan)
        if not np.isnan(hba1c_val) and hba1c_val >= 6.5:
            key_findings.append({
                'parameter': 'Glycohemoglobin (HbA1c)',
                'value': f"{hba1c_val:.1f}%",
                'status': 'Diabetic Range',
                'detail': 'HbA1c >= 6.5% indicates chronic hyperglycemia, a primary driver of diabetic nephropathy.'
            })

        bgr_val = row_data.get('bgr', np.nan)
        if not np.isnan(bgr_val) and bgr_val >= 126.0:
            key_findings.append({
                'parameter': 'Serum Glucose',
                'value': f"{bgr_val} mg/dL",
                'status': 'Elevated',
                'detail': 'Elevated glucose reflects impaired glycemic control.'
            })

        hemo_val = row_data.get('hemo', np.nan)
        if not np.isnan(hemo_val) and hemo_val < 12.0:
            key_findings.append({
                'parameter': 'Hemoglobin',
                'value': f"{hemo_val} g/dL",
                'status': 'Low (Anemia)',
                'detail': 'Normocytic anemia often develops as functioning nephron mass and erythropoietin secretion decline.'
            })

        uric_val = row_data.get('uric_acid', np.nan)
        if not np.isnan(uric_val) and uric_val > 7.0:
            key_findings.append({
                'parameter': 'Serum Uric Acid',
                'value': f"{uric_val} mg/dL",
                'status': 'Hyperuricemia',
                'detail': 'Elevated uric acid is associated with endothelial dysfunction and renal arteriolosclerosis.'
            })

        bmi_val = row_data.get('bmi', np.nan)
        if not np.isnan(bmi_val) and bmi_val >= 30.0:
            key_findings.append({
                'parameter': 'Body Mass Index (BMI)',
                'value': f"{bmi_val:.1f} kg/m²",
                'status': 'Obesity Class I+',
                'detail': 'Obesity induces glomerular hyperfiltration and accelerates chronic renal hemodynamic stress.'
            })

        if str(row_data.get('htn', 'no')) == 'yes':
            key_findings.append({
                'parameter': 'Hypertension History',
                'value': 'Diagnosed',
                'status': 'Comorbidity',
                'detail': 'Established diagnosis of hypertension.'
            })

        if str(row_data.get('dm', 'no')) in ['yes', 'borderline']:
            key_findings.append({
                'parameter': 'Diabetes History',
                'value': str(row_data.get('dm', 'yes')).capitalize(),
                'status': 'Metabolic Risk',
                'detail': 'Established history of diabetes or borderline diabetes.'
            })

        if primary_eval['is_flagged']:
            recommendation = (
                "The Expanded CDC NHANES screening pipeline has identified laboratory indicators consistent with Chronic Kidney Disease "
                "(derived indicator: eGFR < 60 mL/min/1.73m² or UACR >= 30 mg/g). "
                "Because NHANES represents cross-sectional survey data, a single screening cannot establish 3-month chronicity. "
                "Comprehensive diagnostic correlation by a physician or nephrologist is strongly recommended, including repeat serum creatinine, "
                "spot urine albumin-to-creatinine ratio (UACR), and renal imaging. This educational tool is not a medical device."
            )
        else:
            recommendation = (
                "Routine screening indicators show values within non-CKD baseline ranges on the CDC NHANES population distribution. "
                "This educational result does not rule out kidney disease or transient renal injury. "
                "Routine wellness monitoring, periodic blood pressure checks, and annual laboratory evaluation are advised for at-risk individuals."
            )

        probability_explanation = (
            f"The {primary_name} model predicts '{primary_eval['result_label'].split('–')[0].strip()}' with "
            f"{primary_eval['confidence']}% model confidence ({primary_eval['probability_ckd']}% statistical probability of CKD indicators vs "
            f"{primary_eval['probability_notckd']}% baseline). This score represents mathematical class probability trained on the "
            f"CDC NHANES adult population distribution (18.55% derived CKD indicator prevalence) and does NOT represent an individual patient's "
            f"definitive clinical diagnostic certainty."
        )

        return {
            'model_id': 'model_2',
            'model_title': 'Expanded CKD Model (CDC NHANES)',
            'dataset_source': 'CDC NHANES 2017-2018 Survey',
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
            'model_used': f"{primary_name} (CDC NHANES)",
            'missing_parameters': missing_params,
            'missing_count': len(missing_params),
            'key_findings': key_findings,
            'recommendation': recommendation,
            'probability_explanation': probability_explanation,
            'evaluated_parameters': row_data,
            'comparison': comparison
        }

model_service = CKDModelService()
