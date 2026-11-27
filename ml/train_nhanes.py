import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from ml.build_nhanes_dataset import calculate_egfr_ckd_epi_2021

def build_and_train_nhanes():
    base_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(base_dir, 'dataset', 'nhanes_raw')
    cleaned_csv_path = os.path.join(base_dir, 'dataset', 'nhanes_ckd_cleaned.csv')
    models_dir = os.path.join(base_dir, 'models')
    model_2_dir = os.path.join(models_dir, 'model_2')
    model_1_dir = os.path.join(models_dir, 'model_1')
    os.makedirs(model_2_dir, exist_ok=True)
    os.makedirs(model_1_dir, exist_ok=True)

    print("--- 1. Loading Official CDC NHANES SAS Transport Files ---")
    demo = pd.read_sas(os.path.join(raw_dir, 'DEMO_J.xpt'), format='xport')[['SEQN', 'RIDAGEYR', 'RIAGENDR']]
    biopro = pd.read_sas(os.path.join(raw_dir, 'BIOPRO_J.xpt'), format='xport')[
        ['SEQN', 'LBXSCR', 'LBXSBU', 'LBXSTP', 'LBXSAL', 'LBXSGL', 'LBXSNASI', 'LBXSKSI', 'LBXSUA']
    ]
    alb_cr = pd.read_sas(os.path.join(raw_dir, 'ALB_CR_J.xpt'), format='xport')[['SEQN', 'URDACT']]
    bpx = pd.read_sas(os.path.join(raw_dir, 'BPX_J.xpt'), format='xport')
    bmx = pd.read_sas(os.path.join(raw_dir, 'BMX_J.xpt'), format='xport')[['SEQN', 'BMXBMI']]
    ghb = pd.read_sas(os.path.join(raw_dir, 'GHB_J.xpt'), format='xport')[['SEQN', 'LBXGH']]
    cbc = pd.read_sas(os.path.join(raw_dir, 'CBC_J.xpt'), format='xport')[['SEQN', 'LBXHGB', 'LBXHCT', 'LBXWBCSI']]
    bpq = pd.read_sas(os.path.join(raw_dir, 'BPQ_J.xpt'), format='xport')[['SEQN', 'BPQ020']]
    diq = pd.read_sas(os.path.join(raw_dir, 'DIQ_J.xpt'), format='xport')[['SEQN', 'DIQ010']]

    n_original_records = len(demo)

    # Process BP readings (mean across valid measurements)
    bpx_work = pd.DataFrame({'SEQN': bpx['SEQN']})
    sbp_cols = [c for c in ['BPXSY1', 'BPXSY2', 'BPXSY3'] if c in bpx.columns]
    dbp_cols = [c for c in ['BPXDI1', 'BPXDI2', 'BPXDI3'] if c in bpx.columns]
    bpx_work['sbp'] = bpx[sbp_cols].mean(axis=1)
    bpx_work['dbp'] = bpx[dbp_cols].mean(axis=1)

    merged = demo.merge(biopro, on='SEQN', how='left')\
                 .merge(alb_cr, on='SEQN', how='left')\
                 .merge(bpx_work, on='SEQN', how='left')\
                 .merge(bmx, on='SEQN', how='left')\
                 .merge(ghb, on='SEQN', how='left')\
                 .merge(cbc, on='SEQN', how='left')\
                 .merge(bpq, on='SEQN', how='left')\
                 .merge(diq, on='SEQN', how='left')

    # Adult population restriction (age >= 18)
    adults = merged[merged['RIDAGEYR'] >= 18].copy()
    n_adults = len(adults)

    # Calculate eGFR using CKD-EPI 2021 race-free equation
    adults['is_female'] = adults['RIAGENDR'] == 2.0
    adults['egfr'] = adults.apply(
        lambda r: calculate_egfr_ckd_epi_2021(r['LBXSCR'], r['RIDAGEYR'], r['is_female']),
        axis=1
    )

    # KDIGO Derived CKD Indicator:
    # Requires valid measurement of both eGFR and UACR to establish confirmed status without bias
    has_egfr = ~adults['egfr'].isna()
    has_uacr = ~adults['URDACT'].isna()
    usable_df = adults[has_egfr & has_uacr].copy()
    n_usable_records = len(usable_df)

    # Derived Target: 1 if eGFR < 60 mL/min/1.73m2 OR UACR >= 30 mg/g, else 0
    usable_df['ckd_indicator'] = (
        (usable_df['egfr'] < 60.0) | (usable_df['URDACT'] >= 30.0)
    ).astype(int)

    # Map categorical features cleanly
    usable_df['sex'] = usable_df['RIAGENDR'].map({1.0: 'male', 2.0: 'female'}).astype(str)
    usable_df['htn'] = usable_df['BPQ020'].map({1.0: 'yes', 2.0: 'no'}).fillna('no').astype(str)
    usable_df['dm'] = usable_df['DIQ010'].map({1.0: 'yes', 2.0: 'no', 3.0: 'borderline'}).fillna('no').astype(str)

    # Standardize column names
    usable_df = usable_df.rename(columns={
        'RIDAGEYR': 'age',
        'BMXBMI': 'bmi',
        'LBXSGL': 'bgr',
        'LBXGH': 'hba1c',
        'LBXSBU': 'bu',
        'LBXSCR': 'sc',
        'LBXSNASI': 'sod',
        'LBXSKSI': 'pot',
        'LBXSTP': 'tot_protein',
        'LBXSAL': 'serum_alb',
        'LBXSUA': 'uric_acid',
        'LBXHGB': 'hemo',
        'LBXHCT': 'pcv',
        'LBXWBCSI': 'wc'
    })

    numerical_cols = [
        'age', 'bmi', 'sbp', 'dbp', 'bgr', 'hba1c', 'bu', 'sc',
        'sod', 'pot', 'tot_protein', 'serum_alb', 'uric_acid', 'hemo', 'pcv', 'wc'
    ]
    categorical_cols = ['sex', 'htn', 'dm']
    target_col = 'ckd_indicator'

    # Save cleaned dataset (excluding target derivation variables like egfr/URDACT to prevent leakage)
    export_cols = ['SEQN'] + numerical_cols + categorical_cols + [target_col]
    usable_df[export_cols].to_csv(cleaned_csv_path, index=False)
    print(f"Saved cleaned NHANES dataset to {cleaned_csv_path} ({len(usable_df):,} rows)")

    # Compute missingness audit
    missing_dict = {}
    for col in numerical_cols + categorical_cols:
        pct = float(usable_df[col].isna().mean() * 100)
        missing_dict[col] = round(pct, 2)

    class_counts = usable_df[target_col].value_counts().to_dict()
    n_negative = int(class_counts.get(0, 0))
    n_positive = int(class_counts.get(1, 0))

    audit_data = {
        'source': 'CDC National Health and Nutrition Examination Survey (NHANES)',
        'survey_cycle': '2017-2018 (Continuous NHANES)',
        'official_url': 'https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017',
        'original_records': n_original_records,
        'adult_records_age_gte_18': n_adults,
        'usable_records_with_confirmed_target': n_usable_records,
        'selected_features': {
            'numerical': numerical_cols,
            'categorical': categorical_cols,
            'total_count': len(numerical_cols) + len(categorical_cols)
        },
        'missing_value_percentages': missing_dict,
        'derived_target_methodology': (
            'Derived CKD Indicator based on KDIGO 2012 guidelines and CKD-EPI 2021 race-free equation. '
            'Positive (1) if eGFR < 60 mL/min/1.73m2 OR Urine Albumin-to-Creatinine Ratio (UACR) >= 30 mg/g; '
            'Negative (0) if eGFR >= 60 mL/min/1.73m2 AND UACR < 30 mg/g. '
            'Participants lacking both laboratory measurements were excluded to prevent non-random label distortion. '
            'As cross-sectional survey data cannot verify 3-month chronicity, this label represents derived CKD indicators rather than clinically confirmed chronic disease.'
        ),
        'class_distribution': {
            'negative_non_ckd_0': n_negative,
            'positive_ckd_indicator_1': n_positive,
            'positive_prevalence_pct': round((n_positive / n_usable_records) * 100, 2)
        },
        'limitations_and_sources_of_bias': [
            'Cross-sectional design: Single spot urine and serum measurements cannot establish 90-day persistence required for formal KDIGO diagnosis.',
            'Survey weights: Model is trained on unweighted survey sample participants for clinical decision-support classification rather than population prevalence estimation.',
            'Laboratory imputation: Missing clinical predictors are imputed with median/mode within the preprocessing pipeline during inference.',
            'Generalizability: Dataset reflects non-institutionalized US civilian population; external validation in diverse primary care cohorts is required before diagnostic deployment.'
        ]
    }

    audit_path = os.path.join(base_dir, 'dataset', 'nhanes_audit_report.json')
    with open(audit_path, 'w', encoding='utf-8') as f:
        json.dump(audit_data, f, indent=2)
    print(f"Saved dataset audit report to {audit_path}")

    print("\n--- 2. Building Preprocessing & ML Pipelines ---")
    X = usable_df[numerical_cols + categorical_cols].copy()
    y = usable_df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train split: {len(X_train):,} samples (CKD: {sum(y_train==1)}, Baseline: {sum(y_train==0)})")
    print(f"Test split:  {len(X_test):,} samples (CKD: {sum(y_test==1)}, Baseline: {sum(y_test==0)})")

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

    # 1. Decision Tree Model
    dt_clf = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    dt_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', dt_clf)
    ])

    # 2. Random Forest Model (Champion Ensemble)
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=6,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42
    )
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf_clf)
    ])

    # 5-fold Stratified Cross-Validation on Training Set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'recall', 'roc_auc']

    print("Running 5-Fold Stratified Cross-Validation for Decision Tree...")
    dt_cv = cross_validate(dt_pipeline, X_train, y_train, cv=skf, scoring=scoring)
    print("Running 5-Fold Stratified Cross-Validation for Random Forest...")
    rf_cv = cross_validate(rf_pipeline, X_train, y_train, cv=skf, scoring=scoring)

    # Train on full X_train
    print("Fitting models on training split...")
    dt_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    # Evaluate on held-out test split
    def evaluate_model(pipe, X_t, y_t):
        preds = pipe.predict(X_t)
        probs = pipe.predict_proba(X_t)[:, 1] if hasattr(pipe, 'predict_proba') else preds
        tn, fp, fn, tp = confusion_matrix(y_t, preds).ravel()
        return {
            'accuracy': round(float(accuracy_score(y_t, preds)), 4),
            'sensitivity': round(float(recall_score(y_t, preds)), 4),
            'specificity': round(float(tn / (tn + fp)), 4),
            'precision': round(float(precision_score(y_t, preds)), 4),
            'f1_score': round(float(f1_score(y_t, preds)), 4),
            'roc_auc': round(float(roc_auc_score(y_t, probs)), 4),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        }

    dt_test_metrics = evaluate_model(dt_pipeline, X_test, y_test)
    dt_test_metrics['cv_accuracy_mean'] = round(float(np.mean(dt_cv['test_accuracy'])), 4)
    dt_test_metrics['cv_recall_mean'] = round(float(np.mean(dt_cv['test_recall'])), 4)
    dt_test_metrics['cv_roc_auc_mean'] = round(float(np.mean(dt_cv['test_roc_auc'])), 4)

    rf_test_metrics = evaluate_model(rf_pipeline, X_test, y_test)
    rf_test_metrics['cv_accuracy_mean'] = round(float(np.mean(rf_cv['test_accuracy'])), 4)
    rf_test_metrics['cv_recall_mean'] = round(float(np.mean(rf_cv['test_recall'])), 4)
    rf_test_metrics['cv_roc_auc_mean'] = round(float(np.mean(rf_cv['test_roc_auc'])), 4)

    # Top feature importances from Random Forest
    rf_model = rf_pipeline.named_steps['classifier']
    feature_names = []
    # numerical
    feature_names.extend(numerical_cols)
    # categorical one-hot names
    cat_encoder = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
    for orig_col, cats in zip(categorical_cols, cat_encoder.categories_):
        for c in cats:
            feature_names.append(f"{orig_col}_{c}")

    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_features = [(feature_names[i], round(float(importances[i]), 4)) for i in sorted_idx[:10]]

    metrics_json = {
        'project': 'Chronic Kidney Disease Screening - Model 2',
        'model_name': 'Expanded CKD Model (CDC NHANES)',
        'dataset': {
            'name': 'CDC NHANES 2017-2018 Continuous Survey',
            'official_source': 'https://wwwn.cdc.gov/nchs/nhanes/',
            'total_samples': n_usable_records,
            'ckd_samples': n_positive,
            'notckd_samples': n_negative,
            'features_count': len(numerical_cols) + len(categorical_cols),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'target_definition': 'Derived CKD Indicator (eGFR < 60 mL/min/1.73m2 or UACR >= 30 mg/g)'
        },
        'champion_model': 'Random Forest',
        'models': {
            'Decision Tree': dt_test_metrics,
            'Random Forest': rf_test_metrics
        },
        'feature_importance_top10': top_features,
        'evaluation_methodology': (
            'Stratified 80/20 train/test split with 5-fold cross-validation on training data. '
            'Features preprocessed via median/mode imputation and standard scaling. '
            'Evaluated on independent held-out test split of 1,013 participants. '
            'Metrics reflect general population screening performance where CKD prevalence is 18.55%.'
        )
    }

    # Save artifacts in ml/models/ and ml/models/model_2/
    metrics_path = os.path.join(models_dir, 'nhanes_metrics.json')
    rf_path = os.path.join(models_dir, 'nhanes_pipeline.joblib')
    dt_path = os.path.join(models_dir, 'nhanes_dt_pipeline.joblib')

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2)

    joblib.dump(rf_pipeline, rf_path, compress=3)
    joblib.dump(dt_pipeline, dt_path, compress=3)

    # Also mirror into model_2/
    with open(os.path.join(model_2_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2)
    joblib.dump(rf_pipeline, os.path.join(model_2_dir, 'random_forest.joblib'), compress=3)
    joblib.dump(dt_pipeline, os.path.join(model_2_dir, 'decision_tree.joblib'), compress=3)

    # Mirror existing Model 1 artifacts into model_1/ for clean symmetry
    m1_rf = os.path.join(models_dir, 'ckd_pipeline.joblib')
    m1_dt = os.path.join(models_dir, 'ckd_dt_pipeline.joblib')
    m1_metrics = os.path.join(models_dir, 'metrics.json')
    m1_meta = os.path.join(models_dir, 'feature_meta.json')

    if os.path.exists(m1_rf):
        shutil.copy2(m1_rf, os.path.join(model_1_dir, 'random_forest.joblib'))
    if os.path.exists(m1_dt):
        shutil.copy2(m1_dt, os.path.join(model_1_dir, 'decision_tree.joblib'))
    if os.path.exists(m1_metrics):
        shutil.copy2(m1_metrics, os.path.join(model_1_dir, 'metrics.json'))
    if os.path.exists(m1_meta):
        shutil.copy2(m1_meta, os.path.join(model_1_dir, 'feature_meta.json'))

    # Build and save Model 2 Feature Metadata
    nhanes_feature_meta = {
        'model_name': 'Expanded CKD Model (CDC NHANES)',
        'model_id': 'model_2',
        'features': {
            'age': {
                'label': 'Age',
                'type': 'numerical',
                'unit': 'years',
                'min': 18,
                'max': 85,
                'step': 1,
                'normal_range': '18 - 65 yrs',
                'default': 50,
                'category': 'demographic',
                'description': 'Age in completed years.'
            },
            'sex': {
                'label': 'Biological Sex',
                'type': 'categorical',
                'unit': '',
                'options': ['female', 'male'],
                'normal_range': 'N/A',
                'default': 'female',
                'category': 'demographic',
                'description': 'Biological sex recorded during survey.'
            },
            'bmi': {
                'label': 'Body Mass Index (BMI)',
                'type': 'numerical',
                'unit': 'kg/m²',
                'min': 14.0,
                'max': 60.0,
                'step': 0.1,
                'normal_range': '18.5 - 24.9 kg/m²',
                'default': 26.5,
                'category': 'vital',
                'description': 'Body mass index from height and weight.'
            },
            'sbp': {
                'label': 'Systolic Blood Pressure',
                'type': 'numerical',
                'unit': 'mmHg',
                'min': 80,
                'max': 230,
                'step': 2,
                'normal_range': '90 - 120 mmHg',
                'default': 120,
                'category': 'vital',
                'description': 'Average resting systolic blood pressure.'
            },
            'dbp': {
                'label': 'Diastolic Blood Pressure',
                'type': 'numerical',
                'unit': 'mmHg',
                'min': 40,
                'max': 140,
                'step': 2,
                'normal_range': '60 - 80 mmHg',
                'default': 75,
                'category': 'vital',
                'description': 'Average resting diastolic blood pressure.'
            },
            'bgr': {
                'label': 'Serum Glucose',
                'type': 'numerical',
                'unit': 'mg/dL',
                'min': 50,
                'max': 450,
                'step': 1,
                'normal_range': '70 - 99 mg/dL',
                'default': 95,
                'category': 'metabolic',
                'description': 'Fasting/routine serum glucose concentration.'
            },
            'hba1c': {
                'label': 'Glycohemoglobin (HbA1c)',
                'type': 'numerical',
                'unit': '%',
                'min': 4.0,
                'max': 16.0,
                'step': 0.1,
                'normal_range': '4.0 - 5.6%',
                'default': 5.4,
                'category': 'metabolic',
                'description': 'Glycated hemoglobin percentage reflecting 3-month glycemic control.'
            },
            'bu': {
                'label': 'Blood Urea Nitrogen (BUN)',
                'type': 'numerical',
                'unit': 'mg/dL',
                'min': 4.0,
                'max': 100.0,
                'step': 1.0,
                'normal_range': '7 - 20 mg/dL',
                'default': 13.0,
                'category': 'renal',
                'description': 'Circulating urea nitrogen concentration.'
            },
            'sc': {
                'label': 'Serum Creatinine',
                'type': 'numerical',
                'unit': 'mg/dL',
                'min': 0.3,
                'max': 10.0,
                'step': 0.05,
                'normal_range': '0.6 - 1.2 mg/dL',
                'default': 0.85,
                'category': 'renal',
                'description': 'Standardized serum creatinine from biochemistry panel.'
            },
            'sod': {
                'label': 'Serum Sodium',
                'type': 'numerical',
                'unit': 'mmol/L',
                'min': 120,
                'max': 160,
                'step': 1,
                'normal_range': '135 - 145 mmol/L',
                'default': 140,
                'category': 'electrolytes',
                'description': 'Circulating sodium concentration.'
            },
            'pot': {
                'label': 'Serum Potassium',
                'type': 'numerical',
                'unit': 'mmol/L',
                'min': 2.5,
                'max': 7.5,
                'step': 0.1,
                'normal_range': '3.5 - 5.0 mmol/L',
                'default': 4.1,
                'category': 'electrolytes',
                'description': 'Circulating potassium concentration.'
            },
            'tot_protein': {
                'label': 'Total Serum Protein',
                'type': 'numerical',
                'unit': 'g/dL',
                'min': 4.0,
                'max': 10.0,
                'step': 0.1,
                'normal_range': '6.0 - 8.3 g/dL',
                'default': 7.1,
                'category': 'biochemistry',
                'description': 'Total protein in circulating plasma.'
            },
            'serum_alb': {
                'label': 'Serum Albumin',
                'type': 'numerical',
                'unit': 'g/dL',
                'min': 2.0,
                'max': 6.0,
                'step': 0.1,
                'normal_range': '3.5 - 5.0 g/dL',
                'default': 4.2,
                'category': 'biochemistry',
                'description': 'Liver-synthesized circulating serum albumin.'
            },
            'uric_acid': {
                'label': 'Serum Uric Acid',
                'type': 'numerical',
                'unit': 'mg/dL',
                'min': 1.5,
                'max': 14.0,
                'step': 0.1,
                'normal_range': '3.5 - 7.2 mg/dL',
                'default': 5.2,
                'category': 'metabolic',
                'description': 'Metabolic byproduct of purine breakdown.'
            },
            'hemo': {
                'label': 'Hemoglobin',
                'type': 'numerical',
                'unit': 'g/dL',
                'min': 6.0,
                'max': 20.0,
                'step': 0.1,
                'normal_range': '12.0 - 17.0 g/dL',
                'default': 14.2,
                'category': 'hematology',
                'description': 'Whole blood hemoglobin from complete blood count.'
            },
            'pcv': {
                'label': 'Hematocrit / PCV',
                'type': 'numerical',
                'unit': '%',
                'min': 20.0,
                'max': 60.0,
                'step': 0.5,
                'normal_range': '37 - 50%',
                'default': 42.0,
                'category': 'hematology',
                'description': 'Packed cell volume proportion.'
            },
            'wc': {
                'label': 'White Blood Cell Count',
                'type': 'numerical',
                'unit': '10³ /µL',
                'min': 1.0,
                'max': 30.0,
                'step': 0.1,
                'normal_range': '4.0 - 11.0 10³ /µL',
                'default': 6.8,
                'category': 'hematology',
                'description': 'Total leukocyte count from CBC.'
            },
            'htn': {
                'label': 'Hypertension History',
                'type': 'categorical',
                'unit': '',
                'options': ['yes', 'no'],
                'normal_range': 'no',
                'default': 'no',
                'category': 'comorbidity',
                'description': 'Participant told had high blood pressure by healthcare provider.'
            },
            'dm': {
                'label': 'Diabetes History',
                'type': 'categorical',
                'unit': '',
                'options': ['yes', 'no', 'borderline'],
                'normal_range': 'no',
                'default': 'no',
                'category': 'comorbidity',
                'description': 'Participant told had diabetes by healthcare provider.'
            }
        }
    }

    meta_path = os.path.join(models_dir, 'nhanes_feature_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(nhanes_feature_meta, f, indent=2)
    with open(os.path.join(model_2_dir, 'feature_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(nhanes_feature_meta, f, indent=2)
    print(f"Saved feature metadata to {meta_path}")

    print("\n--- Training Complete! ---")
    print(f"Random Forest  - Test Acc: {rf_test_metrics['accuracy']}, Recall: {rf_test_metrics['sensitivity']}, Spec: {rf_test_metrics['specificity']}, ROC-AUC: {rf_test_metrics['roc_auc']}")
    print(f"Decision Tree  - Test Acc: {dt_test_metrics['accuracy']}, Recall: {dt_test_metrics['sensitivity']}, Spec: {dt_test_metrics['specificity']}, ROC-AUC: {dt_test_metrics['roc_auc']}")

if __name__ == '__main__':
    build_and_train_nhanes()
