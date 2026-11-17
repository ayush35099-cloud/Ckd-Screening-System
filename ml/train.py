import os
import json
import numpy as np
import pandas as pd
import joblib

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

def run_training():
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'chronic_kidney_disease_cleaned.csv')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {dataset_path}. Run inspect_data.py first.")

    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Define feature schemas
    # sg, al, su are quantitative/ordinal laboratory measurements
    numerical_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    target_col = 'class'

    # Convert numerical columns
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clean categorical columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.lower().replace({'nan': np.nan})

    # Features and Target
    X = df[numerical_cols + categorical_cols].copy()
    y = df[target_col].map({'ckd': 1, 'notckd': 0}).astype(int)

    print(f"Class distribution: {y.value_counts().to_dict()} (1=CKD, 0=Not CKD)")

    # Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples (CKD: {sum(y_train==1)}, Not CKD: {sum(y_train==0)})")
    print(f"Test set:     {X_test.shape[0]} samples (CKD: {sum(y_test==1)}, Not CKD: {sum(y_test==0)})")

    # Build Preprocessor Pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ]
    )

    # 1. Decision Tree Model
    dt_clf = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )

    dt_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', dt_clf)
    ])

    # 2. Random Forest Model
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )

    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf_clf)
    ])

    # 5-fold Stratified Cross-Validation on Training Set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    dt_cv_results = cross_validate(dt_pipeline, X_train, y_train, cv=cv, scoring=scoring)
    rf_cv_results = cross_validate(rf_pipeline, X_train, y_train, cv=cv, scoring=scoring)

    print("\n--- 5-Fold Stratified Cross-Validation (Training Set) ---")
    print("Decision Tree:")
    for metric in scoring:
        scores = dt_cv_results[f'test_{metric}']
        print(f"  {metric.capitalize()}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    print("\nRandom Forest:")
    for metric in scoring:
        scores = rf_cv_results[f'test_{metric}']
        print(f"  {metric.capitalize()}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Fit models on full training set
    dt_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    # Evaluate on held-out Test Set
    models = {
        'Decision Tree': (dt_pipeline, dt_clf),
        'Random Forest': (rf_pipeline, rf_clf)
    }

    test_metrics = {}
    print("\n--- Held-Out Test Set Evaluation (N=80: 50 CKD, 30 Not CKD) ---")
    for name, (pipe, clf) in models.items():
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0) # Sensitivity
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)

        print(f"\nModel: {name}")
        print(f"  Accuracy:    {acc:.4f} ({acc*100:.1f}%)")
        print(f"  Sensitivity (Recall): {rec:.4f} ({rec*100:.1f}%) [False Negatives: {fn}]")
        print(f"  Specificity: {spec:.4f} ({spec*100:.1f}%) [False Positives: {fp}]")
        print(f"  Precision:   {prec:.4f} ({prec*100:.1f}%)")
        print(f"  F1 Score:    {f1:.4f}")
        print(f"  ROC-AUC:     {roc:.4f}")
        print(f"  Confusion Matrix: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")

        test_metrics[name] = {
            'accuracy': round(float(acc), 4),
            'sensitivity': round(float(rec), 4),
            'specificity': round(float(spec), 4),
            'precision': round(float(prec), 4),
            'f1_score': round(float(f1), 4),
            'roc_auc': round(float(roc), 4),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            },
            'cv_accuracy_mean': round(float(dt_cv_results['test_accuracy'].mean() if name=='Decision Tree' else rf_cv_results['test_accuracy'].mean()), 4),
            'cv_recall_mean': round(float(dt_cv_results['test_recall'].mean() if name=='Decision Tree' else rf_cv_results['test_recall'].mean()), 4),
            'cv_roc_auc_mean': round(float(dt_cv_results['test_roc_auc'].mean() if name=='Decision Tree' else rf_cv_results['test_roc_auc'].mean()), 4)
        }

    # Extract feature importance from Random Forest
    rf_fitted_clf = rf_pipeline.named_steps['classifier']
    feature_names = rf_pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = rf_fitted_clf.feature_importances_

    feature_importance_map = {}
    for fname, imp in zip(feature_names, importances):
        clean_name = fname.replace('num__', '').replace('cat__', '')
        base_feature = clean_name.split('_')[0] if '_' in clean_name and clean_name.split('_')[0] in categorical_cols else clean_name
        feature_importance_map[base_feature] = feature_importance_map.get(base_feature, 0.0) + float(imp)

    sorted_importance = sorted(feature_importance_map.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Feature Importances (Random Forest):")
    for feat, imp in sorted_importance[:10]:
        print(f"  {feat:10s}: {imp:.4f} ({imp*100:.1f}%)")

    champion_name = 'Random Forest'
    champion_pipeline = rf_pipeline

    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Save models and pipelines
    pipeline_save_path = os.path.join(models_dir, 'ckd_pipeline.joblib')
    dt_pipeline_save_path = os.path.join(models_dir, 'ckd_dt_pipeline.joblib')
    joblib.dump(champion_pipeline, pipeline_save_path)
    joblib.dump(dt_pipeline, dt_pipeline_save_path)
    print(f"\nSaved Champion Model Pipeline to: {pipeline_save_path}")
    print(f"Saved Decision Tree Pipeline to:   {dt_pipeline_save_path}")

    # Feature Metadata
    feature_metadata = {
        "features": {
            "age": {
                "label": "Age",
                "type": "numerical",
                "unit": "years",
                "min": 1,
                "max": 100,
                "step": 1,
                "normal_range": "18 - 65",
                "default": 45,
                "category": "demographic",
                "description": "Patient age in completed years."
            },
            "bp": {
                "label": "Blood Pressure (Diastolic)",
                "type": "numerical",
                "unit": "mm/Hg",
                "min": 40,
                "max": 200,
                "step": 5,
                "normal_range": "70 - 80 mm/Hg",
                "default": 80,
                "category": "vital",
                "description": "Resting blood pressure measured in millimeters of mercury."
            },
            "sg": {
                "label": "Specific Gravity",
                "type": "numerical",
                "unit": "",
                "options": ["1.005", "1.010", "1.015", "1.020", "1.025"],
                "normal_range": "1.015 - 1.025",
                "default": "1.020",
                "category": "urinalysis",
                "description": "Concentration of particles in urine, reflecting kidney concentrating ability."
            },
            "al": {
                "label": "Albumin",
                "type": "numerical",
                "unit": "scale (0-5)",
                "options": ["0", "1", "2", "3", "4", "5"],
                "normal_range": "0 (Negative)",
                "default": "0",
                "category": "urinalysis",
                "description": "Urine protein level. Values above 0 indicate proteinuria."
            },
            "su": {
                "label": "Sugar",
                "type": "numerical",
                "unit": "scale (0-5)",
                "options": ["0", "1", "2", "3", "4", "5"],
                "normal_range": "0 (Negative)",
                "default": "0",
                "category": "urinalysis",
                "description": "Urine glucose level. Elevated values suggest glycosuria."
            },
            "rbc": {
                "label": "Red Blood Cells (Urine)",
                "type": "categorical",
                "unit": "",
                "options": ["normal", "abnormal"],
                "normal_range": "normal",
                "default": "normal",
                "category": "urinalysis",
                "description": "Microscopic examination for hematuria in urine sediment."
            },
            "pc": {
                "label": "Pus Cell (Urine)",
                "type": "categorical",
                "unit": "",
                "options": ["normal", "abnormal"],
                "normal_range": "normal",
                "default": "normal",
                "category": "urinalysis",
                "description": "Presence of leukocytes indicating potential infection or inflammation."
            },
            "pcc": {
                "label": "Pus Cell Clumps",
                "type": "categorical",
                "unit": "",
                "options": ["notpresent", "present"],
                "normal_range": "notpresent",
                "default": "notpresent",
                "category": "urinalysis",
                "description": "Aggregated white blood cells in urine."
            },
            "ba": {
                "label": "Bacteria",
                "type": "categorical",
                "unit": "",
                "options": ["notpresent", "present"],
                "normal_range": "notpresent",
                "default": "notpresent",
                "category": "urinalysis",
                "description": "Bacterial colonization in urinary tract sample."
            },
            "bgr": {
                "label": "Blood Glucose Random",
                "type": "numerical",
                "unit": "mg/dL",
                "min": 50,
                "max": 500,
                "step": 1,
                "normal_range": "70 - 140 mg/dL",
                "default": 110,
                "category": "blood_chemistry",
                "description": "Random plasma glucose level."
            },
            "bu": {
                "label": "Blood Urea",
                "type": "numerical",
                "unit": "mg/dL",
                "min": 5,
                "max": 400,
                "step": 1,
                "normal_range": "15 - 45 mg/dL",
                "default": 35,
                "category": "renal_function",
                "description": "Waste product of protein metabolism filtered by kidneys."
            },
            "sc": {
                "label": "Serum Creatinine",
                "type": "numerical",
                "unit": "mg/dL",
                "min": 0.2,
                "max": 20.0,
                "step": 0.1,
                "normal_range": "0.6 - 1.2 mg/dL",
                "default": 0.9,
                "category": "renal_function",
                "description": "Crucial kidney biomarker reflecting glomerular filtration rate (GFR)."
            },
            "sod": {
                "label": "Sodium",
                "type": "numerical",
                "unit": "mEq/L",
                "min": 100,
                "max": 170,
                "step": 0.5,
                "normal_range": "135 - 145 mEq/L",
                "default": 138,
                "category": "electrolytes",
                "description": "Serum electrolyte critical for fluid balance and blood pressure."
            },
            "pot": {
                "label": "Potassium",
                "type": "numerical",
                "unit": "mEq/L",
                "min": 2.0,
                "max": 10.0,
                "step": 0.1,
                "normal_range": "3.5 - 5.0 mEq/L",
                "default": 4.2,
                "category": "electrolytes",
                "description": "Serum potassium level; elevated in advanced renal dysfunction."
            },
            "hemo": {
                "label": "Hemoglobin",
                "type": "numerical",
                "unit": "g/dL",
                "min": 3.0,
                "max": 20.0,
                "step": 0.1,
                "normal_range": "13.0 - 17.0 (M) / 12.0 - 15.5 (F)",
                "default": 14.5,
                "category": "hematology",
                "description": "Oxygen-carrying protein; kidneys produce erythropoietin regulating this."
            },
            "pcv": {
                "label": "Packed Cell Volume (Hematocrit)",
                "type": "numerical",
                "unit": "%",
                "min": 10,
                "max": 60,
                "step": 1,
                "normal_range": "38 - 50%",
                "default": 42,
                "category": "hematology",
                "description": "Percentage of whole blood volume composed of erythrocytes."
            },
            "wc": {
                "label": "White Blood Cell Count",
                "type": "numerical",
                "unit": "cells/cumm",
                "min": 2000,
                "max": 30000,
                "step": 100,
                "normal_range": "4,000 - 11,000 cells/cumm",
                "default": 7500,
                "category": "hematology",
                "description": "Total circulating leukocyte count."
            },
            "rc": {
                "label": "Red Blood Cell Count",
                "type": "numerical",
                "unit": "millions/cmm",
                "min": 1.5,
                "max": 8.0,
                "step": 0.1,
                "normal_range": "4.2 - 5.9 millions/cmm",
                "default": 4.8,
                "category": "hematology",
                "description": "Total erythrocyte count in circulating blood."
            },
            "htn": {
                "label": "Hypertension",
                "type": "categorical",
                "unit": "",
                "options": ["no", "yes"],
                "normal_range": "no",
                "default": "no",
                "category": "clinical_history",
                "description": "History of diagnosed high blood pressure."
            },
            "dm": {
                "label": "Diabetes Mellitus",
                "type": "categorical",
                "unit": "",
                "options": ["no", "yes"],
                "normal_range": "no",
                "default": "no",
                "category": "clinical_history",
                "description": "History of diagnosed diabetes (major risk factor for CKD)."
            },
            "cad": {
                "label": "Coronary Artery Disease",
                "type": "categorical",
                "unit": "",
                "options": ["no", "yes"],
                "normal_range": "no",
                "default": "no",
                "category": "clinical_history",
                "description": "History of ischemic cardiovascular heart disease."
            },
            "appet": {
                "label": "Appetite",
                "type": "categorical",
                "unit": "",
                "options": ["good", "poor"],
                "normal_range": "good",
                "default": "good",
                "category": "symptoms",
                "description": "Reported appetite status (poor appetite frequently indicates uremia)."
            },
            "pe": {
                "label": "Pedal Edema",
                "type": "categorical",
                "unit": "",
                "options": ["no", "yes"],
                "normal_range": "no",
                "default": "no",
                "category": "symptoms",
                "description": "Swelling in lower extremities indicating fluid retention."
            },
            "ane": {
                "label": "Anemia",
                "type": "categorical",
                "unit": "",
                "options": ["no", "yes"],
                "normal_range": "no",
                "default": "no",
                "category": "symptoms",
                "description": "Clinical diagnosis or manifestation of anemia."
            }
        },
        "categories": [
            {"id": "demographic", "name": "Demographics & Vitals", "icon": "user"},
            {"id": "urinalysis", "name": "Urinalysis Parameters", "icon": "droplet"},
            {"id": "renal_function", "name": "Renal Function & Electrolytes", "icon": "activity"},
            {"id": "hematology", "name": "Hematology / Complete Blood Count", "icon": "disc"},
            {"id": "clinical_history", "name": "Clinical History & Comorbidities", "icon": "file-text"},
            {"id": "symptoms", "name": "Physical Signs & Symptoms", "icon": "alert-circle"}
        ]
    }

    metrics_path = os.path.join(models_dir, 'metrics.json')
    meta_path = os.path.join(models_dir, 'feature_meta.json')

    metrics_payload = {
        'project': 'Chronic Kidney Disease Screening',
        'institution': 'Dayananda Sagar College of Engineering, Bengaluru',
        'department': 'Department of Artificial Intelligence and Machine Learning',
        'course': 'BCP308 – Community Project',
        'team': 'Team A-12 (Ayush Anjan Choudhary, Harsh Kumawat, Krishna Arora, Kunal Kumar Singh)',
        'dataset': {
            'name': 'UCI Chronic Kidney Disease Dataset',
            'total_samples': len(df),
            'ckd_samples': int((df['class'] == 'ckd').sum()),
            'notckd_samples': int((df['class'] == 'notckd').sum()),
            'features_count': len(numerical_cols) + len(categorical_cols),
            'train_size': len(X_train),
            'test_size': len(X_test)
        },
        'champion_model': champion_name,
        'models': test_metrics,
        'feature_importance_top10': sorted_importance[:10]
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics_payload, f, indent=2)
    with open(meta_path, 'w') as f:
        json.dump(feature_metadata, f, indent=2)

    print(f"Saved evaluation metrics to: {metrics_path}")
    print(f"Saved feature metadata to:    {meta_path}")

if __name__ == '__main__':
    run_training()
