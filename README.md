# RenalCare: Chronic Kidney Disease (CKD) Screening System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Flask%203.0-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%201.7-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Smarter Screening. Earlier Awareness.**  
> A machine learning decision-support platform designed to assist healthcare workers in Primary Health Centres (PHCs) and community diagnostic camps in detecting early physiological markers of Chronic Kidney Disease (CKD) using 24 routine laboratory parameters.

---

## Academic Project Information

- **Institution:** Dayananda Sagar College of Engineering (DSCE), Bengaluru
- **Department:** Department of Artificial Intelligence and Machine Learning
- **Course:** BCP308 – Community Project
- **Team Identifier:** Team A-12
- **Team Members:**
  - **Ayush Anjan Choudhary** — *Team Lead*
  - **Harsh Kumawat**
  - **Krishna Arora**
  - **Kunal Kumar Singh**

---

## Important Medical Disclaimer

> **IMPORTANT REGULATORY NOTICE:** This platform is an educational screening and clinical decision-support aid. It is **NOT** a medical diagnostic device, clinical certification, or prescription system. It does not diagnose, cure, mitigate, or treat Chronic Kidney Disease. All flagged indicators must be independently investigated and confirmed by a licensed medical practitioner or nephrologist through definitive diagnostic tests (including measured eGFR, 24-hour urinary protein quantification, and renal ultrasonography).

---

## Visual Design & Human-Centered UX

RenalCare's user interface is inspired by premier healthcare institution design standards (notably Cedars-Sinai and Mayo Clinic), completely rejecting generic "AI dashboard" templates, aggressive glassmorphism, or tacky neon gradients.

- **Editorial Typography:** Refined pairings combining authoritative editorial serifs (*Libre Baskerville*) with crisp, accessible sans-serifs (*Plus Jakarta Sans*).
- **Thoughtful Color System:**
  - **Light Mode (Default):** Warm off-white background (`#F8F7F4`), Deep forest green (`#174D3A`), Muted sage (`#4D7260`), Dark charcoal text (`#1A201C`), Pure white cards with soft neutral borders.
  - **Dark Mode:** Deep charcoal (`#151B19`), Dark green-grey surfaces (`#202A26`), Soft sage green (`#88B39F`), Warm white text (`#EDEFEF`).
- **Dynamic Theme Persistence:** Global light/dark toggle persisted across browser sessions via `localStorage` with real-time Chart.js theme adaptation.
- **Genuine Healthcare Imagery:** High-resolution editorial clinical visual depicting authentic physician-patient consultation.
- **No Fabricated Data:** Fresh installations display calm empty states rather than fake statistics or fabricated patient records.

---

## Dataset Description

The machine learning pipeline is trained and validated on the authentic **UCI Chronic Kidney Disease Benchmark Dataset** (contributed by Dr. P. Soundarapandian, Senior Consultant Nephrologist, Apollo Hospitals, Karaikudi, Tamil Nadu, India):

- **Total Records:** 400 patient instances (250 CKD positive, 150 non-CKD baseline).
- **Features (24 Input Variables):**
  - **Urinalysis (7):** Specific Gravity (`sg`), Albumin / Proteinuria (`al`), Urine Sugar (`su`), Red Blood Cells (`rbc`), Pus Cells (`pc`), Pus Cell Clumps (`pcc`), Bacteria (`ba`).
  - **Renal Chemistry & Electrolytes (6):** Blood Pressure (`bp`), Blood Glucose Random (`bgr`), Blood Urea (`bu`), Serum Creatinine (`sc`), Sodium (`sod`), Potassium (`pot`).
  - **Hematology / CBC (4):** Hemoglobin (`hemo`), Packed Cell Volume / Hematocrit (`pcv`), White Blood Cell Count (`wc`), Red Blood Cell Count (`rc`).
  - **Systemic Comorbidities & Symptoms (7):** Hypertension (`htn`), Diabetes Mellitus (`dm`), Coronary Artery Disease (`cad`), Appetite (`appet`), Pedal Edema (`pe`), Anemia (`ane`).
  - **Target (1):** Classification (`ckd` vs `notckd`).

---

## Machine Learning Pipeline & Evaluation

### Data Preprocessing & Leakage Prevention
1. **Inconsistency Resolution:** Normalized inconsistent strings, tab escapes (`\t`), trailing commas, and label variations present in raw ARFF files.
2. **Numerical Processing:** Standardized quantitative features (`age`, `bp`, `sg`, `al`, `su`, `bgr`, `bu`, `sc`, `sod`, `pot`, `hemo`, `pcv`, `wc`, `rc`) with `SimpleImputer(strategy='median')` and `StandardScaler()`.
3. **Categorical Processing:** Nominal clinical indicators (`rbc`, `pc`, `pcc`, `ba`, `htn`, `dm`, `cad`, `appet`, `pe`, `ane`) processed with `SimpleImputer(strategy='most_frequent')` and `OneHotEncoder(handle_unknown='ignore')`.
4. **Leakage Safeguard:** Complete preprocessing and estimation are encapsulated within Scikit-Learn `Pipeline` objects.

### Comparative Benchmark (Decision Tree vs. Random Forest)

Because missing a CKD-positive patient can delay life-saving medical management, **Sensitivity (Recall) was heavily prioritized** while maintaining high specificity.

| Metric | Random Forest (Champion) | Decision Tree |
| :--- | :---: | :---: |
| **5-Fold CV Accuracy** | **98.75% (&plusmn;1.17%)** | 95.63% (&plusmn;1.82%) |
| **5-Fold CV Recall (Sensitivity)** | **99.00% (&plusmn;1.22%)** | 95.50% (&plusmn;1.87%) |
| **Held-Out Test Accuracy (N=80)** | **100.0%** | 96.25% |
| **Held-Out Sensitivity (Recall)** | **100.0%** | 94.00% |
| **Held-Out Specificity** | **100.0%** | 100.0% |
| **Precision** | **100.0%** | 100.0% |
| **F1-Score** | **1.0000** | 0.9691 |
| **ROC-AUC** | **1.0000** | 0.9890 |
| **False Negatives (Held-Out)** | **0 (on held-out test split N=80)** | 3 |

> **Academic Evaluation Note:** The 100% evaluation metrics reflect retrospective benchmark testing on the standardized UCI held-out test split (N=80). They do not constitute prospective clinical validation or guaranteed real-world diagnostic certainty. In real-world clinical practice, no screening model achieves 100% sensitivity across heterogeneous patient cohorts.

### Top Predictive Features
1. **Packed Cell Volume (`pcv` - 17.4%):** Directly mirrors impaired renal erythropoietin production.
2. **Hemoglobin (`hemo` - 16.7%):** Normocytic normochromic anemia severity.
3. **Serum Creatinine (`sc` - 13.2%):** Direct chemical surrogate for glomerular filtration reduction.
4. **Specific Gravity (`sg` - 11.1%):** Impaired tubular osmolality concentration.
5. **Hypertension (`htn` - 10.5%):** Vascular etiology and accelerated nephron sclerosis.

---

## Key Features

1. **Editorial Landing Page:** Cedars-Sinai inspired composition with institutional transparency, clinical workflow guide, and disease education.
2. **Healthcare Dashboard:** Live statistics (Total Screened, Flagged, Non-Flagged, Sensitivity), screening distribution chart, ML benchmark comparison, and recent activity.
3. **Structured Screening Form:** Grouped across 5 clinical fieldsets with inline normal reference ranges, real-time guidance, and pre-submission confirmation review modal.
4. **Sample Pre-Fill Controls:** One-click "Sample: Normal Baseline" and "Sample: High-Risk CKD Profile" buttons for instant diagnostic camp demonstrations.
5. **Automated Risk Stratification:** Calibrated probability calculations generating "Low", "Moderate", and "High" risk tiers with detailed laboratory anomalies.
6. **Printable PDF Reports:** Clinical summaries generated via ReportLab with institutional branding, patient demographics, parameter comparison tables, and disclaimers.
7. **Screening History:** Searchable and filterable database table (by Patient ID, Name, Outcome) with pagination and direct PDF export.
8. **REST API Suite:** Endpoints for health checks, model inference, dashboard statistics, history queries, and PDF streaming.

---

## Project Architecture

```
ckd-screening-system/
├── app.py                 # Flask application entry point & route definitions
├── config.py              # Configuration module (environment variables & settings)
├── database.py            # SQLite & PostgreSQL database abstraction layer
├── report_generator.py    # ReportLab clinical PDF generation engine
├── Procfile               # Production WSGI process definition (Gunicorn)
├── render.yaml            # Render Blueprint deployment configuration
├── requirements.txt       # Production Python dependencies
├── .env.example           # Example environment variables template
├── .gitignore             # Git exclusion rules
├── README.md              # Project documentation
├── ml/
│   ├── train.py           # ML training, cross-validation & artifact serialization
│   ├── model.py           # Model inference service & clinical rule engine
│   ├── dataset/
│   │   ├── chronic_kidney_disease_cleaned.csv  # Standardized 400-row dataset
│   │   └── inspect_data.py                     # EDA & ARFF parser script
│   └── models/
│       ├── ckd_pipeline.joblib                 # Champion Random Forest pipeline
│       ├── ckd_dt_pipeline.joblib              # Decision Tree pipeline
│       ├── metrics.json                        # Serialized validation metrics
│       └── feature_meta.json                   # Feature metadata & reference ranges
├── static/
│   ├── css/
│   │   └── style.css      # Healthcare design system & theme variables
│   ├── js/
│   │   ├── main.js        # Theme toggle & responsive mobile navigation
│   │   ├── screening.js   # Pre-fills & submission confirmation modal
│   │   └── dashboard.js   # Dynamic Chart.js visualizations
│   └── img/
│       ├── logo.svg       # Custom RenalCare SVG brand logo
│       └── hero-doctor.jpg# Editorial healthcare photograph
├── templates/
│   ├── base.html          # Base layout, navbar, theme toggle, footer
│   ├── index.html         # Editorial landing page
│   ├── dashboard.html     # Clinical dashboard with charts & activity
│   ├── screen.html        # 24-parameter structured screening form
│   ├── result.html        # Prediction outcome, findings & recommendations
│   ├── history.html       # Searchable & filterable history records
│   └── research.html      # Technical ML methodology & benchmark report
└── tests/
    ├── test_model.py      # ML unit tests
    ├── test_pdf.py        # PDF generation tests
    └── test_routes.py     # Web view & REST API route integration tests
```

---

## Local Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/ayush35099-cloud/Ckd-Screening-System.git
cd Ckd-Screening-System
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Re-train the Machine Learning Models
```bash
python ml/train.py
```

### 5. Run the Test Suites
```bash
python -m unittest discover -s tests -p "test_*.py"
```
*(All 27 unit and integration tests should pass in under 1 second).*

### 6. Start the Development Server
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

---

## REST API Documentation

### 1. Health Check
```http
GET /api/health
```
**Response:**
```json
{
  "status": "healthy",
  "application": "RenalCare CKD Screening System",
  "champion_model": "Random Forest Classifier",
  "models_loaded": {
    "random_forest": true,
    "decision_tree": true
  }
}
```

### 2. Predict CKD Indicators
```http
POST /api/predict
Content-Type: application/json
```
**Request Body:**
```json
{
  "patient_id": "PT-DEMO-101",
  "model": "Random Forest",
  "parameters": {
    "age": 55,
    "bp": 90,
    "sg": "1.010",
    "al": "2",
    "su": "1",
    "sc": 2.4,
    "bu": 65,
    "hemo": 10.2,
    "pcv": 32,
    "htn": "yes",
    "dm": "yes"
  }
}
```
**Response:**
```json
{
  "success": true,
  "screening_uuid": "e8d7c2a1-...",
  "prediction": {
    "is_flagged": true,
    "result_label": "CKD indicators detected – further clinical evaluation recommended.",
    "probability_ckd": 98.2,
    "risk_level": "High",
    "probability_explanation": "Statistical model output; does not represent guaranteed clinical certainty.",
    "missing_parameters": [],
    "key_findings": [
      {
        "parameter": "Serum Creatinine",
        "value": "2.4 mg/dL",
        "status": "Elevated",
        "detail": "Elevated creatinine suggests decreased glomerular filtration rate (GFR)."
      }
    ],
    "recommendation": "The machine learning screening pipeline has detected markers indicative of Chronic Kidney Disease..."
  }
}
```

### 3. Dashboard Statistics
```http
GET /api/dashboard
```

### 4. Download PDF Report
```http
GET /api/report/<screening_uuid>
```

---

## Render Deployment Guide

The application is pre-configured for deployment to **Render**'s free tier.

### Free-Tier Hosting Characteristics & Limitations
- **Sleeping Instances:** Free Render web services spin down after 15 minutes of inactivity. The first incoming request may experience a 30–50 second cold start.
- **Ephemeral Filesystem:** Local SQLite database files (`renalcare.db`) reset when the service restarts or deploys. For permanent data persistence across restarts, provision a free **Render PostgreSQL** database and set the `DATABASE_URL` environment variable.

### Deployment Steps
1. Push the code to a GitHub repository:
   ```bash
   git add .
   git commit -m "Production release: CKD Screening System"
   git push origin main
   ```
2. Log in to [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** &rarr; **Blueprint** (or **Web Service**).
4. Connect your GitHub repository.
5. If using **Blueprint**, Render will automatically detect `render.yaml`.
6. If creating a manual **Web Service**:
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt && python ml/train.py`
   - **Start Command:** `gunicorn app:app`
   - **Health Check Path:** `/api/health`
7. Click **Deploy**. Your application will be live at `https://<your-subdomain>.onrender.com`.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
