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

## Dataset Descriptions & Dual-Model Architecture

RenalCare features a **Dual Independent Dataset & Model Architecture**, enabling healthcare workers to evaluate patients using either a clinical laboratory benchmark model or an expanded national population survey model:

### 1. Model 1 (Standard): UCI Chronic Kidney Disease Benchmark
Contributed by Dr. P. Soundarapandian, Senior Consultant Nephrologist, Apollo Hospitals, Karaikudi, Tamil Nadu, India:
- **Total Records:** 400 patient instances (250 CKD positive, 150 non-CKD baseline).
- **Features (24 Input Variables):**
  - **Urinalysis (7):** Specific Gravity (`sg`), Albumin / Proteinuria (`al`), Urine Sugar (`su`), Red Blood Cells (`rbc`), Pus Cells (`pc`), Pus Cell Clumps (`pcc`), Bacteria (`ba`).
  - **Renal Chemistry & Electrolytes (6):** Blood Pressure (`bp`), Blood Glucose Random (`bgr`), Blood Urea (`bu`), Serum Creatinine (`sc`), Sodium (`sod`), Potassium (`pot`).
  - **Hematology / CBC (4):** Hemoglobin (`hemo`), Packed Cell Volume / Hematocrit (`pcv`), White Blood Cell Count (`wc`), Red Blood Cell Count (`rc`).
  - **Systemic Comorbidities & Symptoms (7):** Hypertension (`htn`), Diabetes Mellitus (`dm`), Coronary Artery Disease (`cad`), Appetite (`appet`), Pedal Edema (`pe`), Anemia (`ane`).
  - **Target (1):** Classification (`ckd` vs `notckd`).

### 2. Model 2 (Expanded): CDC NHANES (2017–2018 Survey Cycle)
Derived from the official US Centers for Disease Control and Prevention (CDC) National Health and Nutrition Examination Survey:
- **Total Adult Records:** 5,063 cleaned instances (939 CKD indicator positive, 4,124 baseline negative; 18.55% derived prevalence).
- **Features (19 Non-Invasive Variables):**
  - **Demographics & Examination (6):** Age, Sex, BMI, Systolic BP (`sbp`), Diastolic BP (`dbp`), Hypertension history (`htn`).
  - **Comprehensive Metabolic Panel & Glycemia (6):** Blood Glucose (`bgr`), Blood Urea Nitrogen (`bu`), Serum Creatinine (`sc`), HbA1c (`hba1c`), Uric Acid (`uric_acid`), Diabetes history (`dm`).
  - **Biochemistry & Serum Electrolytes (4):** Sodium (`sod`), Potassium (`pot`), Total Protein (`tot_protein`), Serum Albumin (`serum_alb`).
  - **Complete Blood Count (3):** Hemoglobin (`hemo`), Packed Cell Volume / Hematocrit (`pcv`), White Blood Cell Count (`wc`).
- **Target Derivation & Leakage Prevention:**
  CKD indicator target is derived according to KDIGO guidelines: eGFR < 60 mL/min/1.73m² (computed via the 2021 CKD-EPI race-free equation) OR urine albumin-to-creatinine ratio (UACR) &ge; 30 mg/g. Neither eGFR nor UACR is present in the feature matrix, preventing target leakage.

### Dataset Download & Model Training Instructions

To reproduce or retrain both models independently from source:

```bash
# 1. Download official CDC NHANES XPT files and generate cleaned dataset:
python ml/build_nhanes_dataset.py

# 2. Train Model 2 (CDC NHANES Random Forest & Decision Tree):
python ml/train_nhanes.py

# 3. Train Model 1 (UCI CKD Random Forest & Decision Tree):
python ml/train.py

# 4. Run the complete automated test suite:
python -m unittest discover -s tests -p "test_*.py"
```

---

## Machine Learning Pipeline & Evaluation

### Data Preprocessing & Leakage Prevention
1. **Inconsistency Resolution:** Normalized inconsistent strings, tab escapes (`\t`), trailing commas, and label variations present in raw ARFF files.
2. **Numerical Processing:** Standardized quantitative features (`age`, `bp`, `sg`, `al`, `su`, `bgr`, `bu`, `sc`, `sod`, `pot`, `hemo`, `pcv`, `wc`, `rc`) with `SimpleImputer(strategy='median')` and `StandardScaler()`.
3. **Categorical Processing:** Nominal clinical indicators (`rbc`, `pc`, `pcc`, `ba`, `htn`, `dm`, `cad`, `appet`, `pe`, `ane`) processed with `SimpleImputer(strategy='most_frequent')` and `OneHotEncoder(handle_unknown='ignore')`.
4. **Leakage Safeguard:** Complete preprocessing and estimation are encapsulated within Scikit-Learn `Pipeline` objects.

### Comparative Benchmark & Dual-Pipeline Architecture

RenalCare implements a **Dual-Pipeline Architecture** that executes both the champion **Random Forest** ensemble and an interpretable **Decision Tree** on each patient screening submission to provide comparative cross-validation:

| Metric | Random Forest (Champion) | Decision Tree |
| :--- | :---: | :---: |
| **Model Architecture** | 100-Tree Forest (`n_estimators=100`) | Pruned Tree (`max_depth=5, min_samples_split=4`) |
| **5-Fold CV Accuracy** | **98.75% (&plusmn;1.17%)** | 95.63% (&plusmn;1.82%) |
| **5-Fold CV Recall (Sensitivity)** | **99.00% (&plusmn;1.22%)** | 95.50% (&plusmn;1.87%) |
| **Held-Out Test Accuracy (N=80)** | **100.0%** | 96.25% |
| **Held-Out Sensitivity (Recall)** | **100.0%** | 94.00% |
| **Held-Out Specificity** | **100.0%** | 100.0% |
| **Precision** | **100.0%** | 100.0% |
| **F1-Score** | **1.0000** | 0.9691 |
| **ROC-AUC** | **1.0000** | 0.9890 |
| **Confusion Matrix (Held-Out N=80)** | TP: 50, TN: 30, FP: 0, FN: 0 | TP: 47, TN: 30, FP: 0, FN: 3 |
| **Clinical Role** | Primary Screening Classifier | Comparative Cross-Validation & Interpretable Rules |

### Distinction Between Model Confidence and Statistical Indication

To ensure clinical and mathematical rigor, RenalCare strictly distinguishes between:
1. **Model Confidence in Prediction:** The mathematical probability associated with the *predicted class* ($P(\text{class}) = \max(P(\text{CKD}), P(\text{Not CKD}))$). For a healthy patient with $P(\text{Not CKD}) = 99.5\%$ and $P(\text{CKD}) = 0.5\%$, the model confidence in predicting "Indicators Not Detected" is **99.5%**.
2. **Statistical CKD Indication Score:** Specifically the probability of CKD marker presence ($P(\text{CKD})$). For a healthy patient, this is **0.5%**; for a CKD-positive patient, this is **98.2%**.
3. **Academic Benchmark Accuracy:** Retrospective test split accuracy (100% RF / 96.25% DT on UCI test split N=80), which evaluates historical population performance rather than individual screening confidence.

> **Academic Evaluation Note:** The benchmark metrics reflect retrospective testing on the standardized UCI held-out test split (N=80). They do not constitute prospective clinical validation or guaranteed real-world diagnostic certainty. In real-world clinical practice, no screening model achieves 100% sensitivity across heterogeneous patient cohorts.

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
9. **Batch CSV Screening:** Community health camp feature supporting batch uploads of up to 100 patient records in a single CSV file, with downloadable result summaries.
10. **CSV History Export:** Export filtered clinical logs to CSV with built-in protection against spreadsheet formula injection.
11. **Automatic Email Report Delivery:** Consent-gated transactional email delivery via Resend issuing secure, 24-hour cryptographic PDF download links with automatic retry handling and manual resend controls.

---

## Project Architecture

```
ckd-screening-system/
├── app.py                 # Flask application entry point & route definitions
├── config.py              # Configuration module (environment variables & settings)
├── database.py            # SQLite & PostgreSQL database abstraction layer
├── email_service.py       # Modular email provider, token security & retry engine
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
│       └── renalcare-doctor.jpg # Physician portrait in consultation setting
├── templates/
│   ├── base.html          # Base layout, navbar, theme toggle, footer
│   ├── index.html         # Editorial landing page
│   ├── dashboard.html     # Clinical dashboard with charts & activity
│   ├── screen.html        # 24-parameter structured screening form
│   ├── result.html        # Prediction outcome, email delivery & recommendations
│   ├── history.html       # Searchable & filterable history records
│   ├── research.html      # Technical ML methodology & benchmark report
│   ├── report_link_error.html # Branded expired/invalid secure download link page
│   └── emails/
│       ├── report_ready.html  # Patient HTML notification email
│       └── report_ready.txt   # Plain-text notification fallback
└── tests/
    ├── test_model.py          # ML unit tests
    ├── test_pdf.py            # PDF generation tests
    ├── test_routes.py         # Web view & REST API route integration tests
    ├── test_email_delivery.py # Email workflow, consent, tokens & security tests
    └── test_patient_profile.py # Persistent patient profiles, auto-lookup & diagnostics tests
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
*(All 69 unit and integration tests should pass cleanly against isolated temporary test databases).*

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
  "version": "1.1.0",
  "database_backend": "sqlite",
  "database_connected": true,
  "database_persistence": "ephemeral_sqlite",
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
    "prediction_code": "ckd",
    "confidence": 98.2,
    "probability_ckd": 98.2,
    "probability_notckd": 1.8,
    "risk_level": "High Statistical Indication",
    "model_used": "Random Forest",
    "comparison": {
      "random_forest": {
        "prediction_code": "ckd",
        "confidence": 98.2,
        "probability_ckd": 98.2,
        "is_flagged": true
      },
      "decision_tree": {
        "prediction_code": "ckd",
        "confidence": 100.0,
        "probability_ckd": 100.0,
        "is_flagged": true
      },
      "agreement": true,
      "consensus_label": "Complete Consensus (Both Models Agree)"
    },
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

### 5. Patient Profile Lookup
```http
GET /api/patient/<patient_id>
```
**Response:**
```json
{
  "found": true,
  "profile": {
    "patient_id": "PT-2026-081",
    "patient_name": "R. Sharma",
    "email": "patient@example.com",
    "email_verified": false,
    "consent_recorded": true,
    "consent_at": "2026-11-25 14:00:00 UTC",
    "updated_at": "2026-11-25 14:00:00 UTC"
  }
}
```

### 6. Email Service Diagnostics (Clinician Authorized)
```http
GET /api/admin/email-diagnostics?limit=50
```

---

## Persistent Patient Profile System & Automated Report Delivery

RenalCare features a persistent **Patient Profile System** combined with a consent-gated, background **Automatic Email Report Delivery System**.

### Patient Profiles & Automatic Retrieval
1. **Persistent Demographics:** Patient records are persisted in the `patient_profiles` table (SQLite & PostgreSQL) storing `patient_id` (unique), `patient_name`, `email`, `consent_recorded`, `consent_at`, `created_at`, and `updated_at`.
2. **Auto-Lookup on Screening Form:** When a healthcare worker enters a `patient_id` on `/screen`, the application retrieves their saved profile, auto-populating patient name, email, and consent status without requiring repetitive manual re-entry.
3. **Flexible Updates & Optional Intake:** Clinicians can modify email addresses or consent preferences at any time. Patient email entry is strictly optional; screening and immediate PDF generation never require an email.

### Clinical Email Workflow & Consent
1. **Explicit Patient Consent:** Report delivery requires explicit consent:
   > *"I consent to receiving my screening report at the email address provided."*
2. **Non-Blocking Screening:** Missing or invalid email addresses never prevent screening completion. If consent is absent, emails are never dispatched.
3. **Clean Outcome Page:** The screening outcome page (`/result/<screening_uuid>`) focuses strictly on clinical evaluation, risk tiers, and one-click PDF download without extraneous email status boxes.
4. **Developer Diagnostics:** Authorized clinicians can inspect real-time delivery state and error classifications via `/api/admin/email-diagnostics` without exposing clinical data or API credentials.

### Security, Privacy & Time-Limited Download Links
- **Zero Clinical Data in Emails:** Subject lines (`"Your RenalCare Screening Report Is Ready"`) and email bodies intentionally exclude CKD risk predictions, lab parameters, and patient identifiers.
- **Cryptographic Tokens:** Each delivery creates a 256-bit `secrets.token_urlsafe(32)` token and stores **only its SHA-256 hash** (`token_hash`) in `email_deliveries` alongside a **24-hour UTC expiry** (`token_expiry`).
- **HTTPS-Only Links & Clean Error Handling:** Patients download reports via `https://<app>/report/download/<token>` with `Cache-Control: no-store`. Expired links return HTTP `410 Gone` with clear instructions to contact the clinic; original screening records remain intact.
- **Authorized Resend & Abuse Protection:** Authorized healthcare workers (clinician session + CSRF token + rate limiting) can issue a fresh 24-hour replacement link from `/result/<screening_uuid>`, automatically invalidating previous tokens.

### Retry & Failure Handling
- Transient errors (`429 rate_limit_exceeded`, `5xx`, network timeouts) are retried up to 3 total attempts using bounded exponential backoff and `Idempotency-Key` headers to prevent duplicate sends.
- Permanent errors (`401`, `403`, `400/422`, `daily_quota_exceeded`) fail immediately without spinning retries.
- **Free-Tier Background Worker Note:** To stay compatible with Render's free tier without external Redis/Celery infrastructure, email dispatch runs in a daemon background thread inside Gunicorn. If the free container spins down mid-retry, healthcare workers can safely trigger a manual resend from the screening result page.

### Resend Setup & Free-Tier Verification Limitations
Set the following environment variables in your `.env` file or Render Web Service **Environment** tab:

| Variable | Description | Example / Default |
| :--- | :--- | :--- |
| `RESEND_API_KEY` | Resend API key (keep secret; leave blank to disable email cleanly) | `re_...` |
| `MAIL_FROM` | Verified sender identity permitted by Resend | `RenalCare <onboarding@resend.dev>` |
| `APP_BASE_URL` | Public HTTPS base URL used inside email download links | `https://ckd-screening-system.onrender.com` |
| `EMAIL_REPORT_ENABLED` | Master toggle (`true` / `false`) | `true` |
| `DOWNLOAD_TOKEN_EXPIRY_HOURS` | Token validity duration in hours | `24` |

> **Important Free-Tier Limitation (Stage A Testing):** Without a verified custom domain in Resend, accounts using `onboarding@resend.dev` can **only deliver to the account owner's verified email address**. Arbitrary patient email addresses will return a `403` verification restriction until custom domain DNS authentication is configured.

### Stage B: Future Custom Domain & Email Authentication (SPF, DKIM, DMARC)
When deploying with an institutional or clinic domain (e.g. `reports@renalcare.org`), add and verify the domain in Resend by publishing the required DNS records:
- **SPF (Sender Policy Framework):** Authorizes Resend's mail servers to send on behalf of your domain.
- **DKIM (DomainKeys Identified Mail):** Cryptographically signs outgoing emails so receiving mail servers verify authenticity and integrity.
- **DMARC (Domain-based Message Authentication, Reporting, and Conformance):** Instructs recipient servers how to handle messages that fail SPF/DKIM alignment, protecting against spoofing.

After verification, update `MAIL_FROM` to `RenalCare <reports@yourdomain.com>`. Always comply with clinical privacy laws and understand that no technical setup can guarantee Primary Inbox placement across all external mail providers.

---

## Render Deployment Guide

The application is pre-configured for deployment to **Render**'s free tier.

### Free-Tier Hosting Characteristics & Limitations
- **Sleeping Instances:** Free Render web services spin down after 15 minutes of inactivity. The first incoming request may experience a 30–50 second cold start.
- **Ephemeral Filesystem (SQLite):** By default without a managed database attached, RenalCare operates in `ephemeral_sqlite` mode. On Render's free tier, the local SQLite database file (`renalcare.db`) is stored in an ephemeral disk that resets when the service spins down, restarts, or deploys.
- **Persistent Storage (PostgreSQL):** For persistent production storage across restarts and redeployments, connect a Render PostgreSQL database instance via the `DATABASE_URL` environment variable as detailed below.

### Configuring Persistent Render PostgreSQL (Step-by-Step)
RenalCare features a dual-dialect database abstraction layer supporting both SQLite and PostgreSQL without code changes.

1. **Create a Render PostgreSQL Database:**
   - Log in to your [Render Dashboard](https://dashboard.render.com/).
   - Click **New +** &rarr; **PostgreSQL**.
   - Enter a name (e.g. `renalcare-db`) and select your closest region.
   - Choose the **Free** instance tier.
   - Click **Create Database**.
2. **Retrieve the Connection String:**
   - On the newly created database page, scroll down to the **Connections** section.
   - Copy the **Internal Database URL** (recommended for services in the same Render region) or the **External Database URL**.
   - The connection string will look like: `postgres://renalcare_user:password@dpg-...-a/renalcare_db`.
3. **Configure the RenalCare Web Service:**
   - Open your RenalCare Web Service from the Render Dashboard.
   - Click **Environment** in the left navigation menu.
   - Click **Add Environment Variable**.
   - Set **Key**: `DATABASE_URL`
   - Set **Value**: `<Pasted Internal Database URL>`
   - Click **Save Changes**.
4. **Verification:**
   - Render automatically triggers a zero-downtime redeployment.
   - Once deployed, visit `https://<your-app>.onrender.com/api/health`.
   - Verify the health response returns:
     ```json
     {
       "status": "healthy",
       "database_backend": "postgres",
       "database_connected": true,
       "database_persistence": "persistent"
     }
     ```
   - Schema tables (`screenings`) and indexes are auto-migrated on startup. All screening records, batch runs, and dashboard analytics will now persist permanently across all redeployments and sleep cycles.

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
