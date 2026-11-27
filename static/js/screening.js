/**
 * RenalCare - Screening Form Logic
 * Interactive model switching, pre-fills, input validation, and submission confirmation
 */

document.addEventListener('DOMContentLoaded', () => {
    initModelSelection();
    initModel1Workflow();
    initModel2Workflow();
});

function initModelSelection() {
    const cardModel1 = document.getElementById('cardModel1');
    const cardModel2 = document.getElementById('cardModel2');
    const radioModel1 = document.getElementById('radioModel1');
    const radioModel2 = document.getElementById('radioModel2');
    const btnConfirmModel = document.getElementById('btnConfirmModel');
    const activeModelIndicator = document.getElementById('activeModelIndicator');
    const activeModelLabel = document.getElementById('activeModelLabel');
    const btnSwitchModel = document.getElementById('btnSwitchModel');
    const modelSelectionSection = document.getElementById('modelSelectionSection');

    const model1Workflow = document.getElementById('model1Workflow');
    const model2Workflow = document.getElementById('model2Workflow');

    function selectCard(modelId) {
        if (modelId === 'model_2') {
            if (radioModel2) radioModel2.checked = true;
            if (cardModel2) cardModel2.classList.add('active');
            if (cardModel1) cardModel1.classList.remove('active');
        } else {
            if (radioModel1) radioModel1.checked = true;
            if (cardModel1) cardModel1.classList.add('active');
            if (cardModel2) cardModel2.classList.remove('active');
        }
    }

    if (cardModel1) {
        cardModel1.addEventListener('click', () => selectCard('model_1'));
    }
    if (cardModel2) {
        cardModel2.addEventListener('click', () => selectCard('model_2'));
    }

    function activateModel(modelId, smoothScroll = true) {
        if (modelId === 'model_2') {
            if (model1Workflow) model1Workflow.style.display = 'none';
            if (model2Workflow) model2Workflow.style.display = 'block';
            if (activeModelLabel) activeModelLabel.textContent = 'Expanded CKD Model (CDC NHANES • 19 Features)';
            if (activeModelIndicator) activeModelIndicator.style.display = 'flex';
            if (smoothScroll && model2Workflow) {
                model2Workflow.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        } else {
            if (model2Workflow) model2Workflow.style.display = 'none';
            if (model1Workflow) model1Workflow.style.display = 'block';
            if (activeModelLabel) activeModelLabel.textContent = 'Standard CKD Model (UCI CKD • 24 Features)';
            if (activeModelIndicator) activeModelIndicator.style.display = 'flex';
            if (smoothScroll && model1Workflow) {
                model1Workflow.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }
        if (window.lucide) {
            window.lucide.createIcons();
        }
    }

    if (btnConfirmModel) {
        btnConfirmModel.addEventListener('click', () => {
            const selected = radioModel2 && radioModel2.checked ? 'model_2' : 'model_1';
            activateModel(selected, true);
        });
    }

    if (btnSwitchModel) {
        btnSwitchModel.addEventListener('click', () => {
            if (modelSelectionSection) {
                modelSelectionSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        });
    }

    // Community Diagnostic Camp Mode Switch Handlers
    const btnCampSelectNhanes = document.getElementById('btnCampSelectNhanes');
    if (btnCampSelectNhanes) {
        btnCampSelectNhanes.addEventListener('click', () => {
            selectCard('model_2');
            activateModel('model_2', true);
        });
        btnCampSelectNhanes.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                selectCard('model_2');
                activateModel('model_2', true);
            }
        });
    }

    const btnCampSwitchToModel1 = document.getElementById('btnCampSwitchToModel1');
    if (btnCampSwitchToModel1) {
        btnCampSwitchToModel1.addEventListener('click', (e) => {
            e.stopPropagation();
            selectCard('model_1');
            activateModel('model_1', true);
        });
    }

    // Check URL parameters (e.g. ?model=model_2)
    const urlParams = new URLSearchParams(window.location.search);
    const paramModel = urlParams.get('model');
    if (paramModel === 'model_2' || paramModel === 'nhanes') {
        selectCard('model_2');
        activateModel('model_2', false);
    } else if (paramModel === 'model_1' || paramModel === 'uci') {
        selectCard('model_1');
        activateModel('model_1', false);
    }
}

function initModel1Workflow() {
    const form = document.getElementById('ckdScreeningForm');
    if (!form) return;

    const btnFillNormal = document.getElementById('btnFillNormal');
    const btnFillModerate = document.getElementById('btnFillModerate');
    const btnFillHighRisk = document.getElementById('btnFillHighRisk');
    const btnReset = document.getElementById('btnResetForm');
    const reviewModal = document.getElementById('reviewModal');
    const btnConfirmSubmit = document.getElementById('btnConfirmSubmit');
    const btnCloseModal = document.getElementById('btnCloseModal');

    function getNormalSample() {
        return {
            patient_id: 'PT-NORM-' + Math.floor(1000 + Math.random() * 9000),
            patient_name: 'Sample Baseline Patient',
            age: 34,
            sex: 'Female',
            model_choice: 'Random Forest',
            bp: 75,
            sg: '1.025',
            al: '0',
            su: '0',
            rbc: 'normal',
            pc: 'normal',
            pcc: 'notpresent',
            ba: 'notpresent',
            bgr: 98,
            bu: 26,
            sc: 0.8,
            sod: 140,
            pot: 4.2,
            hemo: 15.2,
            pcv: 45,
            wc: 6800,
            rc: 5.1,
            htn: 'no',
            dm: 'no',
            cad: 'no',
            appet: 'good',
            pe: 'no',
            ane: 'no'
        };
    }

    function getModerateSample() {
        return {
            patient_id: 'PT-MOD-' + Math.floor(1000 + Math.random() * 9000),
            patient_name: 'Sample Moderate Indicators Patient',
            age: 54,
            sex: 'Male',
            model_choice: 'Random Forest',
            bp: 80,
            sg: '1.015',
            al: '1',
            su: '0',
            rbc: 'normal',
            pc: 'normal',
            pcc: 'notpresent',
            ba: 'notpresent',
            bgr: 135,
            bu: 46,
            sc: 1.5,
            sod: 137,
            pot: 4.6,
            hemo: 11.8,
            pcv: 36,
            wc: 8200,
            rc: 4.2,
            htn: 'yes',
            dm: 'no',
            cad: 'no',
            appet: 'good',
            pe: 'no',
            ane: 'no'
        };
    }

    function getHighRiskSample() {
        return {
            patient_id: 'PT-EVAL-' + Math.floor(1000 + Math.random() * 9000),
            patient_name: 'Sample Evaluation Patient',
            age: 63,
            sex: 'Male',
            model_choice: 'Random Forest',
            bp: 90,
            sg: '1.010',
            al: '3',
            su: '2',
            rbc: 'abnormal',
            pc: 'abnormal',
            pcc: 'present',
            ba: 'present',
            bgr: 220,
            bu: 86,
            sc: 3.2,
            sod: 130,
            pot: 5.4,
            hemo: 9.4,
            pcv: 29,
            wc: 11400,
            rc: 3.2,
            htn: 'yes',
            dm: 'yes',
            cad: 'yes',
            appet: 'poor',
            pe: 'yes',
            ane: 'yes'
        };
    }

    function populateForm(data) {
        if (!form) return;

        if (form.elements['patient_email']) form.elements['patient_email'].value = '';
        if (form.elements['email_consent']) form.elements['email_consent'].checked = false;

        const profileBadge = document.getElementById('patientProfileBadge');
        if (profileBadge) profileBadge.style.display = 'none';

        for (const [key, val] of Object.entries(data)) {
            const el = form.elements[key];
            if (el) el.value = val;
        }
        checkClinicalWarnings();
    }

    if (btnFillNormal) {
        btnFillNormal.addEventListener('click', () => populateForm(getNormalSample()));
    }
    if (btnFillModerate) {
        btnFillModerate.addEventListener('click', () => populateForm(getModerateSample()));
    }
    if (btnFillHighRisk) {
        btnFillHighRisk.addEventListener('click', () => populateForm(getHighRiskSample()));
    }
    if (btnReset) {
        btnReset.addEventListener('click', () => {
            form.reset();
            checkClinicalWarnings();
        });
    }

    function checkClinicalWarnings() {
        // Serum Creatinine Warning
        const scEl = form.elements['sc'];
        const scWarn = document.getElementById('warning-sc');
        if (scEl && scWarn) {
            const val = parseFloat(scEl.value);
            if (!isNaN(val) && val > 3.0) {
                scWarn.innerHTML = '<strong>Clinical Observation:</strong> Serum Creatinine > 3.0 mg/dL represents a marked elevation consistent with advanced impairment.';
                scWarn.style.display = 'block';
            } else {
                scWarn.style.display = 'none';
            }
        }

        // Blood Pressure Warning
        const bpEl = form.elements['bp'];
        const bpWarn = document.getElementById('warning-bp');
        if (bpEl && bpWarn) {
            const val = parseFloat(bpEl.value);
            if (!isNaN(val) && val >= 100) {
                bpWarn.innerHTML = '<strong>Vascular Check:</strong> Diastolic BP >= 100 mm/Hg indicates severe hypertension risk.';
                bpWarn.style.display = 'block';
            } else {
                bpWarn.style.display = 'none';
            }
        }

        // Potassium Warning
        const potEl = form.elements['pot'];
        const potWarn = document.getElementById('warning-pot');
        if (potEl && potWarn) {
            const val = parseFloat(potEl.value);
            if (!isNaN(val) && val >= 6.0) {
                potWarn.innerHTML = '<strong>Critical Electrolyte Alert:</strong> Potassium >= 6.0 mEq/L (Hyperkalemia) requires urgent validation.';
                potWarn.style.display = 'block';
            } else {
                potWarn.style.display = 'none';
            }
        }

        // Hemoglobin Warning
        const hemoEl = form.elements['hemo'];
        const hemoWarn = document.getElementById('warning-hemo');
        if (hemoEl && hemoWarn) {
            const val = parseFloat(hemoEl.value);
            if (!isNaN(val) && val > 0 && val < 7.0) {
                hemoWarn.innerHTML = '<strong>Hematology Check:</strong> Hemoglobin < 7.0 g/dL represents severe anemia outlier.';
                hemoWarn.style.display = 'block';
            } else {
                hemoWarn.style.display = 'none';
            }
        }
    }

    ['sc', 'bp', 'pot', 'hemo'].forEach(field => {
        const el = form.elements[field];
        if (el) {
            el.addEventListener('input', checkClinicalWarnings);
            el.addEventListener('change', checkClinicalWarnings);
        }
    });
    checkClinicalWarnings();

    // Modal Confirmation Step for Model 1
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }
        buildReviewSummary(form, 'Standard CKD Model (UCI CKD)');
        if (reviewModal) reviewModal.style.display = 'flex';
    });

    if (btnConfirmSubmit) {
        btnConfirmSubmit.addEventListener('click', () => {
            const activeForm = document.getElementById('model2Workflow').style.display === 'block'
                ? document.getElementById('ckdNhanesScreeningForm')
                : document.getElementById('ckdScreeningForm');
            if (activeForm) {
                btnConfirmSubmit.disabled = true;
                btnConfirmSubmit.innerHTML = '<span>Evaluating Laboratory Data...</span>';
                activeForm.submit();
            }
        });
    }

    if (btnCloseModal) {
        btnCloseModal.addEventListener('click', () => {
            if (reviewModal) reviewModal.style.display = 'none';
        });
    }
}

function initModel2Workflow() {
    const form = document.getElementById('ckdNhanesScreeningForm');
    if (!form) return;

    const btnFillNormal = document.getElementById('btnFillNhanesNormal');
    const btnFillModerate = document.getElementById('btnFillNhanesModerate');
    const btnFillHighRisk = document.getElementById('btnFillNhanesHighRisk');
    const btnReset = document.getElementById('btnResetNhanesForm');
    const reviewModal = document.getElementById('reviewModal');

    function getNhanesNormalSample() {
        return {
            patient_id: 'PT-NH-NORM-' + Math.floor(1000 + Math.random() * 9000),
            patient_name: 'Sample Baseline NHANES Patient',
            age: 42,
            sex: 'female',
            model_choice: 'Random Forest',
            bmi: 24.2,
            sbp: 118,
            dbp: 76,
            bgr: 92,
            hba1c: 5.2,
            bu: 13,
            sc: 0.8,
            sod: 140,
            pot: 4.2,
            tot_protein: 7.2,
            serum_alb: 4.4,
            uric_acid: 4.9,
            hemo: 14.2,
            pcv: 42.0,
            wc: 6.5,
            htn: 'no',
            dm: 'no'
        };
    }

    function getNhanesModerateSample() {
        return {
            patient_id: 'PT-NH-MOD-' + Math.floor(1000 + Math.random() * 9000),
            patient_name: 'Sample Moderate NHANES Patient',
            age: 58,
            sex: 'male',
            model_choice: 'Random Forest',
            bmi: 29.4,
            sbp: 138,
            dbp: 86,
            bgr: 122,
            hba1c: 6.2,
            bu: 24,
            sc: 1.45,
            sod: 138,
            pot: 4.6,
            tot_protein: 6.8,
            serum_alb: 3.8,
            uric_acid: 7.4,
            hemo: 12.4,
            pcv: 37.0,
            wc: 7.8,
            htn: 'yes',
            dm: 'no'
        };
    }

    function getNhanesHighRiskSample() {
        return {
            patient_id: 'PT-NH-EVAL-' + Math.floor(1000 + Math.random() * 9000),
            patient_name: 'Sample High-Risk NHANES Patient',
            age: 69,
            sex: 'female',
            model_choice: 'Random Forest',
            bmi: 33.8,
            sbp: 156,
            dbp: 92,
            bgr: 195,
            hba1c: 8.4,
            bu: 38,
            sc: 2.5,
            sod: 134,
            pot: 5.1,
            tot_protein: 6.2,
            serum_alb: 3.1,
            uric_acid: 8.8,
            hemo: 10.2,
            pcv: 31.0,
            wc: 9.8,
            htn: 'yes',
            dm: 'yes'
        };
    }

    function populateNhanesForm(data) {
        if (form.elements['patient_email']) form.elements['patient_email'].value = '';
        if (form.elements['email_consent']) form.elements['email_consent'].checked = false;

        for (const [key, val] of Object.entries(data)) {
            const el = form.elements[key];
            if (el) el.value = val;
        }
        checkNhanesWarnings();
    }

    if (btnFillNormal) {
        btnFillNormal.addEventListener('click', () => populateNhanesForm(getNhanesNormalSample()));
    }
    if (btnFillModerate) {
        btnFillModerate.addEventListener('click', () => populateNhanesForm(getNhanesModerateSample()));
    }
    if (btnFillHighRisk) {
        btnFillHighRisk.addEventListener('click', () => populateNhanesForm(getNhanesHighRiskSample()));
    }
    if (btnReset) {
        btnReset.addEventListener('click', () => {
            form.reset();
            checkNhanesWarnings();
        });
    }

    function checkNhanesWarnings() {
        const scEl = form.elements['sc'];
        const scWarn = document.getElementById('warning-nh-sc');
        if (scEl && scWarn) {
            const val = parseFloat(scEl.value);
            if (!isNaN(val) && val > 3.0) {
                scWarn.innerHTML = '<strong>Serum Creatinine Check:</strong> Value > 3.0 mg/dL indicates marked elevation.';
                scWarn.style.display = 'block';
            } else {
                scWarn.style.display = 'none';
            }
        }

        const sbpEl = form.elements['sbp'];
        const sbpWarn = document.getElementById('warning-nh-sbp');
        if (sbpEl && sbpWarn) {
            const val = parseFloat(sbpEl.value);
            if (!isNaN(val) && val >= 160) {
                sbpWarn.innerHTML = '<strong>Vascular Alert:</strong> Systolic BP >= 160 mmHg indicates Stage 2 hypertension.';
                sbpWarn.style.display = 'block';
            } else {
                sbpWarn.style.display = 'none';
            }
        }

        const hba1cEl = form.elements['hba1c'];
        const hba1cWarn = document.getElementById('warning-nh-hba1c');
        if (hba1cEl && hba1cWarn) {
            const val = parseFloat(hba1cEl.value);
            if (!isNaN(val) && val >= 9.0) {
                hba1cWarn.innerHTML = '<strong>Glycemic Check:</strong> HbA1c >= 9.0% represents significant chronic hyperglycemia.';
                hba1cWarn.style.display = 'block';
            } else {
                hba1cWarn.style.display = 'none';
            }
        }

        const hemoEl = form.elements['hemo'];
        const hemoWarn = document.getElementById('warning-nh-hemo');
        if (hemoEl && hemoWarn) {
            const val = parseFloat(hemoEl.value);
            if (!isNaN(val) && val > 0 && val < 7.0) {
                hemoWarn.innerHTML = '<strong>Hematology Check:</strong> Hemoglobin < 7.0 g/dL signals severe anemia.';
                hemoWarn.style.display = 'block';
            } else {
                hemoWarn.style.display = 'none';
            }
        }
    }

    ['sc', 'sbp', 'hba1c', 'hemo', 'pot'].forEach(field => {
        const el = form.elements[field];
        if (el) {
            el.addEventListener('input', checkNhanesWarnings);
            el.addEventListener('change', checkNhanesWarnings);
        }
    });
    checkNhanesWarnings();

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }
        buildReviewSummary(form, 'Expanded CKD Model (CDC NHANES)');
        if (reviewModal) reviewModal.style.display = 'flex';
    });
}

function escapeHtml(str) {
    return String(str || '').replace(/[&<>"']/g, function(ch) {
        return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[ch];
    });
}

function buildReviewSummary(formEl, modelTitle) {
    const summaryContainer = document.getElementById('reviewSummaryContent');
    if (!summaryContainer || !formEl) return;

    const formData = new FormData(formEl);
    const ptId = escapeHtml(formData.get('patient_id') || 'Auto-assigned');
    const age = escapeHtml(formData.get('age') || 'N/A');
    const model = escapeHtml(formData.get('model_choice') || 'Random Forest');
    const sc = escapeHtml(formData.get('sc') || 'N/A');
    const rawEmail = (formData.get('patient_email') || '').trim();
    const emailConsent = formData.get('email_consent') === 'yes';

    let emailSummaryLine = 'Not provided (PDF report downloadable locally)';
    if (rawEmail) {
        emailSummaryLine = emailConsent
            ? `${escapeHtml(rawEmail)} (Consent recorded — automated email enabled)`
            : `${escapeHtml(rawEmail)} (Consent unchecked — email will NOT be sent)`;
    }

    summaryContainer.innerHTML = `
        <div style="background-color: var(--bg-surface-subtle); padding: 1rem; border-radius: 6px; margin-bottom: 1rem;">
            <p style="margin-bottom: 0.25rem;"><strong>Patient ID:</strong> ${ptId} &nbsp;|&nbsp; <strong>Age:</strong> ${age}</p>
            <p style="margin-bottom: 0.25rem;"><strong>Screening Model:</strong> ${escapeHtml(modelTitle)} (${model})</p>
            <p style="margin-bottom: 0.25rem; font-size: 0.85rem;"><strong>Email Delivery:</strong> ${emailSummaryLine}</p>
            <p style="font-size: 0.85rem; color: var(--text-subtle);">Serum Creatinine: ${sc} mg/dL</p>
        </div>
        <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.4;">
            Please verify that the recorded parameters match the clinical laboratory report. The selected machine learning model will evaluate renal risk indicators.
        </p>
    `;
}
