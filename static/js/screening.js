/**
 * RenalCare - Screening Form Logic
 * Interactive pre-fills, input validation, and submission confirmation
 */

document.addEventListener('DOMContentLoaded', () => {
    initScreeningForm();
});

function initScreeningForm() {
    const form = document.getElementById('ckdScreeningForm');
    const btnFillNormal = document.getElementById('btnFillNormal');
    const btnFillHighRisk = document.getElementById('btnFillHighRisk');
    const btnReset = document.getElementById('btnResetForm');
    const reviewModal = document.getElementById('reviewModal');
    const btnConfirmSubmit = document.getElementById('btnConfirmSubmit');
    const btnCloseModal = document.getElementById('btnCloseModal');

    // Sample Datasets for Fast Field Testing / Camps
    const normalSample = {
        patient_id: 'PT-NORM-' + Math.floor(1000 + Math.random() * 9000),
        patient_name: 'Aditi Sharma',
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

    const ckdSample = {
        patient_id: 'PT-EVAL-' + Math.floor(1000 + Math.random() * 9000),
        patient_name: 'Rajesh Verma',
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

    function populateForm(data) {
        for (const [key, val] of Object.entries(data)) {
            const el = form.elements[key];
            if (el) {
                el.value = val;
            }
        }
    }

    if (btnFillNormal) {
        btnFillNormal.addEventListener('click', () => {
            populateForm(normalSample);
        });
    }

    if (btnFillHighRisk) {
        btnFillHighRisk.addEventListener('click', () => {
            populateForm(ckdSample);
        });
    }

    if (btnReset) {
        btnReset.addEventListener('click', () => {
            form.reset();
        });
    }

    // Modal Confirmation Step
    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            // Client-side HTML5 constraint validation
            if (!form.checkValidity()) {
                form.reportValidity();
                return;
            }
            buildReviewSummary();
            if (reviewModal) {
                reviewModal.style.display = 'flex';
            }
        });
    }

    if (btnConfirmSubmit) {
        btnConfirmSubmit.addEventListener('click', () => {
            if (form) {
                btnConfirmSubmit.disabled = true;
                btnConfirmSubmit.innerHTML = '<span>Evaluating Laboratory Data...</span>';
                form.submit();
            }
        });
    }

    if (btnCloseModal) {
        btnCloseModal.addEventListener('click', () => {
            if (reviewModal) reviewModal.style.display = 'none';
        });
    }

    function buildReviewSummary() {
        const summaryContainer = document.getElementById('reviewSummaryContent');
        if (!summaryContainer) return;

        const formData = new FormData(form);
        const ptId = formData.get('patient_id') || 'Auto-assigned';
        const age = formData.get('age') || 'N/A';
        const model = formData.get('model_choice') || 'Random Forest';
        const bp = formData.get('bp') || 'N/A';
        const sc = formData.get('sc') || 'N/A';
        const hemo = formData.get('hemo') || 'N/A';
        const al = formData.get('al') || '0';

        summaryContainer.innerHTML = `
            <div style="background-color: var(--bg-surface-subtle); padding: 1rem; border-radius: 6px; margin-bottom: 1rem;">
                <p style="margin-bottom: 0.25rem;"><strong>Patient ID:</strong> ${ptId} &nbsp;|&nbsp; <strong>Age:</strong> ${age}</p>
                <p style="margin-bottom: 0.25rem;"><strong>Screening Model:</strong> ${model}</p>
                <p style="font-size: 0.85rem; color: var(--text-subtle);">Key Lab Samples: Creatinine: ${sc} mg/dL, Hemoglobin: ${hemo} g/dL, Albumin: Grade ${al}, BP: ${bp} mm/Hg</p>
            </div>
            <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.4;">
                Please verify that the routine blood and urine chemistry values match the patient laboratory report. The screening model will generate a calibrated risk evaluation.
            </p>
        `;
    }
}
