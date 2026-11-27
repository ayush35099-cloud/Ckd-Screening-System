import os
import io
import html
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Language Metadata (Kept for external module reference compatibility)
LANG_NAMES = {
    'en': 'English',
    'hi': 'Hindi (हिन्दी)',
    'bn': 'Bengali (বাংলা)',
    'mr': 'Marathi (मराठी)',
    'te': 'Telugu (తెలుగు)',
    'ta': 'Tamil (தமிழ்)',
    'ur': 'Urdu (اردو)',
    'gu': 'Gujarati (ગુજરાતી)',
}

_FONTS_REGISTERED = False
_FONT_REGULAR = 'Helvetica'
_FONT_BOLD = 'Helvetica-Bold'


def _safe_xml(val):
    """Safely escapes text for ReportLab Paragraph XML parsing without double-escaping."""
    if val is None:
        return ''
    s = str(val)
    return html.escape(s, quote=True)


def _register_report_fonts():
    """
    Registers reliable TrueType fonts with robust English typography and Unicode support.
    Falls back gracefully to standard Helvetica and Helvetica-Bold Type 1 fonts.
    """
    global _FONTS_REGISTERED, _FONT_REGULAR, _FONT_BOLD
    if _FONTS_REGISTERED:
        return _FONT_REGULAR, _FONT_BOLD

    import reportlab
    rl_fonts = os.path.join(os.path.dirname(reportlab.__file__), 'fonts')
    vera_reg = os.path.join(rl_fonts, 'Vera.ttf')
    vera_bold = os.path.join(rl_fonts, 'VeraBd.ttf')

    candidates = [
        # Windows TrueType fonts with comprehensive Unicode coverage
        ('C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/arialbd.ttf'),
        ('C:/Windows/Fonts/segoeui.ttf', 'C:/Windows/Fonts/segoeuib.ttf'),
        # Linux standard system TrueType fonts (Render / Ubuntu / Debian)
        ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
        ('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'),
        # Built-in ReportLab TrueType fonts
        (vera_reg, vera_bold)
    ]

    for reg_path, bold_path in candidates:
        if os.path.exists(reg_path):
            try:
                pdfmetrics.registerFont(TTFont('RenalSans', reg_path))
                if os.path.exists(bold_path):
                    pdfmetrics.registerFont(TTFont('RenalSans-Bold', bold_path))
                else:
                    pdfmetrics.registerFont(TTFont('RenalSans-Bold', reg_path))
                _FONT_REGULAR = 'RenalSans'
                _FONT_BOLD = 'RenalSans-Bold'
                _FONTS_REGISTERED = True
                return _FONT_REGULAR, _FONT_BOLD
            except Exception:
                continue

    _FONT_REGULAR = 'Helvetica'
    _FONT_BOLD = 'Helvetica-Bold'
    _FONTS_REGISTERED = True
    return _FONT_REGULAR, _FONT_BOLD


def generate_pdf_report(screening_data, feature_meta=None, lang='en'):
    """
    Generates a professional healthcare PDF screening report using ReportLab.
    
    IMPORTANT: All PDF screening reports are MANDATORILY generated in clean, professional ENGLISH,
    irrespective of website language, user locale, or cookies. This completely eliminates
    missing glyphs, boxes, corrupted strings, and font fragmentation.
    
    Returns bytes buffer of the generated PDF.
    """
    font_reg, font_bold = _register_report_fonts()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    story = []

    # Palette
    C_PRIMARY = colors.HexColor('#174D3A')      # Deep Forest Green
    C_SECONDARY = colors.HexColor('#4D7260')    # Muted Sage Green
    C_TEXT = colors.HexColor('#1A201C')         # Dark Charcoal
    C_MUTED = colors.HexColor('#5A655F')        # Medium Slate Grey
    C_LIGHT_BG = colors.HexColor('#F8F7F4')     # Warm Off-white
    C_BORDER = colors.HexColor('#DDE2DD')       # Soft Neutral Border
    C_ALERT = colors.HexColor('#B91C1C')        # Deep Red/Amber
    C_SUCCESS = colors.HexColor('#15803D')      # Deep Emerald

    styles = getSampleStyleSheet()

    # Configure styles to consistently use registered font family
    normal_style = styles['Normal']
    normal_style.fontName = font_reg
    normal_style.fontSize = 8.5
    normal_style.leading = 11.5
    normal_style.textColor = C_TEXT

    title_style = ParagraphStyle(
        'DocTitle',
        parent=normal_style,
        fontName=font_bold,
        fontSize=18,
        leading=22,
        textColor=C_PRIMARY
    )

    academic_header_style = ParagraphStyle(
        'AcademicHeader',
        parent=normal_style,
        fontName=font_reg,
        fontSize=8,
        leading=11,
        textColor=C_MUTED,
        alignment=TA_RIGHT
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=normal_style,
        fontName=font_bold,
        fontSize=10.5,
        leading=13,
        textColor=C_PRIMARY,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=normal_style,
        fontName=font_reg,
        fontSize=8.5,
        leading=11.5,
        textColor=C_TEXT
    )

    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName=font_bold
    )

    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=normal_style,
        fontName=font_reg,
        fontSize=7,
        leading=9.5,
        textColor=C_MUTED,
        alignment=TA_JUSTIFY
    )

    # 1. Header Table (Brand on left, Academic metadata on right)
    brand_p = Paragraph(
        "<b>RenalCare</b> &nbsp;<font size=9 color='#4D7260'>| Clinical Decision Support</font><br/>"
        "<font size=7.5 color='#5A655F'>Smarter Screening. Earlier Awareness.</font>",
        title_style
    )
    academic_p = Paragraph(
        "<b>Dayananda Sagar College of Engineering, Bengaluru</b><br/>"
        "Dept. of AI & Machine Learning | BCP308 Community Project<br/>"
        "Team A-12: A. Choudhary, H. Kumawat, K. Arora, K. Singh",
        academic_header_style
    )

    header_table = Table([[brand_p, academic_p]], colWidths=[3.7 * inch, 3.7 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_PRIMARY, spaceBefore=2, spaceAfter=6))

    # 2. Patient Demographics & Report Meta Grid
    pt_id = _safe_xml(screening_data.get('patient_id') or 'Unknown')
    pt_name = _safe_xml(screening_data.get('patient_name') or 'Not Recorded')
    pt_age = _safe_xml(f"{screening_data.get('age', 'N/A')} yrs")
    pt_sex = _safe_xml(screening_data.get('sex') or 'Not Specified')
    report_date = _safe_xml(screening_data.get('formatted_date') or datetime.now().strftime('%b %d, %Y, %I:%M %p'))
    screening_uuid = _safe_xml(screening_data.get('screening_uuid', 'N/A')[:18] + '...')

    is_nhanes = (
        screening_data.get('model_id') in ['model_2', 'nhanes', 'expanded'] or
        'nhanes' in str(screening_data.get('model_used', '')).lower() or
        'expanded' in str(screening_data.get('model_used', '')).lower()
    )

    pipeline_label = "CDC NHANES (Expanded Survey)" if is_nhanes else "UCI CKD (Standard Clinical)"
    model_name_eval = _safe_xml(screening_data.get('model_used') or 'Random Forest')

    pt_info_data = [
        [Paragraph("<b>Patient Identifier:</b>", body_style), Paragraph(str(pt_id), body_bold),
         Paragraph("<b>Screening Date:</b>", body_style), Paragraph(str(report_date), body_style)],
        [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(str(pt_name), body_style),
         Paragraph("<b>Screening ID:</b>", body_style), Paragraph(str(screening_uuid), body_style)],
        [Paragraph("<b>Age / Sex:</b>", body_style), Paragraph(f"{pt_age} / {pt_sex}", body_style),
         Paragraph("<b>Model Evaluated:</b>", body_style), Paragraph(str(model_name_eval), body_style)],
        [Paragraph("<b>Report Language:</b>", body_style), Paragraph("English", body_style),
         Paragraph("<b>Screening Pipeline:</b>", body_style), Paragraph(str(pipeline_label), body_style)]
    ]
    pt_table = Table(pt_info_data, colWidths=[1.4 * inch, 2.3 * inch, 1.4 * inch, 2.3 * inch])
    pt_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#ECEEEC')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(pt_table)
    story.append(Spacer(1, 8))

    # 3. Screening Outcome Banner (Fixed Professional English Nuanced Wording)
    is_flagged = screening_data.get('prediction_code') == 'ckd' or 'Detected' in str(screening_data.get('prediction_label', ''))
    if is_flagged:
        result_text = "CKD INDICATORS DETECTED — Further Clinical Evaluation Recommended"
        banner_bg = colors.HexColor('#FEF2F2')
        banner_border = colors.HexColor('#F87171')
        banner_text_color = C_ALERT
    else:
        result_text = "CKD INDICATORS NOT DETECTED BY THIS MODEL — Does Not Rule Out CKD"
        banner_bg = colors.HexColor('#F0FDF4')
        banner_border = colors.HexColor('#86EFAC')
        banner_text_color = C_SUCCESS

    raw_prob = float(screening_data.get('probability', 0.0))
    conf_val = screening_data.get('confidence')
    if conf_val is not None:
        confidence = round(float(conf_val), 1)
    else:
        confidence = round(raw_prob if is_flagged else (100.0 - raw_prob), 1)

    prob_ckd = round(raw_prob, 1)
    risk_level = _safe_xml(screening_data.get('risk_level', 'Moderate' if is_flagged else 'Low'))
    model_name = _safe_xml(screening_data.get('model_used', 'Random Forest'))

    outcome_p = Paragraph(
        f"<font size=10 color='{banner_text_color.hexval()}'><b>RESULT: {result_text}</b></font><br/>"
        f"<font size=8 color='#334155'>Statistical Stratification: <b>{risk_level}</b> &nbsp;|&nbsp; "
        f"Primary Evaluator: <b>{model_name}</b> &nbsp;|&nbsp; "
        f"Model Confidence: <b>{confidence}%</b> &nbsp;|&nbsp; "
        f"Statistical Indication P(CKD): <b>{prob_ckd}%</b></font>",
        body_style
    )
    outcome_table = Table([[outcome_p]], colWidths=[7.4 * inch])
    outcome_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), banner_bg),
        ('BOX', (0,0), (-1,-1), 1.0, banner_border),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(outcome_table)
    story.append(Spacer(1, 8))

    # Dual-Model Comparative Cross-Validation Section (if comparison data present)
    comparison = screening_data.get('comparison') or {}
    rf_comp = comparison.get('random_forest')
    dt_comp = comparison.get('decision_tree')
    if rf_comp and dt_comp:
        consensus_text = _safe_xml(comparison.get('consensus_label', 'Complete Consensus'))
        is_consensus = comparison.get('agreement', True)
        status_color = C_SUCCESS if is_consensus else C_ALERT

        dual_head = Paragraph(
            f"<b>Dual-Model Comparative Cross-Validation</b> &nbsp;&bull;&nbsp; "
            f"<font color='{status_color.hexval()}'><b>Status: {consensus_text}</b></font>",
            section_heading
        )
        story.append(dual_head)

        dual_rows = [
            [Paragraph("<b>Evaluation Pipeline</b>", body_style),
             Paragraph("<b>Predicted Classification</b>", body_style),
             Paragraph("<b>Model Confidence</b>", body_style),
             Paragraph("<b>Statistical P(CKD)</b>", body_style),
             Paragraph("<b>Benchmark Role</b>", body_style)]
        ]

        # Random Forest Row
        dual_rows.append([
            Paragraph("<b>Random Forest</b>", body_bold),
            Paragraph("Flagged (Indicators Detected)" if rf_comp.get('is_flagged') else "Non-Flagged (Baseline)", body_style),
            Paragraph(f"<b>{rf_comp.get('confidence', 0.0)}%</b>", body_style),
            Paragraph(f"{rf_comp.get('probability_ckd', 0.0)}%", body_style),
            Paragraph("Primary Champion Ensemble (100% test sensitivity)", body_style)
        ])

        # Decision Tree Row
        dual_rows.append([
            Paragraph("<b>Decision Tree</b>", body_bold),
            Paragraph("Flagged (Indicators Detected)" if dt_comp.get('is_flagged') else "Non-Flagged (Baseline)", body_style),
            Paragraph(f"<b>{dt_comp.get('confidence', 0.0)}%</b>", body_style),
            Paragraph(f"{dt_comp.get('probability_ckd', 0.0)}%", body_style),
            Paragraph("Comparative Tree Classifier (94% test sensitivity)", body_style)
        ])

        dual_table = Table(dual_rows, colWidths=[1.5 * inch, 1.8 * inch, 1.1 * inch, 1.1 * inch, 1.9 * inch])
        dual_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
            ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('LEFTPADDING', (0,0), (-1,-1), 5),
            ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(dual_table)
        story.append(Spacer(1, 8))

    # Missing Values Warning (if any)
    missing_params = screening_data.get('missing_parameters') or []
    if missing_params:
        missing_list = _safe_xml(', '.join(str(p) for p in missing_params))
        missing_text = (
            f"<b>Missing Information Notice:</b> {len(missing_params)} routine parameter(s) were unmeasured in this screening "
            f"({missing_list}). Statistical imputation was applied based on training data distributions. "
            f"Missing laboratory values can impact screening sensitivity."
        )
        missing_p = Paragraph(f"<font size=7.5 color='#92400E'>{missing_text}</font>", body_style)
        missing_table = Table([[missing_p]], colWidths=[7.4 * inch])
        missing_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FFFBEB')),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#FDE68A')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(missing_table)
        story.append(Spacer(1, 6))

    # 4. Clinical Guidance & Recommendations (Fixed Professional English)
    rec_text = screening_data.get('recommendation', '')
    if not rec_text or not isinstance(rec_text, str) or len(rec_text.strip()) < 10:
        if is_flagged:
            rec_text = (
                "The machine learning screening pipeline has detected markers indicative of Chronic Kidney Disease. "
                "Further clinical correlation by a medical professional or nephrologist is recommended, including formal "
                "diagnostic tests (eGFR, 24-hour urine collection, and renal ultrasound)."
            )
        else:
            rec_text = (
                "Routine screening indicators show values within non-CKD baseline ranges. Note that this result does not "
                "rule out Chronic Kidney Disease or other renal disorders."
            )
    else:
        # If recommendation was passed in English, keep it; if not standard, use clear English
        rec_text = _safe_xml(rec_text)

    story.append(Paragraph("Clinical Screening Guidance & Next Steps", section_heading))
    story.append(Paragraph(rec_text, body_style))
    story.append(Spacer(1, 6))

    # 5. Key Abnormal Findings (if any)
    findings = screening_data.get('findings') or []
    if findings:
        story.append(Paragraph("Identified Laboratory Anomalies", section_heading))
        finding_rows = [
            [Paragraph("<b>Parameter</b>", body_style),
             Paragraph("<b>Recorded Value</b>", body_style),
             Paragraph("<b>Status</b>", body_style),
             Paragraph("<b>Clinical Correlation</b>", body_style)]
        ]
        for f in findings:
            param_lbl = _safe_xml(f.get('parameter', ''))
            param_val = _safe_xml(str(f.get('value', '')))
            param_stat = _safe_xml(f.get('status', ''))
            param_detail = _safe_xml(f.get('detail', ''))
            finding_rows.append([
                Paragraph(param_lbl, body_bold),
                Paragraph(param_val, body_style),
                Paragraph(f"<font color='{C_ALERT.hexval()}'><b>{param_stat}</b></font>", body_style),
                Paragraph(param_detail, body_style),
            ])
        f_table = Table(finding_rows, colWidths=[1.8 * inch, 1.2 * inch, 1.4 * inch, 3.0 * inch])
        f_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
            ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0,0), (-1,-1), 2.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
            ('LEFTPADDING', (0,0), (-1,-1), 5),
            ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(f_table)
        story.append(Spacer(1, 8))

    # 6. Complete Parameters Table (Clean 2-column breakdown)
    raw_params = screening_data.get('parameters') or {}

    if is_nhanes:
        story.append(Paragraph("Submitted Laboratory Parameters (CDC NHANES 19-Feature Schema)", section_heading))
        param_meta = {
            'age': ('Age', '18 - 85 yrs'),
            'sex': ('Biological Sex', 'male / female'),
            'bmi': ('Body Mass Index (BMI)', '18.5 - 24.9 kg/m²'),
            'sbp': ('Systolic Blood Pressure', '90 - 120 mmHg'),
            'dbp': ('Diastolic Blood Pressure', '60 - 80 mmHg'),
            'bgr': ('Serum Glucose', '70 - 99 mg/dL'),
            'hba1c': ('Glycohemoglobin (HbA1c)', '4.0 - 5.6%'),
            'bu': ('Blood Urea Nitrogen (BUN)', '7 - 20 mg/dL'),
            'sc': ('Serum Creatinine', '0.6 - 1.2 mg/dL'),
            'sod': ('Serum Sodium', '135 - 145 mmol/L'),
            'pot': ('Serum Potassium', '3.5 - 5.0 mmol/L'),
            'tot_protein': ('Total Serum Protein', '6.0 - 8.3 g/dL'),
            'serum_alb': ('Serum Albumin', '3.5 - 5.0 g/dL'),
            'uric_acid': ('Serum Uric Acid', '3.5 - 7.2 mg/dL'),
            'hemo': ('Hemoglobin', '12.0 - 17.0 g/dL'),
            'pcv': ('Hematocrit / PCV', '37 - 50%'),
            'wc': ('White Blood Cell Count', '4.0 - 11.0 10³/µL'),
            'htn': ('Hypertension History', 'no'),
            'dm': ('Diabetes History', 'no')
        }
    else:
        story.append(Paragraph("Submitted Laboratory Parameters (UCI Standard 24-Feature Schema)", section_heading))
        param_meta = {
            'age': ('Age', '18 - 65 yrs'),
            'bp': ('Blood Pressure', '70 - 80 mm/Hg'),
            'sg': ('Specific Gravity', '1.015 - 1.025'),
            'al': ('Albumin (Protein)', '0 (Negative)'),
            'su': ('Sugar (Glucose)', '0 (Negative)'),
            'bgr': ('Blood Glucose Random', '70 - 140 mg/dL'),
            'bu': ('Blood Urea', '15 - 45 mg/dL'),
            'sc': ('Serum Creatinine', '0.6 - 1.2 mg/dL'),
            'sod': ('Sodium', '135 - 145 mEq/L'),
            'pot': ('Potassium', '3.5 - 5.0 mEq/L'),
            'hemo': ('Hemoglobin', '13.0 - 17.0 g/dL'),
            'pcv': ('Packed Cell Volume', '38 - 50%'),
            'wc': ('White Blood Cell Count', '4,000 - 11,000 /cumm'),
            'rc': ('Red Blood Cell Count', '4.2 - 5.9 M/cmm'),
            'rbc': ('Red Blood Cells (Urine)', 'normal'),
            'pc': ('Pus Cells (Urine)', 'normal'),
            'pcc': ('Pus Cell Clumps', 'notpresent'),
            'ba': ('Bacteria', 'notpresent'),
            'htn': ('Hypertension', 'no'),
            'dm': ('Diabetes Mellitus', 'no'),
            'cad': ('Coronary Artery Disease', 'no'),
            'appet': ('Appetite', 'good'),
            'pe': ('Pedal Edema', 'no'),
            'ane': ('Anemia', 'no')
        }

    items = list(param_meta.items())
    half = (len(items) + 1) // 2

    table_rows = [
        [Paragraph("<b>Parameter</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Ref Range</b>", body_style),
         Paragraph("<b>Parameter</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Ref Range</b>", body_style)]
    ]

    for i in range(half):
        k1, (name1, ref1) = items[i]
        val1 = raw_params.get(k1, 'N/A')
        if val1 is None or val1 == '' or str(val1).lower() == 'nan':
            val1 = 'Unmeasured'

        k2, (name2, ref2) = items[i + half] if (i + half) < len(items) else (None, ('', ''))
        val2 = raw_params.get(k2, 'N/A') if k2 else ''
        if val2 is None or val2 == '' or str(val2).lower() == 'nan':
            val2 = 'Unmeasured'

        table_rows.append([
            Paragraph(str(name1), body_style), Paragraph(_safe_xml(str(val1)), body_bold), Paragraph(f"<font size=7 color='#5A655F'>{ref1}</font>", body_style),
            Paragraph(str(name2), body_style), Paragraph(_safe_xml(str(val2)), body_bold), Paragraph(f"<font size=7 color='#5A655F'>{ref2}</font>", body_style)
        ])

    param_table = Table(table_rows, colWidths=[1.4 * inch, 0.8 * inch, 1.4 * inch, 1.5 * inch, 0.8 * inch, 1.5 * inch])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#EEF0EE')),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(param_table)
    story.append(Spacer(1, 8))

    # 7. Model Performance & Regulatory Disclaimer Footer (Fixed Professional English)
    if is_nhanes:
        disclaimer_text = (
            "<b>IMPORTANT REGULATORY AND MEDICAL DISCLAIMER:</b><br/>"
            "This document is an automated decision-support screening report produced by an educational machine learning model "
            "trained on official CDC NHANES (National Health and Nutrition Examination Survey) cross-sectional data (N=5,063). "
            "Derived indicators are based on KDIGO criteria (eGFR &lt; 60 mL/min/1.73m² or UACR &ge; 30 mg/g). "
            "Because cross-sectional survey data cannot verify 3-month chronicity, <b>this report does NOT constitute a clinical diagnosis, "
            "medical certificate, or treatment prescription.</b> The model does not definitively confirm or rule out kidney disease. "
            "Comprehensive diagnostic follow-up by a physician or nephrologist is strongly recommended."
        )
        spec_text = (
            "<b>Model Specification:</b> Expanded CKD Pipeline (CDC NHANES 2017-2018) &bull; "
            "<b>Population Benchmark Testing:</b> 82.1% ROC-AUC on held-out test split (N=1,013) &bull; "
            "<b>Institutional Affiliation:</b> Dept. of AIML, Dayananda Sagar College of Engineering, Bengaluru."
        )
    else:
        disclaimer_text = (
            "<b>IMPORTANT REGULATORY AND MEDICAL DISCLAIMER:</b><br/>"
            "This document is an automated decision-support screening report produced by an educational machine learning model "
            "trained on the UCI Chronic Kidney Disease benchmark dataset. <b>This report does NOT constitute a clinical diagnosis, "
            "medical certificate, or treatment prescription.</b> The model does not diagnose or definitively rule out kidney disease. "
            "Kidney disease is multifactorial; any abnormal indicator must be investigated through comprehensive clinical nephrological evaluation "
            "(e.g. formal eGFR calculation, inulin clearance, renal ultrasound). Independent prospective clinical validation has not been performed."
        )
        spec_text = (
            "<b>Model Specification:</b> Random Forest / Decision Tree Pipeline &bull; "
            "<b>Academic Benchmark Testing:</b> 100.0% sensitivity on held-out UCI test split (N=80) &bull; "
            "<b>Institutional Affiliation:</b> Dept. of AIML, Dayananda Sagar College of Engineering, Bengaluru."
        )

    disclaimer_block = [
        Paragraph(disclaimer_text, disclaimer_style),
        Spacer(1, 4),
        Paragraph(spec_text, ParagraphStyle('DocFooterMeta', parent=normal_style, fontSize=7, textColor=C_MUTED))
    ]

    disclaimer_table = Table([[disclaimer_block]], colWidths=[7.4 * inch])
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(KeepTogether(disclaimer_table))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
