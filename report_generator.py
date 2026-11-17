import os
import io
import json
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

def generate_pdf_report(screening_data, feature_meta=None):
    """
    Generates a professional healthcare PDF screening report using ReportLab.
    Returns bytes buffer of the generated PDF.
    """
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

    # Styles
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=C_PRIMARY
    )

    tagline_style = ParagraphStyle(
        'DocTagline',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=C_MUTED
    )

    academic_header_style = ParagraphStyle(
        'AcademicHeader',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=C_MUTED,
        alignment=TA_RIGHT
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=C_PRIMARY,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=C_TEXT
    )

    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=C_MUTED,
        alignment=TA_JUSTIFY
    )

    # 1. Header Table (Brand on left, Academic metadata on right)
    brand_p = Paragraph("<b>RenalCare</b> &nbsp;<font size=9 color='#4D7260'>| CKD Screening Platform</font><br/><font size=8 color='#5A655F'>Smarter Screening. Earlier Awareness.</font>", title_style)
    academic_p = Paragraph(
        "<b>Dayananda Sagar College of Engineering, Bengaluru</b><br/>"
        "Dept. of AI & Machine Learning | BCP308 Community Project<br/>"
        "Team A-12: A. Choudhary, H. Kumawat, K. Arora, K. Singh",
        academic_header_style
    )

    header_table = Table([[brand_p, academic_p]], colWidths=[3.6 * inch, 3.8 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_PRIMARY, spaceBefore=2, spaceAfter=8))

    # 2. Patient Demographics & Report Meta Grid
    pt_id = screening_data.get('patient_id', 'Unknown')
    pt_name = screening_data.get('patient_name') or 'Not Recorded'
    pt_age = f"{screening_data.get('age', 'N/A')} yrs"
    pt_sex = screening_data.get('sex', 'Not Specified')
    report_date = screening_data.get('formatted_date') or datetime.now().strftime('%b %d, %Y, %I:%M %p')
    screening_uuid = screening_data.get('screening_uuid', 'N/A')[:18] + '...'

    pt_info_data = [
        [Paragraph("<b>Patient ID:</b>", body_style), Paragraph(str(pt_id), body_bold),
         Paragraph("<b>Screening Date:</b>", body_style), Paragraph(str(report_date), body_style)],
        [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(str(pt_name), body_style),
         Paragraph("<b>Record Reference:</b>", body_style), Paragraph(str(screening_uuid), body_style)],
        [Paragraph("<b>Age / Sex:</b>", body_style), Paragraph(f"{pt_age} / {pt_sex}", body_style),
         Paragraph("<b>Evaluation Model:</b>", body_style), Paragraph(str(screening_data.get('model_used', 'Random Forest')), body_style)]
    ]
    pt_table = Table(pt_info_data, colWidths=[1.3 * inch, 2.4 * inch, 1.4 * inch, 2.3 * inch])
    pt_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#ECEEEC')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(pt_table)
    story.append(Spacer(1, 10))

    # 3. Screening Outcome Banner
    is_flagged = screening_data.get('prediction_code') == 'ckd' or 'Detected' in screening_data.get('prediction_label', '')
    result_text = screening_data.get('prediction_label', 'CKD Indicators Detected' if is_flagged else 'CKD Indicators Not Detected')
    prob_val = screening_data.get('probability', 0.0)
    risk_level = screening_data.get('risk_level', 'Moderate' if is_flagged else 'Low')

    banner_bg = colors.HexColor('#FEF2F2') if is_flagged else colors.HexColor('#F0FDF4')
    banner_border = colors.HexColor('#F87171') if is_flagged else colors.HexColor('#86EFAC')
    banner_text_color = C_ALERT if is_flagged else C_SUCCESS

    outcome_p1 = Paragraph(
        f"<font size=12 color='{banner_text_color.hexval()}'><b>RESULT: {result_text.upper()}</b></font><br/>"
        f"<font size=8.5 color='#334155'>Clinical Risk Tier: <b>{risk_level.upper()}</b> &nbsp;|&nbsp; Calibrated Probability: <b>{prob_val}%</b></font>",
        styles['Normal']
    )
    outcome_table = Table([[outcome_p1]], colWidths=[7.4 * inch])
    outcome_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), banner_bg),
        ('BOX', (0,0), (-1,-1), 1.2, banner_border),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(outcome_table)
    story.append(Spacer(1, 10))

    # 4. Clinical Interpretation & Recommendations
    rec_text = screening_data.get('recommendation', '')
    if not rec_text:
        if is_flagged:
            rec_text = "The machine learning screening pipeline has detected markers indicative of Chronic Kidney Disease. Urgent clinical correlation is advised: arrange formal laboratory evaluation including eGFR, 24-hour urinary protein quantification, serum electrolytes, and renal ultrasonography."
        else:
            rec_text = "Screening indicators show values within non-CKD baseline ranges. Annual wellness monitoring is advised."

    story.append(Paragraph("Clinical Screening Guidance & Next Steps", section_heading))
    story.append(Paragraph(rec_text, body_style))
    story.append(Spacer(1, 8))

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
            finding_rows.append([
                Paragraph(f.get('parameter', ''), body_bold),
                Paragraph(str(f.get('value', '')), body_style),
                Paragraph(f"<font color='{C_ALERT.hexval()}'><b>{f.get('status', '')}</b></font>", body_style),
                Paragraph(f.get('detail', ''), body_style),
            ])
        f_table = Table(finding_rows, colWidths=[1.8 * inch, 1.2 * inch, 1.4 * inch, 3.0 * inch])
        f_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
            ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('LEFTPADDING', (0,0), (-1,-1), 5),
            ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(f_table)
        story.append(Spacer(1, 10))

    # 6. Complete Parameters Table (Clean 2-column or 3-column breakdown)
    story.append(Paragraph("Recorded Laboratory Parameters (UCI Standard 24-Feature Schema)", section_heading))
    raw_params = screening_data.get('parameters', {})

    # Feature definitions for reporting
    param_meta = {
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

    # Split into 2 columns for compact, clean layout
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
            val1 = 'Not Tested'

        k2, (name2, ref2) = items[i + half] if (i + half) < len(items) else (None, ('', ''))
        val2 = raw_params.get(k2, 'N/A') if k2 else ''
        if val2 is None or val2 == '' or str(val2).lower() == 'nan':
            val2 = 'Not Tested'

        table_rows.append([
            Paragraph(name1, body_style), Paragraph(str(val1), body_bold), Paragraph(f"<font size=7.5 color='#5A655F'>{ref1}</font>", body_style),
            Paragraph(name2, body_style), Paragraph(str(val2), body_bold), Paragraph(f"<font size=7.5 color='#5A655F'>{ref2}</font>", body_style)
        ])

    param_table = Table(table_rows, colWidths=[1.4 * inch, 0.8 * inch, 1.4 * inch, 1.5 * inch, 0.8 * inch, 1.5 * inch])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#EEF0EE')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(param_table)
    story.append(Spacer(1, 10))

    # 7. Model Performance & Legal Disclaimer Footer
    disclaimer_block = [
        Paragraph(
            "<b>IMPORTANT REGULATORY AND MEDICAL DISCLAIMER:</b><br/>"
            "This document is an automated screening and risk stratification report produced by an educational machine learning model "
            "trained on the UCI Chronic Kidney Disease benchmark dataset. <b>This report does NOT constitute a formal medical diagnosis, "
            "clinical certification, or therapeutic prescription.</b> Kidney disease pathology is multifactorial and requires comprehensive "
            "investigation by a licensed medical practitioner. Any elevated or abnormal findings must be confirmed via definitive nephrological "
            "investigation (e.g., inulin clearance, formal estimated GFR, renal biopsy, renal ultrasonography).",
            disclaimer_style
        ),
        Spacer(1, 6),
        Paragraph(
            "<b>Model Specification:</b> Random Forest Classifier / Decision Tree Pipeline &bull; "
            "<b>Held-out Test Sensitivity:</b> 100.0% &bull; <b>Specificity:</b> 100.0% &bull; "
            "<b>Institutional Affiliation:</b> Department of AIML, Dayananda Sagar College of Engineering, Bengaluru.",
            ParagraphStyle('DocFooterMeta', parent=styles['Normal'], fontSize=7.5, textColor=C_MUTED)
        )
    ]

    disclaimer_table = Table([[disclaimer_block]], colWidths=[7.4 * inch])
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(KeepTogether(disclaimer_table))

    # Build document
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
