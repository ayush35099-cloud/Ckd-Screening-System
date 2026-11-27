import os
import urllib.request
import pandas as pd
import numpy as np
import json

BASE_URL = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/'
RAW_DIR = os.path.join(os.path.dirname(__file__), 'dataset', 'nhanes_raw')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
os.makedirs(RAW_DIR, exist_ok=True)

FILES = [
    'DEMO_J.xpt',
    'BIOPRO_J.xpt',
    'ALB_CR_J.xpt',
    'BPX_J.xpt',
    'BMX_J.xpt',
    'GHB_J.xpt',
    'CBC_J.xpt',
    'BPQ_J.xpt',
    'DIQ_J.xpt'
]

def download_nhanes():
    headers = {'User-Agent': 'Mozilla/5.0'}
    for f in FILES:
        dest = os.path.join(RAW_DIR, f)
        if os.path.exists(dest) and os.path.getsize(dest) > 50000:
            print(f"[NHANES] Exists: {f} ({os.path.getsize(dest):,} bytes)")
            continue
        url = BASE_URL + f
        print(f"[NHANES] Downloading {f} from {url}...")
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            with open(dest, 'wb') as out_f:
                out_f.write(data)
        print(f"[NHANES] Saved {f} ({len(data):,} bytes)")

def calculate_egfr_ckd_epi_2021(scr, age, is_female):
    """
    CKD-EPI 2021 Race-Free Creatinine Equation (Inker et al., NEJM 2021)
    eGFR = 142 * min(Scr/kappa, 1)**alpha * max(Scr/kappa, 1)**(-1.200) * (0.9938**Age) * (1.012 if female else 1.0)
    For female: kappa = 0.7, alpha = -0.241
    For male:   kappa = 0.9, alpha = -0.302
    """
    if pd.isna(scr) or pd.isna(age) or pd.isna(is_female):
        return np.nan
    scr = float(scr)
    age = float(age)
    if scr <= 0 or age < 18:
        return np.nan

    if is_female:
        kappa = 0.7
        alpha = -0.241
        gender_mult = 1.012
    else:
        kappa = 0.9
        alpha = -0.302
        gender_mult = 1.0

    scr_k = scr / kappa
    egfr = 142.0 * (min(scr_k, 1.0) ** alpha) * (max(scr_k, 1.0) ** (-1.200)) * (0.9938 ** age) * gender_mult
    return egfr

def audit_and_process():
    download_nhanes()
    print("\nReading SAS transport files...")
    dfs = {}
    for f in FILES:
        fpath = os.path.join(RAW_DIR, f)
        df_temp = pd.read_sas(fpath, format='xport')
        name = f.replace('.xpt', '')
        dfs[name] = df_temp
        print(f"  {name}: {df_temp.shape[0]:,} rows, {df_temp.shape[1]} columns")

    # Start merging on SEQN
    demo = dfs['DEMO_J']
    n_original_records = len(demo)
    merged = demo[['SEQN', 'RIDAGEYR', 'RIAGENDR']].copy()

    # Biochemistry
    biopro = dfs['BIOPRO_J']
    bio_cols = ['SEQN', 'LBXSCR', 'LBXSBU', 'LBXSTCA', 'LBXSTP', 'LBXSAL', 'LBXSGL', 'LBXSNAS', 'LBXSKSI', 'LBXSUA']
    bio_avail = [c for c in bio_cols if c in biopro.columns]
    merged = merged.merge(biopro[bio_avail], on='SEQN', how='left')

    # Albumin & Creatinine Urine
    alb_cr = dfs['ALB_CR_J']
    alb_cols = ['SEQN', 'URXUMA', 'URXUCR', 'URDACT']
    alb_avail = [c for c in alb_cols if c in alb_cr.columns]
    merged = merged.merge(alb_cr[alb_avail], on='SEQN', how='left')

    # Blood Pressure examination
    bpx = dfs['BPX_J']
    # Calculate mean SBP and DBP across up to 3 readings
    bpx_work = bpx[['SEQN']].copy()
    sbp_cols = [c for c in ['BPXSY1', 'BPXSY2', 'BPXSY3'] if c in bpx.columns]
    dbp_cols = [c for c in ['BPXDI1', 'BPXDI2', 'BPXDI3'] if c in bpx.columns]
    bpx_work['sbp'] = bpx[sbp_cols].mean(axis=1, skipna=True)
    bpx_work['dbp'] = bpx[dbp_cols].mean(axis=1, skipna=True)
    merged = merged.merge(bpx_work, on='SEQN', how='left')

    # Body measures
    bmx = dfs['BMX_J']
    bmx_cols = ['SEQN', 'BMXBMI', 'BMXWAIST']
    bmx_avail = [c for c in bmx_cols if c in bmx.columns]
    merged = merged.merge(bmx[bmx_avail], on='SEQN', how='left')

    # Glycohemoglobin
    ghb = dfs['GHB_J']
    if 'LBXGH' in ghb.columns:
        merged = merged.merge(ghb[['SEQN', 'LBXGH']], on='SEQN', how='left')

    # Complete Blood Count
    cbc = dfs['CBC_J']
    cbc_cols = ['SEQN', 'LBXHGB', 'LBXWBCSI', 'LBXRBCSI', 'LBXPLTSI']
    cbc_avail = [c for c in cbc_cols if c in cbc.columns]
    merged = merged.merge(cbc[cbc_avail], on='SEQN', how='left')

    # Questionnaire Hypertension & Diabetes
    bpq = dfs['BPQ_J']
    if 'BPQ020' in bpq.columns:
        # BPQ020: 1 = Yes, 2 = No
        bpq_clean = bpq[['SEQN', 'BPQ020']].copy()
        bpq_clean['htn'] = bpq_clean['BPQ020'].map({1.0: 'yes', 2.0: 'no'})
        merged = merged.merge(bpq_clean[['SEQN', 'htn']], on='SEQN', how='left')

    diq = dfs['DIQ_J']
    if 'DIQ010' in diq.columns:
        # DIQ010: 1 = Yes, 2 = No, 3 = Borderline
        diq_clean = diq[['SEQN', 'DIQ010']].copy()
        diq_clean['dm'] = diq_clean['DIQ010'].map({1.0: 'yes', 2.0: 'no', 3.0: 'borderline'})
        merged = merged.merge(diq_clean[['SEQN', 'dm']], on='SEQN', how='left')

    print(f"\nMerged dataset shape: {merged.shape[0]} rows, {merged.shape[1]} columns")

    # Filter to adults (age >= 18)
    adults = merged[merged['RIDAGEYR'] >= 18].copy()
    print(f"Adults (age >= 18): {len(adults):,} records")

    # Compute eGFR using CKD-EPI 2021
    # RIAGENDR: 1 = Male, 2 = Female
    is_female_series = adults['RIAGENDR'] == 2.0
    adults['is_female'] = is_female_series
    adults['egfr'] = adults.apply(
        lambda r: calculate_egfr_ckd_epi_2021(r['LBXSCR'], r['RIDAGEYR'], r['is_female']),
        axis=1
    )

    # Derived CKD indicator:
    # eGFR < 60 mL/min/1.73m2 OR UACR >= 30 mg/g
    # Note: Valid target evaluation requires at least one valid measurement (eGFR or UACR).
    # If both are missing, the target cannot be derived and must NOT be assumed normal.
    has_egfr = ~adults['egfr'].isna()
    has_uacr = ~adults['URDACT'].isna()

    # Records with both missing cannot be reliably labeled
    evaluable = adults[has_egfr | has_uacr].copy()
    print(f"Evaluable records (valid eGFR and/or UACR): {len(evaluable):,}")

    def derive_ckd_target(row):
        egfr_val = row['egfr']
        uacr_val = row['URDACT']
        # If either is positive, it's flagged as CKD indicator
        if (pd.notna(egfr_val) and egfr_val < 60.0) or (pd.notna(uacr_val) and uacr_val >= 30.0):
            return 1
        # If both are present and neither is positive, not flagged
        if pd.notna(egfr_val) and pd.notna(uacr_val):
            return 0
        # If only one is present and it is normal, what is KDIGO convention?
        # If eGFR >= 60 but UACR is unmeasured, albuminuria cannot be ruled out.
        # Conversely, if UACR < 30 but eGFR is unmeasured, reduced GFR cannot be ruled out.
        # To strictly prevent bias, require both eGFR and UACR for confirmed non-CKD baseline,
        # OR require both measurements present. Let's see counts:
        return np.nan

    evaluable['ckd_indicator'] = evaluable.apply(derive_ckd_target, axis=1)

    # Let's check records having both measurements
    complete_target = evaluable[~evaluable['ckd_indicator'].isna()].copy()
    print(f"Records with complete target confirmation: {len(complete_target):,}")
    print(f"Target breakdown:\n{complete_target['ckd_indicator'].value_counts(dropna=False)}")

    # Also check if we include records where eGFR is measured and UACR is measured
    both_measured = adults[has_egfr & has_uacr].copy()
    print(f"\nRecords with BOTH eGFR and UACR measured: {len(both_measured):,}")
    both_measured['ckd_indicator'] = (
        (both_measured['egfr'] < 60.0) | (both_measured['URDACT'] >= 30.0)
    ).astype(int)
    print("Class distribution for both_measured:")
    print(both_measured['ckd_indicator'].value_counts())
    print(f"CKD Indicator prevalence: {both_measured['ckd_indicator'].mean()*100:.2f}%")

    return both_measured

if __name__ == '__main__':
    audit_and_process()
