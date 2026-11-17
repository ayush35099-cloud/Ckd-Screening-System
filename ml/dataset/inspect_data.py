import pandas as pd
import numpy as np
import os

arff_path = os.path.join(os.path.dirname(__file__), 'Chronic_Kidney_Disease', 'chronic_kidney_disease.arff')

with open(arff_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

data_started = False
data_rows = []
attributes = []

for line in lines:
    line = line.strip()
    if not line or line.startswith('%'):
        continue
    if line.lower().startswith('@attribute'):
        parts = line.split()
        attr_name = parts[1].replace("'", "").strip()
        attributes.append(attr_name)
    elif line.lower().startswith('@data'):
        data_started = True
    elif data_started:
        # Strip trailing commas or whitespace
        cleaned_line = line.rstrip(',').strip()
        # Handle known typo in UCI original file where line 398 has 'no,,no,no'
        if ',,' in cleaned_line:
            cleaned_line = cleaned_line.replace(',,', ',')
        vals = [v.strip().replace('\t', '') for v in cleaned_line.split(',')]
        # If there are empty values, treat empty as '?'
        vals = [('?' if v == '' else v) for v in vals]
        if len(vals) == len(attributes):
            data_rows.append(vals)
        else:
            print(f"Row length mismatch: expected {len(attributes)}, got {len(vals)}: {line}")

print(f"Attributes ({len(attributes)}):", attributes)
print(f"Total data rows parsed: {len(data_rows)}")

df = pd.DataFrame(data_rows, columns=attributes)

# Standardize column names (aliasing wbcc -> wc, rbcc -> rc for consistency with info.txt and project spec)
rename_map = {
    'wbcc': 'wc',
    'rbcc': 'rc'
}
df.rename(columns=rename_map, inplace=True)

# Replace '?' with np.nan
df.replace('?', np.nan, inplace=True)

# Check target column
# In UCI dataset, 'ckd\t' or spaces might exist
df['class'] = df['class'].str.strip()
print("\nTarget class counts:")
print(df['class'].value_counts(dropna=False))

# Standardize categorical columns where typos exist in raw UCI dataset
# e.g., 'dm': '\tno', ' yes', '\tyes'
categorical_cols = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip().replace({'nan': np.nan})

# Specific known corrections in UCI dataset:
df['dm'] = df['dm'].replace({'yes': 'yes', 'no': 'no', ' yes': 'yes', '\tyes': 'yes', '\tno': 'no'})
df['cad'] = df['cad'].replace({'no': 'no', 'yes': 'yes', '\tno': 'no'})
df['class'] = df['class'].replace({'ckd': 'ckd', 'notckd': 'notckd', 'ckd\t': 'ckd'})

# Numerical columns conversion
numerical_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nDataset Summary:")
print(f"Shape: {df.shape}")
print("\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

output_csv = os.path.join(os.path.dirname(__file__), 'chronic_kidney_disease_cleaned.csv')
df.to_csv(output_csv, index=False)
print(f"\nSaved cleaned dataset to: {output_csv}")
