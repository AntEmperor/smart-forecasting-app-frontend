import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# --- Configuration ---
OUTPUT_FILENAME = 'nigeria_stlf_dataset.csv'
# The split date is the first moment of 2024. All data before is training.
SPLIT_DATE = '2024-01-01' 

# 1. Load the corrected, leakage-free dataset
print(f"Loading and processing '{OUTPUT_FILENAME}'...")
df_clean = pd.read_csv(OUTPUT_FILENAME)

# CRITICAL FIX: Set the 'datetime' column as the DataFrame index
df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])
df_clean = df_clean.set_index('datetime')

# 2. Define Target (y) and Features (X)
y = df_clean['load_kw']
# Exclude the target column from the features
feature_cols = [col for col in df_clean.columns if col != 'load_kw']
X = df_clean[feature_cols]

# 3. Perform the Temporal Split (4 Years Train, 1 Year Test)
X_train = X[X.index < SPLIT_DATE]
X_test = X[X.index >= SPLIT_DATE]
y_train = y[y.index < SPLIT_DATE]
y_test = y[y.index >= SPLIT_DATE]

print("\n--- Temporal Split Complete ---")
print(f"Training data end: {X_train.index.max()}")
print(f"Testing data start: {X_test.index.min()}")
print(f"Train size: {len(X_train)} samples. (2020-2023)")
print(f"Test size: {len(X_test)} samples. (2024)")