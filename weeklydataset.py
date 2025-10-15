import pandas as pd
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = 'smartgrid_5yr_weekly.csv'
OUTPUT_FILE = 'nigeria_weekly_dataset.csv'

# --- 1. Load and Clean Base Data ---
print(f"1. Loading and cleaning {INPUT_FILE}...")
df_weekly = pd.read_csv(INPUT_FILE)
df_weekly['timestamp'] = pd.to_datetime(df_weekly['timestamp'])
df_weekly = df_weekly.set_index('timestamp').sort_index()

# Drop flawed/redundant features
df_weekly = df_weekly.drop(columns=['base_load']) 

# --- 2. Temporal & Season Features ---
print("2. Engineering temporal and seasonal features...")
df_weekly['year'] = df_weekly.index.year
df_weekly['weekofyear'] = df_weekly.index.isocalendar().week.astype(int) # Critical weekly cycle feature
df_weekly['month'] = df_weekly.index.month

# Nigerian Season Definition (same logic as before)
df_weekly['is_rainy'] = df_weekly['month'].isin(range(4, 11)).astype(int)
df_weekly['is_harmattan'] = df_weekly['month'].isin([11, 12, 1]).astype(int)
df_weekly['is_dry'] = df_weekly['month'].isin([2, 3]).astype(int)

# --- 3. Leakage-Free Lag & Rolling Features ---
print("3. Engineering leakage-free lag and rolling features...")
# CRITICAL: Weekly lags
df_weekly['lag_1w'] = df_weekly['weekly_load_kwh'].shift(1)       # Load from last week
df_weekly['lag_4w'] = df_weekly['weekly_load_kwh'].shift(4)       # Load from last month
df_weekly['lag_52w'] = df_weekly['weekly_load_kwh'].shift(52)     # Load from same week last year

# Leakage-Free Rolling Mean (exclude the current week)
df_weekly['rolling_mean_4w'] = df_weekly['weekly_load_kwh'].rolling(window=4).mean().shift(1)

# --- 4. Climate Interaction Feature ---
# Weekly load correlation is strong with seasonal average temperature
df_weekly['dry_temp_interaction'] = df_weekly['avg_temp'] * df_weekly['is_dry']

# --- 5. Final Cleanup and Export ---
df_clean_weekly = df_weekly.dropna()

print("\n--- Weekly Dataset Readiness Check ---")
print(f"Clean rows ready for training: {len(df_clean_weekly)}.")
print(f"Exporting data to {OUTPUT_FILE}...")
df_clean_weekly.to_csv(OUTPUT_FILE)
print("SUCCESS: Weekly dataset ready for training.")