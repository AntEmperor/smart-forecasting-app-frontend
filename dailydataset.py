import pandas as pd
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = 'smartgrid_5yr_daily.csv'
OUTPUT_FILE = 'nigeria_daily_dataset.csv'
SPLIT_DATE = '2024-01-01'

# --- 1. Load and Clean Base Data ---
print(f"1. Loading and cleaning {INPUT_FILE}...")
df_daily = pd.read_csv(INPUT_FILE)
df_daily['timestamp'] = pd.to_datetime(df_daily['timestamp'])
df_daily = df_daily.set_index('timestamp').sort_index()

# Drop flawed/redundant features
df_daily = df_daily.drop(columns=['base_load']) 

# --- 2. Temporal & Season Features ---
print("2. Engineering temporal and seasonal features...")
df_daily['dayofweek'] = df_daily.index.dayofweek # 0=Monday, 6=Sunday
df_daily['dayofyear'] = df_daily.index.dayofyear
df_daily['month'] = df_daily.index.month
df_daily['is_weekend'] = (df_daily['dayofweek'] >= 5).astype(int)

# Nigerian Season Definition (same logic as hourly)
df_daily['is_rainy'] = df_daily['month'].isin(range(4, 11)).astype(int)       # April - October
df_daily['is_harmattan'] = df_daily['month'].isin([11, 12, 1]).astype(int)     # November - January
df_daily['is_dry'] = df_daily['month'].isin([2, 3]).astype(int)               # February - March

# --- 3. Leakage-Free Lag & Rolling Features ---
print("3. Engineering leakage-free lag and rolling features...")
# CRITICAL: Use .shift(N) to look only at past days
df_daily['lag_1d'] = df_daily['daily_load_kwh'].shift(1)       # Load from yesterday
df_daily['lag_7d'] = df_daily['daily_load_kwh'].shift(7)       # Load from same day last week

# Leakage-Free Rolling Mean (exclude the current day)
df_daily['rolling_mean_7d'] = df_daily['daily_load_kwh'].rolling(window=7).mean().shift(1)

# --- 4. Climate Interaction Features ---
# Daily load correlation with average temperature is strong, especially in Dry Season.
df_daily['dry_temp_interaction'] = df_daily['avg_temp'] * df_daily['is_dry']

# --- 5. Final Cleanup and Export ---
df_clean_daily = df_daily.dropna()

print("\n--- Daily Dataset Readiness Check ---")
print(f"Clean rows ready for training: {len(df_clean_daily)}.")
print(f"Exporting data to {OUTPUT_FILE}...")
df_clean_daily.to_csv(OUTPUT_FILE)
print("SUCCESS: Daily dataset ready for training.")