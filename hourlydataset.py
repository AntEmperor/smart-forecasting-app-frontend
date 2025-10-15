import pandas as pd
import numpy as np

# --- 1. CONFIGURATION ---
START_DATE = '2020-01-01'
END_DATE = '2025-01-01'
BASE_LOAD_MIN = 7000  # kW (~7 MW)
MAX_EXPECTED_LOAD = 10000 # kW (10 MW)
TEMP_BASELINE = 20    
OUTPUT_FILENAME = 'nigeria_stlf_dataset.csv'

# --- 2. DATASET CREATION & LOCAL CONTEXT ---
print("1. Creating the base 5-year hourly dataset...")
# FIX: Use inclusive='left' and freq='h'
dates = pd.date_range(start=START_DATE, end=END_DATE, freq='h', inclusive='left') 
df = pd.DataFrame(index=dates)
df['month'] = df.index.month
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)

# --- A. Nigerian Season Definition ---
df['is_rainy'] = df['month'].isin(range(4, 11)).astype(int)
df['is_harmattan'] = df['month'].isin([11, 12, 1]).astype(int)
df['is_dry'] = df['month'].isin([2, 3]).astype(int)

# --- B. Temperature and Humidity (SIMULATED DATA - REPLACE WITH REAL DATA) ---
df['temperature'] = 27 + 8 * np.sin(2 * np.pi * df.index.dayofyear / 365) + np.random.normal(0, 1.5, size=len(df))
df['humidity'] = 75 - 20 * df['is_dry'] + 15 * df['is_rainy'] + np.random.normal(0, 5, size=len(df))
df['humidity'] = df['humidity'].clip(40, 95)

# --- C. Load Generation (SIMULATED DATA - REPLACE WITH REAL DATA) ---
df['load_kw'] = BASE_LOAD_MIN + (df.index.dayofyear / 365) * 500
df['load_kw'] += np.sin(2 * np.pi * df['hour'] / 24) * 2000
df['load_kw'] -= df['is_weekend'] * 500
df['load_kw'] += ((df['temperature'] - 30).clip(lower=0) ** 1.5) * 400 * df['is_dry']

# --- 3. LEAKAGE-FREE FEATURE ENGINEERING ---
print("2. Engineering leakage-free features (Lags, Rolling, and Interactions)...")

# A. Leakage-Free Lag & Rolling Features
df['lag_1'] = df['load_kw'].shift(1)       
df['lag_24'] = df['load_kw'].shift(24)     
df['lag_168'] = df['load_kw'].shift(168)   
df['lag_diff_24h'] = df['lag_24'] - df['lag_24'].shift(1)
df['rolling_mean_24h'] = df['load_kw'].rolling(window=24).mean().shift(1)
df['rolling_std_24h'] = df['load_kw'].rolling(window=24).std().shift(1)

# B. Leakage-Free Daily Max Temperature
daily_max_temp = df['temperature'].resample('D').max().shift(1).resample('h').ffill()
df['max_temp_yesterday'] = daily_max_temp

# C. Simplified Fixed Holiday Feature
FIXED_HOLIDAYS = [(1, 1), (5, 1), (10, 1), (12, 25)]
holiday_dates = []
for year in range(df.index.min().year, df.index.max().year + 1):
    for month, day in FIXED_HOLIDAYS:
        try: holiday_dates.append(pd.to_datetime(f'{year}-{month}-{day}').date())
        except: pass
df['is_fixed_holiday'] = pd.Series(df.index.date).isin(holiday_dates).values.astype(int)
df['is_fixed_holiday_day_before'] = df['is_fixed_holiday'].shift(-24).fillna(0).astype(int)

# D. Season-Specific Interaction Features
df['dry_heat_load_spike'] = (df['temperature'] - TEMP_BASELINE).clip(lower=0) * df['is_dry'] * df['hour_sin']
df['dry_night_load_residual'] = df['max_temp_yesterday'] * df['is_dry'] * (1 - df['hour_sin']) 
df['harmattan_cold_effect'] = (TEMP_BASELINE - df['temperature']).clip(lower=0) * df['is_harmattan']
df['rainy_humidity_factor'] = (df['temperature'] - 25).clip(lower=0) * df['is_rainy'] * (df['humidity'] / 100)

# --- 4. FINAL CLEANUP AND EXPORT ---
df_clean = df.dropna()

print("3. Final Cleanup and Readiness Check...")
print(f"Total features created: {len(df_clean.columns)}")
print(f"Clean rows ready for training: {len(df_clean)}. Rows with NaN (initial lags) dropped.")
print("\nFirst 5 rows of the final dataset (Check the Lag and Rolling columns!):")
print(df_clean.head())
print("-" * 70)

# Export to CSV
try:
    df_clean.to_csv(OUTPUT_FILENAME, index=True, index_label='datetime')
    print(f"SUCCESS: Dataset saved as '{OUTPUT_FILENAME}' in the current directory.")
    print("The 'datetime' column will serve as your index.")
except Exception as e:
    print(f"ERROR saving file: {e}")