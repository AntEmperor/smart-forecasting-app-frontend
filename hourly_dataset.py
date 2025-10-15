import pandas as pd
import numpy as np

# --- 1. CONFIGURATION ---
START_DATE = '2020-01-01'
END_DATE = '2025-01-01'
# BASE_LOAD_MIN is the baseline minimum load required for the Max Load to hit ~10MW
BASE_LOAD_MIN = 7000  # kW (~7 MW)
MAX_EXPECTED_LOAD = 10000 # kW (10 MW) - Your specified System Peak Load target
TEMP_BASELINE = 20    # Baseline temperature (in Celsius) for cold effect calculation

# --- 2. DATASET CREATION & LOCAL CONTEXT ---
print("1. Creating the base 5-year hourly dataset with climate awareness...")
dates = pd.date_range(start=START_DATE, end=END_DATE, freq='h', inclusive='left')
df = pd.DataFrame(index=dates)
df['month'] = df.index.month
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24) # Cyclical hour feature

# --- A. Nigerian Season Definition ---
df['is_rainy'] = df['month'].isin(range(4, 11)).astype(int)       # April - October
df['is_harmattan'] = df['month'].isin([11, 12, 1]).astype(int)     # November - January
df['is_dry'] = df['month'].isin([2, 3]).astype(int)               # February - March

# --- B. Temperature and Humidity (SIMULATED DATA - REPLACE WITH REAL DATA) ---
# **ACTION REQUIRED: Replace these simulated columns with your actual data**
df['temperature'] = 27 + 8 * np.sin(2 * np.pi * df.index.dayofyear / 365) + np.random.normal(0, 1.5, size=len(df))
df['humidity'] = 75 - 20 * df['is_dry'] + 15 * df['is_rainy'] + np.random.normal(0, 5, size=len(df))
df['humidity'] = df['humidity'].clip(40, 95)

# --- C. Load Generation (SIMULATED DATA - REPLACE WITH REAL DATA) ---
# The synthetic load is structured to peak near the MAX_EXPECTED_LOAD (10MW)
df['load_kw'] = BASE_LOAD_MIN + (df.index.dayofyear / 365) * 500
df['load_kw'] += np.sin(2 * np.pi * df['hour'] / 24) * 2000
df['load_kw'] -= df['is_weekend'] * 500
df['load_kw'] += ((df['temperature'] - 30).clip(lower=0) ** 1.5) * 400 * df['is_dry']
# **ACTION REQUIRED: Replace this entire section with your true 'load_kw' values**

# --- 3. LEAKAGE-FREE FEATURE ENGINEERING ---
print("2. Engineering leakage-free features (Lags, Rolling, and Interactions)...")

# --- A. Leakage-Free Lag & Rolling Features (Prevents R^2 Leakage) ---

# CRITICAL FIX: The .shift(N) ensures the feature only looks to the past.
df['lag_1'] = df['load_kw'].shift(1)       
df['lag_24'] = df['load_kw'].shift(24)     
df['lag_168'] = df['load_kw'].shift(168)   
df['lag_diff_24h'] = df['lag_24'] - df['lag_24'].shift(1)

# CRITICAL FIX: Rolling mean/std use .shift(1) to exclude the current hour (t)
df['rolling_mean_24h'] = df['load_kw'].rolling(window=24).mean().shift(1)
df['rolling_std_24h'] = df['load_kw'].rolling(window=24).std().shift(1)


# --- B. Leakage-Free Daily Max Temperature (for Residual Heat) ---
# Calculates max temp of *yesterday* and applies it to every hour of today.
daily_max_temp = df['temperature'].resample('D').max().shift(1).resample('h').ffill()
df['max_temp_yesterday'] = daily_max_temp

# --- C. Simplified Fixed Holiday Feature (Max Demand Reduction) ---
FIXED_HOLIDAYS = [(1, 1), (5, 1), (10, 1), (12, 25)]
holiday_dates = []
for year in range(df.index.min().year, df.index.max().year + 1):
    for month, day in FIXED_HOLIDAYS:
        try: holiday_dates.append(pd.to_datetime(f'{year}-{month}-{day}').date())
        except: pass
df['is_fixed_holiday'] = pd.Series(df.index.date).isin(holiday_dates).values.astype(int)
df['is_fixed_holiday_day_before'] = df['is_fixed_holiday'].shift(-24).fillna(0).astype(int) # Day before holiday
# Create a Pandas Series from the index dates to enable the .isin() method


# --- D. Season-Specific Interaction Features (Forces Model Attention) ---

# 1. DRY SEASON HIGH HEAT SPIKE: Afternoon cooling load is high.
# Active during Dry season, peaks during afternoon/evening (high hour_sin).
df['dry_heat_load_spike'] = (df['temperature'] - TEMP_BASELINE).clip(lower=0) * df['is_dry'] * df['hour_sin']

# 2. DRY SEASON RESIDUAL NIGHT LOAD: Models retained heat from walls (Your Key Insight!)
# Active during Dry season night (low hour_sin) using YESTERDAY's max temp.
df['dry_night_load_residual'] = df['max_temp_yesterday'] * df['is_dry'] * (1 - df['hour_sin']) 

# 3. HARMATTAN COLD EFFECT: Captures potential appliance use during the coldest time.
df['harmattan_cold_effect'] = (TEMP_BASELINE - df['temperature']).clip(lower=0) * df['is_harmattan']

# 4. HUMIDITY INTERACTION: Differentiates rainy season cooling from dry season.
df['rainy_humidity_factor'] = (df['temperature'] - 25).clip(lower=0) * df['is_rainy'] * (df['humidity'] / 100)

# --- 4. FINAL CLEANUP AND TRAINING PREPARATION ---
df_clean = df.dropna()

print("3. Final Cleanup and Readiness Check...")
print(f"Total features created: {len(df_clean.columns)}")
print(f"Clean rows ready for training: {len(df_clean)}. Rows with NaN (initial lags) dropped.")
print("-" * 50)
print("NEXT STEP: Split df_clean temporally and train your XGBoost model.")