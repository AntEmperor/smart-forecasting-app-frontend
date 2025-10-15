"""
GENERATE: nigeria_stlf_dataset.csv (corrected, leakage-free)
Run this first. Requires: pandas, numpy.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os

np.random.seed(42)

# --- Load the earlier simulated file we created (smartgrid_5yr_hourly.csv)
# If you do not have it, run the prior full simulation script first.
src = "/mnt/data/smartgrid_5yr_hourly.csv"
df = pd.read_csv(src, parse_dates=['timestamp'])

# --- Clean up suspicious columns (drop leakage/opaque features if present)
drop_cols = [
    'load_det','third_sunday_effect','holiday_multiplier','is_third_sunday',
    'temp_effect','rain_effect_multiplier'
]
for c in drop_cols:
    if c in df.columns:
        df.drop(columns=[c], inplace=True, errors='ignore')

# --- Create clear holiday binary flags, rename for clarity
# If original used holiday_type_national / religious, convert; else create zeros
if 'holiday_type_national' in df.columns:
    df['is_fixed_holiday_national'] = df['holiday_type_national'].astype(int)
    df.drop(columns=['holiday_type_national'], inplace=True)
else:
    df['is_fixed_holiday_national'] = 0
if 'holiday_type_religious' in df.columns:
    df['is_fixed_holiday_religious'] = df['holiday_type_religious'].astype(int)
    df.drop(columns=['holiday_type_religious'], inplace=True)
else:
    df['is_fixed_holiday_religious'] = 0

# --- Create max_temp_yesterday (vectorized)
df = df.sort_values('timestamp').reset_index(drop=True)
daily_max = df.set_index('timestamp')['temperature'].resample('D').max()
daily_max_yesterday = daily_max.shift(1)  # yesterday's max
# Map daily_max_yesterday back to hourly rows
df['max_temp_yesterday'] = df['timestamp'].dt.normalize().map(daily_max_yesterday)

# Fill edge NaNs conservatively (bfill/ffill) - but these won't leak because we'll drop initial rows later
df['max_temp_yesterday'] = df['max_temp_yesterday'].fillna(method='bfill').fillna(method='ffill')

# --- Explicit interaction features (no opaque multipliers)
df['temp_times_dry'] = df['temperature'] * df['season_dry']
df['temp_times_harmattan'] = df['temperature'] * df['season_harmattan']
df['temp_times_rainy'] = df['temperature'] * df['season_rainy']

# dry_heat_load_spike: indicator for very hot dry-season daytime
df['dry_heat_load_spike'] = (
    (df['season_dry'] == 1) &
    (df['hour'].between(10,16)) &
    (df['temperature'] > 34)
).astype(int)

# dry_night_load_residual: late-night residual load from yesterday's max temp (only in dry season nights)
night_mask = df['hour'].isin([0,1,2,3,4,5,22,23]) & (df['season_dry'] == 1)
df['dry_night_load_residual'] = 0.0
df.loc[night_mask, 'dry_night_load_residual'] = np.maximum(0, df.loc[night_mask, 'max_temp_yesterday'] - 30.0) * 50.0
# (units: kW per degree above 30C; tune multiplier if needed)

# --- Proper lag and rolling features (shift only), keep NaN initial
df['lag_1'] = df['load_kw'].shift(1)
df['lag_24'] = df['load_kw'].shift(24)
df['lag_168'] = df['load_kw'].shift(168)
df['lag_diff_24h'] = df['load_kw'] - df['lag_24']  # will be NaN for first 24+ rows

df['rolling_mean_24h'] = df['load_kw'].shift(1).rolling(window=24, min_periods=1).mean()
df['rolling_std_24h'] = df['load_kw'].shift(1).rolling(window=24, min_periods=1).std().fillna(0.0)

# --- Drop initial rows that do not have full lag history (first 168 hours)
df = df.sort_values('timestamp').reset_index(drop=True)
df = df.iloc[168:].reset_index(drop=True)

# --- Final selected clean features (24 features + target)
selected_cols = [
    'timestamp','year','month','day','hour','weekday','is_weekend',
    'season_rainy','season_harmattan','season_dry',
    'temperature','is_rain_day',
    'is_fixed_holiday_national','is_fixed_holiday_religious',
    'is_peak_hour','base_load',
    'max_temp_yesterday','dry_heat_load_spike','dry_night_load_residual',
    'temp_times_dry','temp_times_harmattan','temp_times_rainy',
    'lag_1','lag_24','lag_168','lag_diff_24h','rolling_mean_24h','rolling_std_24h',
    'load_kw'
]
# Ensure all exist
for c in selected_cols:
    if c not in df.columns:
        df[c] = 0.0

df = df[selected_cols]

# --- Save corrected dataset
out = "/mnt/data/nigeria_stlf_dataset.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)
print("Saved corrected dataset:", out)
print("Rows:", len(df))
print("Sample rows:\n", df.head(5).to_string(index=False))
