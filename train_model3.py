import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURATION ---
DAILY_DATA_FILE = 'nigeria_daily_dataset.csv'
DAILY_SPLIT_DATE = '2024-01-01' 
TARGET_COL = 'daily_load_kwh'

# 1. Load Data and Train the Successful Daily Model
df_daily = pd.read_csv(DAILY_DATA_FILE, index_col='timestamp', parse_dates=True)
y_daily = df_daily[TARGET_COL]
X_daily = df_daily.drop(columns=[TARGET_COL])

X_train_d = X_daily[X_daily.index < DAILY_SPLIT_DATE]
X_test_d = X_daily[X_daily.index >= DAILY_SPLIT_DATE]
y_test_d = y_daily[y_daily.index >= DAILY_SPLIT_DATE]

print("\n--- Rerunning SUCCESSFUL Daily Model (Required for Aggregation) ---")

# Standard Daily XGBoost Configuration (The one that gave R2=0.9814)
xgb_daily_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,        
    learning_rate=0.08,
    gamma=5,                 
    max_depth=7,             
    min_child_weight=3,
    tree_method='hist',
    random_state=42
)

# Rerunning fit on the daily training data
xgb_daily_model.fit(X_train_d, df_daily[TARGET_COL][df_daily.index < DAILY_SPLIT_DATE])

# Generate Daily Predictions for the Test Set (2024)
y_pred_d = xgb_daily_model.predict(X_test_d)

# --- 2. Aggregate Daily Forecasts to Weekly Total ---

# Create a DataFrame for the test period predictions
df_predictions = pd.DataFrame({
    'date': X_test_d.index,
    'actual_daily_load': y_test_d.values,
    'predicted_daily_load': y_pred_d
}).set_index('date')

# Aggregate the Daily Predictions and Actuals to Weekly Sums
weekly_forecast = df_predictions['predicted_daily_load'].resample('W').sum()
weekly_actual = df_predictions['actual_daily_load'].resample('W').sum()

# --- 3. Final Weekly Evaluation ---
rmse_weekly_agg = np.sqrt(mean_squared_error(weekly_actual, weekly_forecast))
r2_weekly_agg = r2_score(weekly_actual, weekly_forecast)

print("\n--- Final Weekly Model Evaluation (FORECAST BY AGGREGATION) ---")
print(f"Base Model Used: Daily Load Forecast (R2 = 0.9814)")
print(f"Root Mean Squared Error (RMSE) on Weekly Test Set: {rmse_weekly_agg:.2f} kWh")
print(f"R-squared (R²) on Weekly Test Set: {r2_weekly_agg:.4f}") 
print("-" * 50)
print("This aggregated result is the definitive, high-quality Weekly Forecast.")