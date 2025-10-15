import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib 

# --- CONFIGURATION ---
DAILY_DATA_FILE = 'nigeria_daily_dataset.csv'
DAILY_SPLIT_DATE = '2024-01-01' 

# 1. Load Data and Split
df_daily = pd.read_csv(DAILY_DATA_FILE, index_col='timestamp', parse_dates=True)
y_daily = df_daily['daily_load_kwh']
X_daily = df_daily.drop(columns=['daily_load_kwh'])

X_train_d = X_daily[X_daily.index < DAILY_SPLIT_DATE]
X_test_d = X_daily[X_daily.index >= DAILY_SPLIT_DATE]
y_train_d = y_daily[y_daily.index < DAILY_SPLIT_DATE]
y_test_d = y_daily[y_daily.index >= DAILY_SPLIT_DATE]

print("\n--- Training Daily XGBoost Model ---")

# --- XGBoost Configuration ---
# Hyperparameters: A good balance for daily data.
xgb_daily_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,        # Fewer estimators needed for simpler daily cycle
    learning_rate=0.08,
    gamma=5,                 # Lower gamma than hourly, as daily patterns are less complex
    max_depth=7,             # Sufficient depth for daily features
    min_child_weight=3,
    tree_method='hist',
    random_state=42
)

xgb_daily_model.fit(X_train_d, y_train_d)

# --- Evaluation ---
y_pred_d = xgb_daily_model.predict(X_test_d)

rmse_d = np.sqrt(mean_squared_error(y_test_d, y_pred_d))
r2_d = r2_score(y_test_d, y_pred_d)

joblib.dump(xgb_daily_model, 'daily_mtlf_model.joblib')
print("Daily Model saved as 'daily_mtlf_model.joblib'")
# -----------------------

print("\n--- Daily Model Evaluation Complete ---")
print(f"Root Mean Squared Error (RMSE) on Daily Test Set: {rmse_d:.2f} kWh")
print(f"R-squared (R²) on Daily Test Set: {r2_d:.4f}") 
print("-" * 50)
print("NEXT: We can proceed to the Weekly Model.")