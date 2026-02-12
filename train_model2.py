from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# --- Season-aware Sample Weighting for DAILY model ---

season_weights = {
    'rainy': 1.0,
    'harmattan': 1.4,
    'dry': 1.8
}

sample_weight_d = X_train_d.apply(
    lambda row: season_weights[
        'rainy' if row['is_rainy'] == 1 else
        'harmattan' if row['is_harmattan'] == 1 else
        'dry'
    ],
    axis=1
)

xgb_daily_model.fit(
    X_train_d,
    y_train_d,
    sample_weight=sample_weight_d
)


# --- Evaluation ---
y_pred_d = xgb_daily_model.predict(X_test_d)

rmse = np.sqrt(mean_squared_error(y_test_d, y_pred_d))
mae = mean_absolute_error(y_test_d, y_pred_d)
r2 = r2_score(y_test_d, y_pred_d)
mape = np.mean(np.abs((y_test_d - y_pred_d) / y_test_d)) * 100

joblib.dump(xgb_daily_model, 'daily_mtlf_model1.joblib')
print("Daily Model saved as 'daily_mtlf_model1.joblib'")
# -----------------------
print("\n📊 Evaluation Results:")
print(f"RMSE : {rmse:,.2f}")
print(f"MAE  : {mae:,.2f}")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape:.2f}%")
print("\n--- Daily Model Evaluation Complete ---")
# --- 6. CRITICAL VERIFICATION: Feature Importance Plot ---
print("Generating Feature Importance Plot (Verification Step)...")

for i in range(5):
    print(f"Actual: {y_test_d.iloc[i]:,.2f} | Predicted: {y_pred_d[i]:,.2f}")
# Plot feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_daily_model, height=0.5, max_num_features=15)
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("models/importance1.png")
plt.show()