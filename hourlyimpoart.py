import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# --- Hourly Data and Model Configuration ---
HOURLY_DATA_FILE = 'nigeria_stlf_dataset.csv'
HOURLY_SPLIT_DATE = '2024-01-01'

df_hourly = pd.read_csv(HOURLY_DATA_FILE, index_col='datetime', parse_dates=True)
y_hourly = df_hourly['load_kw']
X_hourly = df_hourly.drop(columns=['load_kw'])

X_train_h = X_hourly[X_hourly.index < HOURLY_SPLIT_DATE]
y_train_h = y_hourly[y_hourly.index < HOURLY_SPLIT_DATE]

# Model trained with conservative, logic-forcing settings
xgb_hourly_model = xgb.XGBRegressor(
    objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, gamma=15, 
    max_depth=9, min_child_weight=5, random_state=42
)
xgb_hourly_model.fit(X_train_h, y_train_h)

# Generate and display feature importance
print("\n--- Hourly Model Feature Importance ---")
feature_importances_h = pd.Series(xgb_hourly_model.get_booster().get_score(importance_type='gain'))
feature_importances_h = feature_importances_h.sort_values(ascending=False).head(10)
print(feature_importances_h)

plt.figure(figsize=(10, 6))
feature_importances_h.plot(kind='barh', color='skyblue')
plt.title('Top 10 Hourly Feature Importances (STLF)')
plt.savefig('final_hourly_importance.png')
plt.close()