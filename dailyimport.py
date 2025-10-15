import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# --- Daily Data and Model Configuration ---
DAILY_DATA_FILE = 'nigeria_daily_dataset.csv'
DAILY_SPLIT_DATE = '2024-01-01'

df_daily = pd.read_csv(DAILY_DATA_FILE, index_col='timestamp', parse_dates=True)
y_daily = df_daily['daily_load_kwh']
X_daily = df_daily.drop(columns=['daily_load_kwh'])

X_train_d = X_daily[X_daily.index < DAILY_SPLIT_DATE]
y_train_d = y_daily[y_daily.index < DAILY_SPLIT_DATE]

# Model trained with the successful Daily settings (R2=0.9814)
xgb_daily_model = xgb.XGBRegressor(
    objective='reg:squarederror', n_estimators=500, learning_rate=0.08, gamma=5, 
    max_depth=7, min_child_weight=3, random_state=42
)
xgb_daily_model.fit(X_train_d, y_train_d)

# Generate and display feature importance
print("\n--- Daily Model Feature Importance (Base for Weekly Aggregation) ---")
feature_importances_d = pd.Series(xgb_daily_model.get_booster().get_score(importance_type='gain'))
feature_importances_d = feature_importances_d.sort_values(ascending=False).head(10)
print(feature_importances_d)

plt.figure(figsize=(10, 6))
feature_importances_d.plot(kind='barh', color='darkgreen')
plt.title('Top 10 Daily Feature Importances (MTLF)')
plt.savefig('final_daily_importance.png')
plt.close()