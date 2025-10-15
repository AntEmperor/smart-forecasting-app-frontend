import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib


# --- 1. Configuration & Data Loading ---
OUTPUT_FILENAME = 'nigeria_stlf_dataset.csv'
SPLIT_DATE = '2024-01-01' # 4 Years Train (2020-2023), 1 Year Test (2024)

print(f"Loading and processing '{OUTPUT_FILENAME}'...")
df_clean = pd.read_csv(OUTPUT_FILENAME)

# Set the 'datetime' column as the DataFrame index
df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])
df_clean = df_clean.set_index('datetime')

# 2. Define Target (y) and Features (X)
y = df_clean['load_kw']
feature_cols = [col for col in df_clean.columns if col != 'load_kw']
X = df_clean[feature_cols]

# 3. Perform the Temporal Split
X_train = X[X.index < SPLIT_DATE]
X_test = X[X.index >= SPLIT_DATE]
y_train = y[y.index < SPLIT_DATE]
y_test = y[y.index >= SPLIT_DATE]

# --- 4. XGBoost Training (Conservative Tuning) ---
print("\n--- Training XGBoost Model with Conservative Settings ---")

# The conservative hyperparameters force the model to look for complex, non-linear 
# relationships (like climate interactions) instead of over-relying on lag features.
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    gamma=15,             # High gamma forces model to seek larger, more significant splits
    max_depth=9,          # Deep trees capture complex interactions (e.g., Dry AND Temp High)
    min_child_weight=5,   # Stabilizes the model against local noise
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# --- 5. Prediction and Evaluation ---
y_pred = xgb_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation on Leakage-Free Test Set (2024) ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kW")
print(f"R-squared (R²): {r2:.4f}") 
print("-" * 50)

joblib.dump(xgb_model, 'hourly_stlf_model.joblib')
print("Hourly Model saved as 'hourly_stlf_model.joblib'")
# --- 6. CRITICAL VERIFICATION: Feature Importance Plot ---
print("Generating Feature Importance Plot (Verification Step)...")

# Get feature importances based on 'gain' (total improvement to the model)
feature_importances = pd.Series(xgb_model.get_booster().get_score(importance_type='gain'))
feature_importances = feature_importances.sort_values(ascending=False)

# Plot the top 15 features
plt.figure(figsize=(10, 6))
feature_importances.head(15).plot(kind='barh', color='skyblue')
plt.title('Top 15 Feature Importances (Gain)')
plt.xlabel('Gain (Total Contribution to Model)')
plt.ylabel('Feature')
plt.gca().invert_yaxis() # Highest gain at the top
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("Verification Plot saved as 'feature_importance.png'.")
print("Check the plot to confirm our custom features are highly ranked!")