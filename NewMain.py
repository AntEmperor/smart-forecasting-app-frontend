"""
Rewritten backend - ALIGNED for Hourly/Daily Inputs.
- Added robust error handling and explicit NaN/None checks to prevent 500 errors.
- Weekly prediction is RESTORED to Daily Model Aggregation.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from collections import deque
from datetime import datetime, date, timedelta
import joblib, json, os, traceback, math
import pandas as pd
import numpy as np

# -----------------------
# Config - update if you move files
# -----------------------
# ==========================
# Model paths - ONLY HOURLY AND DAILY REQUIRED
# ==========================
# IMPORTANT: Ensure this path is 100% correct
MODEL_DIR = r"C:\Users\Admin\smart forecasting app\data set"
HOURLY_MODEL_FILE = os.path.join(MODEL_DIR, "hourly_stlf_model.joblib")
DAILY_MODEL_FILE = os.path.join(MODEL_DIR, "daily_mtlf_model.joblib")

# -----------------------
# App init & CORS (permissive for dev)
# -----------------------
app = FastAPI(title="SmartGrid STLF API (Aligned & Weekly Aggregation)")

# Ensure CORS is permissive and correct
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Globals
# -----------------------
MODELS: Dict[str, Any] = {"hourly": None, "daily": None}
PIPELINES: Dict[str, Any] = {"hourly": None, "daily": None}
FEATURE_COLS: Dict[str, Optional[List[str]]] = {"hourly": None, "daily": None}
PREDICTION_HISTORY: Dict[str, deque] = {
    "hourly": deque(maxlen=24),
    "daily": deque(maxlen=7),
    "weekly": deque(maxlen=12),  # History for aggregated weekly trend
}

# -----------------------
# Utility: JSON sanitization (convert NaN/inf/numpy types to JSON-safe values)
# -----------------------
def sanitize_for_json(obj):
    """
    Recursively convert numpy types and NaN/Inf to JSON-safe Python types.
    NaN/Infinity -> None
    numpy.int64/float64 -> int/float
    """
    if obj is None:
        return None
    # numpy scalar types
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if not math.isfinite(v):
            return None
        return v
    if isinstance(obj, (np.ndarray,)):
        return [sanitize_for_json(v) for v in obj.tolist()]
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, (int, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    # Pandas types
    if isinstance(obj, (pd.Series,)):
        return sanitize_for_json(obj.to_dict())
    if isinstance(obj, (pd.DataFrame,)):
        return sanitize_for_json(obj.to_dict(orient="records"))
    # Fallback: try to convert to native Python
    try:
        return obj.item() if hasattr(obj, "item") else obj
    except Exception:
        return str(obj)

# -----------------------
# Utility functions
# -----------------------
def safe_load_joblib(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"[WARN] failed to joblib.load({path}): {e}")
            return None
    return None

def safe_load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] failed to load json {path}: {e}")
            return None
    return None

def record_prediction(model_key: str, prediction_mw: float):
    hist_key = "weekly" if model_key == "daily_aggregation" else model_key
    if hist_key in PREDICTION_HISTORY:
        PREDICTION_HISTORY[hist_key].append(prediction_mw)

def attach_trend(resp: Dict[str, Any], model_key: str):
    hist_key = "weekly" if model_key == "daily_aggregation" else model_key
    resp["trend"] = list(PREDICTION_HISTORY.get(hist_key, []))
    return resp

# -----------------------
# Load artifacts at startup
# -----------------------
def load_all_artifacts():
    MODELS["hourly"] = safe_load_joblib(HOURLY_MODEL_FILE)
    MODELS["daily"] = safe_load_joblib(DAILY_MODEL_FILE)

    print("Artifact load summary:")
    print(" MODELS:", {k: (v is not None) for k, v in MODELS.items()})
    print(" PIPELINES:", {k: (v is not None) for k, v in PIPELINES.items()})
    print(" FEATURE_COLS loaded:", {k: (v is not None) for k, v in FEATURE_COLS.items()})

load_all_artifacts()

# -----------------------
# Request schemas (ALIGNED WITH NEWJAVA.JS)
# -----------------------
class HourlyInput(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of prediction (0-23)")
    day: int = Field(..., ge=1, le=31, description="Day of month")
    temperature: float = Field(..., description="Temperature in °C (derived from band)")
    season_dry: int = Field(..., ge=0, le=1)
    season_rainy: int = Field(..., ge=0, le=1)
    season_harmattan: int = Field(..., ge=0, le=1)
    last_actual: float = Field(..., description="Load at t-1 (Lag 1)")
    last_pred: float = Field(..., description="Load at t-24 (Lag 24)")
    humidity: Optional[float] = None
    max_temp_yesterday: Optional[float] = None
    rolling_mean_24h: Optional[float] = None
    rolling_std_24h: Optional[float] = None

class DailyInput(BaseModel):
    day: int = Field(..., ge=1, le=31, description="Day of month (for day of week/year calc)")
    temperature: float = Field(..., description="Temperature in °C (derived from band)")
    season_dry: int = Field(..., ge=0, le=1)
    season_rainy: int = Field(..., ge=0, le=1)
    season_harmattan: int = Field(..., ge=0, le=1)
    last_actual: float = Field(..., description="Load at day t-1 (Lag 1d)")
    last_pred: float = Field(..., description="Load at day t-7 (Lag 7d)")
    is_weekend: int = Field(..., ge=0, le=1)
    is_lockdown: int = Field(..., ge=0, le=1)
    holiday_type_national: int = Field(..., ge=0, le=1)
    holiday_type_religious: int = Field(..., ge=0, le=1)

class WeeklyQuery(BaseModel):
    start_date_utc: date

# -----------------------
# Feature engineering (construct feature dict for model) - ALIGNED
# -----------------------
def build_hourly_feature_dict(payload: HourlyInput) -> Dict[str, Any]:
    humidity_val = payload.humidity if payload.humidity is not None else 75.0

    feat = {
        'month': 1,
        'hour': payload.hour,
        'dayofweek': 2,
        'is_weekend': 0,
        'hour_sin': np.sin(2 * np.pi * payload.hour / 24.0),
        'is_rainy': payload.season_rainy,
        'is_harmattan': payload.season_harmattan,
        'is_dry': payload.season_dry,
        'temperature': payload.temperature,
        'humidity': humidity_val,
        'lag_1': payload.last_actual,
        'lag_24': payload.last_pred,
        'lag_168': np.nan,
        'rolling_mean_24h': payload.rolling_mean_24h if payload.rolling_mean_24h is not None else np.nan,
        'rolling_std_24h': payload.rolling_std_24h if payload.rolling_std_24h is not None else np.nan,
        'max_temp_yesterday': payload.max_temp_yesterday if payload.max_temp_yesterday is not None else np.nan,
        'dry_heat_load_spike': (payload.temperature - 20) * payload.season_dry * np.sin(2 * np.pi * payload.hour / 24.0),
        'dry_night_load_residual': 0.0,
        'harmattan_cold_effect': (20 - payload.temperature) * payload.season_harmattan,
        'rainy_humidity_factor': (payload.temperature - 25) * payload.season_rainy * (humidity_val / 100.0),
        'is_fixed_holiday': 0,
        'is_fixed_holiday_day_before': 0,
        'load_kw': np.nan
    }
    return feat

def build_daily_feature_dict(payload: DailyInput) -> Dict[str, Any]:
    feat = {
        'is_rainy': payload.season_rainy,
        'is_harmattan': payload.season_harmattan,
        'is_dry': payload.season_dry,
        'lag_1d': payload.last_actual,
        'lag_7d': payload.last_pred,
        'avg_temp': payload.temperature,
        'is_weekend': payload.is_weekend,
        'holiday_type_national': payload.holiday_type_national,
        'holiday_type_religious': payload.holiday_type_religious,
        'is_rain_day': payload.season_rainy,
        'dayofweek': 2,
        'dayofyear': 100,
        'month': 5,
        'rolling_mean_7d': np.nan,
        'dry_temp_interaction': payload.temperature * payload.season_dry,

    }
    return feat

def build_daily_feature_dict_for_aggregation(d: date) -> Dict[str, Any]:
    dt = pd.to_datetime(d)
    month = dt.month

    is_rainy = 1 if month in range(4,11) else 0
    is_harmattan = 1 if month in [11,12,1] else 0
    is_dry = 1 if month in [2,3] else 0

    feat = {
        'is_rainy': is_rainy,
        'is_harmattan': is_harmattan,
        'is_dry': is_dry,
        'lag_1d': np.nan,
        'lag_7d': np.nan,
        'avg_temp': 28.5,
        'is_weekend': 1 if dt.weekday() >= 5 else 0,
        'holiday_type_national': 0,
        'holiday_type_religious': 0,
        'is_rain_day': is_rainy,
        'dayofweek': dt.weekday(),
        'dayofyear': dt.timetuple().tm_yday,
        'month': month,
        'rolling_mean_7d': np.nan,
        'dry_temp_interaction': 28.5 * is_dry,
        
    }
    return feat

# -----------------------
# Helpers to prepare DF in canonical order and apply pipeline if present (CRITICAL CHANGE)
# -----------------------
def prepare_df_for_model(model_key: str, feat_dict: Dict[str, Any]) -> pd.DataFrame:
    canonical = FEATURE_COLS.get(model_key)
    df = pd.DataFrame([feat_dict])
    if canonical and isinstance(canonical, list) and len(canonical) > 0:
        df = df.reindex(columns=canonical)

    # Convert all columns to numeric, coercing errors to NaN
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Keep NaNs for pipeline; do not replace with None here (pipeline expects numeric dtypes)
    return df

def apply_pipeline(model_key: str, X_df: pd.DataFrame) -> pd.DataFrame:
    pipeline = PIPELINES.get(model_key)

    if pipeline is None:
        return X_df.fillna(X_df.mean().fillna(0.0))

    try:
        X_trans = pipeline.transform(X_df)

        if isinstance(X_trans, np.ndarray):
            cols = FEATURE_COLS.get(model_key)
            if cols is None or len(cols) != X_trans.shape[1]:
                cols = [f"f{i}" for i in range(X_trans.shape[1])]
            return pd.DataFrame(X_trans, columns=cols).fillna(0.0)
        else:
            return X_trans.fillna(0.0)

    except Exception as e:
        print(f"[ERROR] Pipeline transform failed for {model_key}: {e}")
        return X_df.fillna(X_df.mean().fillna(0.0))

# -----------------------
# Core prediction function (NO CHANGE HERE)
# -----------------------
def predict_with_model(model_key: str, feat_dict: Dict[str, Any]) -> Dict[str, Any]:
    model = MODELS.get(model_key)
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded. Check model paths or if the model file is missing: {os.path.join(MODEL_DIR, f'{model_key}_stlf_model.joblib')}")

    try:
        X_df = prepare_df_for_model(model_key, feat_dict)
        X_proc = apply_pipeline(model_key, X_df)
        X_np = X_proc.values if hasattr(X_proc, "values") else np.asarray(X_proc)

        if np.isnan(X_np).any():
            print("[ERROR] FINAL INPUT ARRAY CONTAINS NANs AFTER PIPELINE/IMPUTATION!")
            X_np = np.nan_to_num(X_np, nan=0.0)

        y_pred = model.predict(X_np)
        y = float(y_pred[0]) if hasattr(y_pred, "__len__") else float(y_pred)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"FATAL MODEL PREDICT ERROR: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"Model prediction failed. Check terminal for traceback. Error: {e}")

    pred_mw = round(y / 1000.0, 3)
    record_prediction(model_key, pred_mw)
    resp = {"model_used": model_key, "prediction_kW": round(y, 2), "prediction_MW": pred_mw}
    return attach_trend(resp, model_key)

# -----------------------
# Endpoints (UNCHANGED except sanitize responses)
# -----------------------
@app.get("/")
def root():
    return {"message": "SmartGrid STLF API - ALIGNED and READY for Demo."}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": {k: (v is not None) for k, v in MODELS.items()},
        "pipelines_loaded": {k: (v is not None) for k, v in PIPELINES.items()},
        "feature_cols_available": {k: (v is not None) for k, v in FEATURE_COLS.items()}
    }

@app.post("/debug/feature_vector_hourly")
def debug_feature_vector_hourly(payload: HourlyInput):
    feat = build_hourly_feature_dict(payload)
    prepared = prepare_df_for_model("hourly", feat)
    processed = apply_pipeline("hourly", prepared)  # Run pipeline for full view

    # Convert pandas row to dict and sanitize before returning
    row = processed.head(1).to_dict(orient="records")[0]
    return {"feature_dict": sanitize_for_json(feat), "prepared_feature_row": sanitize_for_json(row)}

@app.post("/debug/feature_vector_daily")
def debug_feature_vector_daily(payload: DailyInput):
    feat = build_daily_feature_dict(payload)
    prepared = prepare_df_for_model("daily", feat)
    processed = apply_pipeline("daily", prepared)
    row = processed.head(1).to_dict(orient="records")[0]
    return {"feature_dict": sanitize_for_json(feat), "prepared_feature_row": sanitize_for_json(row)}


@app.post("/predict/hourly")
def predict_hourly(payload: HourlyInput):
    feat = build_hourly_feature_dict(payload)
    resp = predict_with_model("hourly", feat)
    return sanitize_for_json(resp)

@app.post("/predict/daily")
def predict_daily(payload: DailyInput):
    feat = build_daily_feature_dict(payload)
    resp = predict_with_model("daily", feat)
    return sanitize_for_json(resp)

@app.post("/predict/weekly")
def predict_weekly(payload: WeeklyQuery):
    if MODELS.get("daily") is None:
        raise HTTPException(status_code=500, detail="Daily model not loaded for weekly aggregation")

    total_kw = 0.0
    breakdown = []
    for i in range(7):
        d = payload.start_date_utc + timedelta(days=i)
        feat = build_daily_feature_dict_for_aggregation(d)

        # Predict using the Daily model (ignores trend recording for internal steps)
        res = predict_with_model("daily", feat)

        total_kw += res["prediction_kW"]
        breakdown.append({"date": d.isoformat(), "prediction_kW": res["prediction_kW"]})

    weekly_mw = round(total_kw / 1000.0, 3)
    record_prediction("daily_aggregation", weekly_mw)

    resp = {
        "model_used": "daily_aggregation",
        "total_weekly_kW": round(total_kw, 2),
        "prediction_MW": weekly_mw,
        "daily_breakdown": breakdown
    }
    return sanitize_for_json(attach_trend(resp, "daily_aggregation"))

import csv
from datetime import datetime

FEEDBACK_FILE = "feedback_log.csv"

@app.post("/feedback")
def submit_feedback(data: Dict[str, Any]):
    print(f"FEEDBACK received: {data}")
    
    # Ensure CSV file exists with header
    file_exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Model", "Prediction_MW", "Actual_MW", "Comments"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data.get("model", "N/A"),
            data.get("prediction", ""),
            data.get("actual", ""),
            data.get("comments", "")
        ])
    return {"message": "Feedback saved successfully ✅"}

from fastapi.responses import FileResponse

@app.get("/feedback/csv")
def download_feedback_csv():
    if not os.path.exists(FEEDBACK_FILE):
        return {"detail": "No feedback file found."}
    return FileResponse(
        FEEDBACK_FILE,
        media_type="text/csv",
        filename="feedback_log.csv"
    )

@app.post("/reset")
def reset_history():
    for key in PREDICTION_HISTORY:
        PREDICTION_HISTORY[key].clear()
    return {"message": "Prediction history reset."}

