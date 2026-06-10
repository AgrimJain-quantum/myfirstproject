import os
import json
import sys
import joblib
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Enable UTF-8 console output for Windows to support emojis
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "outputs")
DATASETS_DIR = os.path.join(BASE_DIR, "..", "datasets")
CSV_DIR = os.path.join(OUTPUTS_DIR, "csv files")
JSON_DIR = os.path.join(OUTPUTS_DIR, "json files")

# Global dicts for models and configs
models = {}
scaler = None
mean_features = {}
dashboard_config = {}

# In-memory CSV data caching
metrics_cache = []
features_cache = []
optuna_xgb_cache = []
optuna_lgbm_cache = []
lstm_hist_cache = []
predictions_cache = []

# Ordered list of features expected by the models (24 features)
FEATURES_ORDER = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "weekday", "weekend", "is_peak_hour", "is_day",
    "temp", "dwpt", "rhum", "wspd", "pres",
    "wdir_sin", "wdir_cos",
    "temp_hour", "temp_x_peak",
    "lag_12", "lag_288", "lag_2016",
    "roll_mean_12", "roll_std_12", "roll_max_12", "roll_min_12",
]

def calculate_mean_features():
    """Load the dataset and compute feature engineering to derive average baseline values."""
    print("⚙️  Calculating feature engineering averages from the dataset...")
    csv_file = os.path.join(DATASETS_DIR, "powerdemand_5min_2021_to_2024_with weather.csv")
    if not os.path.exists(csv_file):
        # Fallback search
        possible_paths = [
            os.path.join(BASE_DIR, "../datasets/powerdemand_5min_2021_to_2024_with weather.csv"),
            os.path.join(BASE_DIR, "../../electricityConsumptionAndProductioction.csv"),
            os.path.join(BASE_DIR, "datasets/powerdemand_5min_2021_to_2024_with weather.csv"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                csv_file = p
                break
    
    if not os.path.exists(csv_file):
        print("⚠️ Warning: Delhi power demand dataset CSV not found. Baseline features will use fallback averages.")
        return {f: 0.0 for f in FEATURES_ORDER}

    try:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        if "moving_avg_3" in df.columns:
            df.drop(columns=["moving_avg_3"], inplace=True)
        df = df.rename(columns={"Power demand": "load"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        
        # Preprocessing & engineering identical to V11
        df["wdir"] = df["wdir"].ffill()
        df["wdir_sin"] = np.sin(2 * np.pi * df["wdir"] / 360)
        df["wdir_cos"] = np.cos(2 * np.pi * df["wdir"] / 360)
        df.drop(columns=["wdir"], inplace=True)

        df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
        df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df.drop(columns=["hour", "month", "minute", "day", "year"], inplace=True)

        df["weekday"]      = df["datetime"].dt.weekday
        df["weekend"]      = (df["weekday"] >= 5).astype(int)
        df["is_peak_hour"] = df["datetime"].dt.hour.between(18, 21).astype(int)
        df["is_day"]       = df["datetime"].dt.hour.between(6, 18).astype(int)

        df["temp_hour"] = df["temp"] * df["datetime"].dt.hour
        df["temp_x_peak"] = df["temp"] * df["is_peak_hour"]

        df["lag_12"]   = df["load"].shift(12)
        df["lag_288"]  = df["load"].shift(288)
        df["lag_2016"] = df["load"].shift(2016)

        df["roll_mean_12"] = df["load"].shift(1).rolling(12).mean()
        df["roll_std_12"]  = df["load"].shift(1).rolling(12).std()
        df["roll_max_12"]  = df["load"].shift(1).rolling(12).max()
        df["roll_min_12"]  = df["load"].shift(1).rolling(12).min()
        
        df.dropna(inplace=True)
        
        means = df[FEATURES_ORDER].mean().to_dict()
        
        # Save to outputs folder for caching
        mean_features_path = os.path.join(JSON_DIR, "mean_features.json")
        os.makedirs(JSON_DIR, exist_ok=True)
        with open(mean_features_path, "w") as out:
            json.dump(means, out, indent=4)
        print("✅ Saved computed baseline feature averages to outputs/mean_features.json")
        return means
    except Exception as ex:
        print(f"⚠️ Error computing feature averages: {ex}. Using fallback averages.")
        return {f: 0.0 for f in FEATURES_ORDER}

def download_models_if_missing():
    """Auto-download models from Hugging Face Model Repository if missing and env is set."""
    hf_repo = os.environ.get("HF_MODEL_REPO", "")
    if not hf_repo:
        return
    
    import urllib.request
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    model_files = ["scaler.pkl", "rf_model.pkl", "xgb_model.pkl", "lgbm_model.pkl"]
    
    for filename in model_files:
        filepath = os.path.join(OUTPUTS_DIR, filename)
        if not os.path.exists(filepath):
            url = f"https://huggingface.co/{hf_repo}/resolve/main/{filename}"
            print(f"📥 Downloading {filename} from HF Model Repository: {url} ...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"✅ Downloaded {filename} successfully.")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scaler, mean_features, dashboard_config
    global metrics_cache, features_cache, optuna_xgb_cache, optuna_lgbm_cache, lstm_hist_cache, predictions_cache
    print("⏳ Starting up FastAPI Inference Server...")
    
    # 1. Load pickle files
    try:
        # Trigger Hugging Face download if environment variable is set
        download_models_if_missing()
        
        # Locate pickle directory (support standard outputs folder and external fallback)
        pkl_dir = OUTPUTS_DIR
        if not os.path.exists(os.path.join(OUTPUTS_DIR, "scaler.pkl")):
            fallback_pkl_dir = r"C:\Users\Agrim Jain\Desktop\Coding\pkl files"
            if os.path.exists(os.path.join(fallback_pkl_dir, "scaler.pkl")):
                pkl_dir = fallback_pkl_dir
                print(f"ℹ️ Local pkl files detected in fallback directory: {pkl_dir}")
        
        scaler = joblib.load(os.path.join(pkl_dir, "scaler.pkl"))
        models["Random Forest"] = joblib.load(os.path.join(pkl_dir, "rf_model.pkl"))
        models["XGBoost"] = joblib.load(os.path.join(pkl_dir, "xgb_model.pkl"))
        models["LightGBM"] = joblib.load(os.path.join(pkl_dir, "lgbm_model.pkl"))
        print(f"✅ Models and Scaler pickles loaded successfully from: {pkl_dir}")
    except Exception as e:
        print(f"❌ Error loading pickle models: {e}. Inference endpoint will fail.")
    
    # 2. Load dashboard config
    config_path = os.path.join(JSON_DIR, "dashboard_config.json")
    if not os.path.exists(config_path):
        # Fallback to standard location
        config_path = os.path.join(OUTPUTS_DIR, "dashboard_config.json")
        
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                dashboard_config = json.load(f)
            print("✅ Dashboard weights configuration loaded.")
        except Exception as e:
            print(f"⚠️ Error reading dashboard_config.json: {e}")
    else:
        print("⚠️ Warning: dashboard_config.json missing. Defaulting to 50/50 ensemble weights.")
        dashboard_config = {"w_xgb": 0.5, "w_lgbm": 0.5}

    # 3. Load baseline feature averages
    mean_features_path = os.path.join(JSON_DIR, "mean_features.json")
    if not os.path.exists(mean_features_path):
        # Fallback to standard location
        mean_features_path = os.path.join(OUTPUTS_DIR, "mean_features.json")
        
    if os.path.exists(mean_features_path):
        try:
            with open(mean_features_path, "r") as f:
                mean_features = json.load(f)
            print("✅ Baseline feature averages loaded.")
        except Exception as e:
            print(f"⚠️ Error reading mean_features.json: {e}")
            mean_features = calculate_mean_features()
    else:
        mean_features = calculate_mean_features()

    # 4. Load CSV outputs into memory caching for speed
    try:
        # Helper to resolve CSV paths with fallback
        def get_csv_path(filename):
            nested_path = os.path.join(CSV_DIR, filename)
            return nested_path if os.path.exists(nested_path) else os.path.join(OUTPUTS_DIR, filename)

        # Load metrics
        metrics_df = pd.read_csv(get_csv_path("model_metrics.csv"))
        metrics_cache = metrics_df.to_dict(orient="records")
        
        # Load feature importances
        features_df = pd.read_csv(get_csv_path("feature_importances.csv"))
        features_cache = features_df.to_dict(orient="records")
        
        # Load Optuna histories
        optuna_xgb_cache = pd.read_csv(get_csv_path("optuna_xgb_history.csv")).to_dict(orient="records")
        optuna_lgbm_cache = pd.read_csv(get_csv_path("optuna_lgbm_history.csv")).to_dict(orient="records")
        
        # Load LSTM history
        lstm_hist_cache = pd.read_csv(get_csv_path("lstm_history.csv")).to_dict(orient="records")
        
        # Load test predictions (for rendering charts)
        predictions_df = pd.read_csv(get_csv_path("test_predictions.csv"))
        predictions_cache = predictions_df.to_dict(orient="records")
        print("✅ Data files loaded and cached successfully.")
    except Exception as e:
        print(f"❌ Error loading cached CSV files: {e}")

    yield
    print("🧹 Cleaning up server resources...")

app = FastAPI(
    title="PowerCast AI Inference Service",
    version="2.4.0",
    lifespan=lifespan
)

# CORS middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/api/metrics")
async def get_metrics():
    return metrics_cache

@app.get("/api/features")
async def get_features():
    return features_cache

@app.get("/api/optuna")
async def get_optuna():
    return {
        "xgb": optuna_xgb_cache,
        "lgbm": optuna_lgbm_cache
    }

@app.get("/api/lstm")
async def get_lstm():
    return lstm_hist_cache

@app.get("/api/predictions")
async def get_predictions(limit: int = 2016):
    # Slice the cached predictions array to return a manageable size (last 7 days of 5-min readings = 2016 rows by default)
    if limit <= 0:
        return predictions_cache
    return predictions_cache[-limit:]

# Simulation schema
class SimulatorInput(BaseModel):
    temp: float
    rhum: float
    pres: float
    wspd: float
    hour: int
    weekday: int
    is_peak_hour: bool

@app.post("/api/predict")
async def predict_load(data: SimulatorInput):
    if not models or scaler is None:
        raise HTTPException(status_code=500, detail="Models or Scaler not loaded on FastAPI startup.")
    
    try:
        # 1. Compute Cyclical time encodings
        hour_sin = float(np.sin(2 * np.pi * data.hour / 24))
        hour_cos = float(np.cos(2 * np.pi * data.hour / 24))
        
        # For month, let's use the current month or fallback to mean_features
        month_sin = mean_features.get("month_sin", 0.0)
        month_cos = mean_features.get("month_cos", 0.0)
        
        # Weekday/Weekend
        weekday = data.weekday
        weekend = 1 if weekday >= 5 else 0
        
        # Peak & Day flags
        is_peak_hour = 1 if data.is_peak_hour else 0
        is_day = 1 if 6 <= data.hour <= 18 else 0
        
        # Interaction features
        temp_hour = float(data.temp * data.hour)
        temp_x_peak = float(data.temp * is_peak_hour)
        
        # Build raw feature vector using user input and falling back to baseline means
        input_data = {}
        for feature in FEATURES_ORDER:
            if feature == "hour_sin":
                input_data[feature] = hour_sin
            elif feature == "hour_cos":
                input_data[feature] = hour_cos
            elif feature == "month_sin":
                input_data[feature] = month_sin
            elif feature == "month_cos":
                input_data[feature] = month_cos
            elif feature == "weekday":
                input_data[feature] = weekday
            elif feature == "weekend":
                input_data[feature] = weekend
            elif feature == "is_peak_hour":
                input_data[feature] = is_peak_hour
            elif feature == "is_day":
                input_data[feature] = is_day
            elif feature == "temp":
                input_data[feature] = data.temp
            elif feature == "rhum":
                input_data[feature] = data.rhum
            elif feature == "pres":
                input_data[feature] = data.pres
            elif feature == "wspd":
                input_data[feature] = data.wspd
            elif feature == "temp_hour":
                input_data[feature] = temp_hour
            elif feature == "temp_x_peak":
                input_data[feature] = temp_x_peak
            else:
                # Fallback to historical mean features for missing lags, rolling variables, dwpt, wdir sin/cos
                input_data[feature] = mean_features.get(feature, 0.0)
        
        # Construct the array
        feature_vector = np.array([[input_data[f] for f in FEATURES_ORDER]], dtype=np.float32)
        
        # Scaled feature vector (for distance/linear regression models, not needed for tree boosters but we load scaler anyway)
        # scaler is StandardScaler fitted on X_train (24 columns)
        scaled_vector = scaler.transform(feature_vector)
        
        # Run inference using the loaded model objects
        rf_pred = float(models["Random Forest"].predict(feature_vector)[0])
        xgb_pred = float(models["XGBoost"].predict(feature_vector)[0])
        lgbm_pred = float(models["LightGBM"].predict(feature_vector)[0])
        
        # Ensemble Weighted Prediction
        w_xgb = dashboard_config.get("w_xgb", 0.5)
        w_lgbm = dashboard_config.get("w_lgbm", 0.5)
        ensemble_pred = w_xgb * xgb_pred + w_lgbm * lgbm_pred
        
        return {
            "Random Forest": rf_pred,
            "XGBoost (Tuned)": xgb_pred,
            "LightGBM (Tuned)": lgbm_pred,
            "Ensemble (XGB+LGBM)": ensemble_pred
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference computation error: {str(e)}")

# Mount static files (HTML, JS, CSS)
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Respect PORT env variable (default to 8001 locally, 7860 on Hugging Face Spaces)
    port = int(os.environ.get("PORT", 8001))
    # Bind to 0.0.0.0 if running in Docker/HuggingFace or dynamic PORT environment
    host = "0.0.0.0" if os.environ.get("PORT") or os.environ.get("HF_MODEL_REPO") else "127.0.0.1"
    reload = True if host == "127.0.0.1" else False
    print(f"🚀 Starting server on {host}:{port} (reload={reload})")
    uvicorn.run("server:app", host=host, port=port, reload=reload)
