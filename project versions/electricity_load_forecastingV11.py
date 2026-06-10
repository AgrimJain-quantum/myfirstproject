# ============================================================
# ELECTRICITY LOAD FORECASTING — COMPLETE ML PIPELINE  v13
# Dataset  : Power Demand 2021–2024 | 5-minute intervals
# Target   : Power Demand (MW)
# Models   : Naive Baseline, Linear Regression, Decision Tree,
#            Random Forest, KNN, Gradient Boosting, XGBoost,
#            XGBoost (Tuned), LightGBM, LightGBM (Tuned),
#            Ensemble (XGB+LGBM), LSTM
# Tuning   : Optuna — XGBoost & LightGBM (50 trials × 3-fold)
# Ensemble : Inverse-RMSE weighted blend of tuned XGB + LGBM
# Metrics  : MAE, RMSE, R², MAPE
# Plots    : Research-style seaborn / matplotlib plots
#            Plot 1  : Actual vs Predicted — Last 7 Days (lineplot)
#            Plot 2a : Residual Bivariate — Simple ML Models
#            Plot 2b : Residual Bivariate — Advanced Models
#            Plot 3  : Feature Importances (RF / XGB / LGBM)
#            Plot 4  : MAE / RMSE / MAPE model comparison (barplot)
#            Plot 5  : Zoom — Default models (lineplot)
#            Plot 6  : Ensemble vs Components zoom (lineplot)
#            Plot 7  : Ensemble residual distribution (hist+KDE)
#            Plot 8  : Optuna convergence — XGBoost (scatter)
#            Plot 9  : Optuna convergence — LightGBM (scatter)
#            Plot 10 : R² comparison (horizontal barplot)
#            Plot 11 : Top 3 models zoomed — Last 24 h (lineplot)
#            Plot 12 : LSTM vs Ensemble zoom (lineplot)
#            Plot 13 : LSTM training loss curves (lineplot)
#            Plot 14 : Correlation heatmap (selected features)
# v13 Changes:
#   ✅ Removed console print loops for feature importances
#   ✅ Replaced giant residual subplot grid with two polished
#      bivariate residual plots (scatter + hist + KDE contours)
#   ✅ Added correlation heatmap (Plot 14)
#   ✅ Cleaner section headers and output
# Author   : Academic Project — Clean & Modular Pipeline
# Env      : Kaggle (TensorFlow 2.x pre-installed)
# ============================================================


# ============================================================
# SECTION 1: IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings

from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from xgboost                 import XGBRegressor
import lightgbm              as lgb
import optuna
from optuna.samplers         import TPESampler

import tensorflow as tf
from tensorflow.keras.models           import Sequential
from tensorflow.keras.layers           import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks        import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers       import Adam

optuna.logging.set_verbosity(optuna.logging.WARNING)   # suppress per-trial noise
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")                      # suppress TF info noise

sns.set_theme(style="darkgrid", palette="deep")
sns.set_context("talk", font_scale=0.85)

print(f"✅ All libraries imported successfully.  TensorFlow {tf.__version__}\n")


# ============================================================
# SECTION 2: LOAD & CLEAN DATASET
# ============================================================
import os
# Robust path resolution for Kaggle or local environments
possible_paths = [
    "/kaggle/input/datasets/yug201/delhi-5-minute-electricity-demand-for-forecasting/powerdemand_5min_2021_to_2024_with weather.csv",
    "/kaggle/input/delhi-5-minute-electricity-demand-for-forecasting/powerdemand_5min_2021_to_2024_with weather.csv",
    "../../electricityConsumptionAndProductioction.csv",
    "../electricityConsumptionAndProductioction.csv",
    "electricityConsumptionAndProductioction.csv",
    "datasets/powerdemand_5min_2021_to_2024_with weather.csv"
]
CSV_PATH = None
for p in possible_paths:
    if os.path.exists(p):
        CSV_PATH = p
        break

if CSV_PATH is None:
    raise FileNotFoundError("❌ Could not find the dataset CSV file in any expected path!")

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# Drop irrelevant index column and pre-computed rolling average
df.drop(columns=["Unnamed: 0", "moving_avg_3"], inplace=True)

# Rename target
df = df.rename(columns={"Power demand": "load"})

# Parse datetime and sort
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

print(f"📂 Dataset loaded : {len(df):,} rows")
print(f"   Date range     : {df['datetime'].min().date()} → {df['datetime'].max().date()}")
print(f"   Load range     : {df['load'].min():.1f} – {df['load'].max():.1f} MW\n")


# ============================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================
print("⚙️  Engineering features …")

# ── 3a. Fix wind direction (circular 0–360°) ────────────────
# Raw wdir is meaningless as a linear number (359° ≈ 1°)
df["wdir"] = df["wdir"].ffill()
df["wdir_sin"] = np.sin(2 * np.pi * df["wdir"] / 360)
df["wdir_cos"] = np.cos(2 * np.pi * df["wdir"] / 360)
df.drop(columns=["wdir"], inplace=True)

# ── 3b. Cyclical time encoding ───────────────────────────────
# sin/cos preserves circular continuity (23:55 → 00:00)
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Drop raw and redundant time columns
df.drop(columns=["hour", "month", "minute", "day", "year"], inplace=True)

# ── 3c. Binary time flags ────────────────────────────────────
df["weekday"]      = df["datetime"].dt.weekday
df["weekend"]      = (df["weekday"] >= 5).astype(int)
df["is_peak_hour"] = df["datetime"].dt.hour.between(18, 21).astype(int)
df["is_day"]       = df["datetime"].dt.hour.between(6, 18).astype(int)

# ── 3d. Interaction features ─────────────────────────────────
# temp_hour  : captures gradual intraday temperature effect
df["temp_hour"] = df["temp"] * df["datetime"].dt.hour

# temp_x_peak : binary gate — fires only during 18–21 peak window
# AC/heating load spikes when hot/cold AND it is peak time
df["temp_x_peak"] = df["temp"] * df["is_peak_hour"]

# ── 3e. Lag features (lag_1 removed — causes 99% dominance) ─
df["lag_12"]   = df["load"].shift(12)    # 1 hour ago
df["lag_288"]  = df["load"].shift(288)   # 24 hours ago
df["lag_2016"] = df["load"].shift(2016)  # 7 days ago

# ── 3f. Rolling statistics ───────────────────────────────────
df["roll_mean_12"] = df["load"].shift(1).rolling(12).mean()  # 1-hr rolling mean
df["roll_std_12"]  = df["load"].shift(1).rolling(12).std()   # 1-hr rolling std
df["roll_max_12"]  = df["load"].shift(1).rolling(12).max()   # 1-hr rolling peak
df["roll_min_12"]  = df["load"].shift(1).rolling(12).min()   # 1-hr rolling trough

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# ── Final feature set (25 features) ─────────────────────────
FEATURES = [
    # Cyclical time
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    # Binary time flags
    "weekday", "weekend", "is_peak_hour", "is_day",
    # Weather
    "temp", "dwpt", "rhum", "wspd", "pres",
    "wdir_sin", "wdir_cos",
    # Interaction
    "temp_hour", "temp_x_peak",
    # Lag
    "lag_12", "lag_288", "lag_2016",
    # Rolling
    "roll_mean_12", "roll_std_12", "roll_max_12", "roll_min_12",
]
TARGET = "load"

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise KeyError(f"❌ Missing features: {missing}")

print(f"✅ {len(FEATURES)} features confirmed.")
print(f"   Final dataset  : {len(df):,} rows\n")
print("   Features used  :")
for i, f in enumerate(FEATURES, 1):
    print(f"     {i:2}. {f}")
print()


# ============================================================
# SECTION 4: TRAIN / TEST SPLIT (Chronological)
# ============================================================
# Strict chronological split — no data leakage
# Train: 2021–2023  |  Test: 2024
SPLIT_DATE = "2024-01-01"

train_mask = df["datetime"] < SPLIT_DATE
test_mask  = df["datetime"] >= SPLIT_DATE

X_train    = df.loc[train_mask, FEATURES]
y_train    = df.loc[train_mask, TARGET]
X_test     = df.loc[test_mask,  FEATURES]
y_test     = df.loc[test_mask,  TARGET]
dates_test = df.loc[test_mask,  "datetime"].reset_index(drop=True)

print(f"📊 Train : {len(X_train):,} samples  (2021-01-01 → 2023-12-31)")
print(f"   Test  : {len(X_test):,}  samples  (2024-01-01 → end)\n")

# Scale features for distance/linear models only
# Tree-based models (RF, XGB, LGBM) do NOT need scaling
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ============================================================
# SECTION 5: NAIVE BASELINE
# ============================================================
# Prediction = load from same time yesterday (lag_288)
# All ML models must beat this to be considered useful
naive_pred = X_test["lag_288"].values
print("📏 Naive baseline: prediction = load(t-288) [same time yesterday]\n")


# ============================================================
# SECTION 6: MODEL DEFINITIONS
# ============================================================
# Format: name → (model_object, "scaled" | "raw")
# scaled = StandardScaler applied before fit  (LR, KNN)
# raw    = original feature scale used        (tree-based)

models = {
    "Linear Regression": (LinearRegression(),                                 "scaled"),

    "Decision Tree":     (DecisionTreeRegressor(max_depth=10,
                                                min_samples_leaf=10,
                                                random_state=42),             "raw"),

    "Random Forest":     (RandomForestRegressor(n_estimators=150,
                                                max_depth=15,
                                                min_samples_leaf=4,
                                                n_jobs=-1,
                                                random_state=42),             "raw"),

    "KNN":               (KNeighborsRegressor(n_neighbors=10,
                                              weights="distance",
                                              n_jobs=-1),                     "scaled"),

    "Gradient Boosting": (GradientBoostingRegressor(n_estimators=200,
                                                    max_depth=5,
                                                    learning_rate=0.05,
                                                    subsample=0.8,
                                                    random_state=42),         "raw"),

    # XGBoost: depth-wise tree growth, L1+L2 regularisation
    "XGBoost":           (XGBRegressor(n_estimators=300,
                                       max_depth=6,
                                       learning_rate=0.05,
                                       subsample=0.8,
                                       colsample_bytree=0.8,
                                       reg_alpha=0.1,
                                       reg_lambda=1.0,
                                       n_jobs=-1,
                                       random_state=42,
                                       verbosity=0),                          "raw"),

    # LightGBM: leaf-wise growth, histogram binning — faster on large datasets
    "LightGBM":          (lgb.LGBMRegressor(n_estimators=300,
                                             max_depth=6,
                                             learning_rate=0.05,
                                             subsample=0.8,
                                             colsample_bytree=0.8,
                                             reg_alpha=0.1,
                                             reg_lambda=1.0,
                                             n_jobs=-1,
                                             random_state=42,
                                             verbose=-1),                     "raw"),
}


# ============================================================
# SECTION 7: TRAIN ALL MODELS + EVALUATE
# ============================================================
def compute_metrics(name, y_true, y_pred):
    """Compute MAE, RMSE, R², MAPE for a model."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred))
                          / np.array(y_true))) * 100
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2, "MAPE (%)": mape}

all_metrics = []
all_preds   = {}

print("🤖 Training models …\n")

# Naive Baseline
m = compute_metrics("Naive Baseline", y_test, naive_pred)
all_metrics.append(m)
all_preds["Naive Baseline"] = naive_pred
print(f"  ✅ {'Naive Baseline':<22} MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}"
      f"  R²={m['R²']:.4f}  MAPE={m['MAPE (%)']:.2f}%")

# ML Models
for name, (model, data_type) in models.items():
    X_tr = X_train_sc if data_type == "scaled" else X_train
    X_te = X_test_sc  if data_type == "scaled" else X_test
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    m = compute_metrics(name, y_test, preds)
    all_metrics.append(m)
    all_preds[name] = preds
    print(f"  ✅ {name:<22} MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}"
          f"  R²={m['R²']:.4f}  MAPE={m['MAPE (%)']:.2f}%")

# Interim results table
results_df = pd.DataFrame(all_metrics).set_index("Model").round(3)
results_df = results_df.sort_values("RMSE")
best_model = results_df.index[0]

print("\n" + "=" * 72)
print("  BASELINE RESULTS — sorted by RMSE ↑")
print("=" * 72)
print(results_df.to_string())
print("=" * 72)
print(f"\n  🏆 Best so far: {best_model}\n")


# ============================================================
# SECTION 8: TIME SERIES CROSS-VALIDATION — XGBoost
# ============================================================
print("🔁 TimeSeriesSplit cross-validation on XGBoost (5 folds) …")

xgb_cv = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       n_jobs=-1, random_state=42, verbosity=0)
tscv     = TimeSeriesSplit(n_splits=5)
cv_maes  = []
X_full   = df[FEATURES]
y_full   = df[TARGET]

for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_full), 1):
    xgb_cv.fit(X_full.iloc[tr_idx], y_full.iloc[tr_idx])
    preds = xgb_cv.predict(X_full.iloc[te_idx])
    mae   = mean_absolute_error(y_full.iloc[te_idx], preds)
    cv_maes.append(mae)
    print(f"   Fold {fold}: MAE = {mae:.2f} MW")

print(f"\n   CV Mean MAE : {np.mean(cv_maes):.2f} MW")
print(f"   CV Std  MAE : {np.std(cv_maes):.2f} MW\n")


# ============================================================
# SECTION 8b: OPTUNA TUNING — XGBoost
# Searches : n_estimators, max_depth, learning_rate
# Strategy : 3-fold TimeSeriesSplit CV, minimise mean MAE
# ============================================================
print("🔍 Optuna tuning — XGBoost (50 trials × 3-fold CV) …\n")

OPTUNA_TRIALS = 50
OPTUNA_CV     = 3
_tscv_opt     = TimeSeriesSplit(n_splits=OPTUNA_CV)


def _xgb_objective(trial: optuna.Trial) -> float:
    """Objective: mean CV-MAE for a candidate XGBoost configuration."""
    params = dict(
        n_estimators         = trial.suggest_int("n_estimators",   100, 600, step=50),
        max_depth            = trial.suggest_int("max_depth",         3,  10),
        learning_rate        = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        subsample            = 0.8,
        colsample_bytree     = 0.8,
        reg_alpha            = 0.1,
        reg_lambda           = 1.0,
        n_jobs               = -1,
        random_state         = 42,
        verbosity            = 0,
        early_stopping_rounds= 20,
        eval_metric          = "mae",
    )
    fold_maes = []
    for tr_idx, va_idx in _tscv_opt.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        mdl = XGBRegressor(**params)
        mdl.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        fold_maes.append(mean_absolute_error(y_va, mdl.predict(X_va)))
    return float(np.mean(fold_maes))


xgb_study = optuna.create_study(direction="minimize",
                                  sampler=TPESampler(seed=42),
                                  study_name="xgb_load_forecast")
xgb_study.optimize(_xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

xgb_best_params = xgb_study.best_params
xgb_best_cv_mae = xgb_study.best_value

print(f"\n✅ XGBoost Optuna finished — best CV-MAE: {xgb_best_cv_mae:.4f} MW")
print("   Best hyperparameters:")
for k, v in xgb_best_params.items():
    print(f"     {k:<22} = {v}")

# Re-train tuned XGBoost on full training set
xgb_tuned = XGBRegressor(
    **xgb_best_params,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, random_state=42, verbosity=0,
)
xgb_tuned.fit(X_train, y_train)
tuned_preds  = xgb_tuned.predict(X_test)
m_tuned      = compute_metrics("XGBoost (Tuned)", y_test, tuned_preds)
all_metrics.append(m_tuned)
all_preds["XGBoost (Tuned)"] = tuned_preds

# Rebuild results
results_df = pd.DataFrame(all_metrics).set_index("Model").round(3)
results_df = results_df.sort_values("RMSE")
best_model = results_df.index[0]

print(f"\n📊 XGBoost default vs Tuned — Test 2024")
print(f"   {'Metric':<10}  {'Default':>10}  {'Tuned':>10}")
print(f"   {'-'*34}")
for metric in ["MAE", "RMSE", "R²", "MAPE (%)"]:
    dv = results_df.loc["XGBoost", metric]
    tv = results_df.loc["XGBoost (Tuned)", metric]
    print(f"   {metric:<10}  {dv:>10.4f}  {tv:>10.4f}")

# XGBoost Optuna convergence data calculation
xgb_trial_nums  = [t.number + 1            for t in xgb_study.trials]
xgb_trial_maes  = [t.value                 for t in xgb_study.trials]
xgb_running_min = pd.Series(xgb_trial_maes).cummin().tolist()

# ============================================================
# SECTION 8c: OPTUNA TUNING — LightGBM
# Searches : num_leaves, min_child_samples, n_estimators,
#            learning_rate, max_depth, colsample_bytree
# Strategy : 3-fold TimeSeriesSplit CV, minimise mean MAE
# ============================================================
print("🔍 Optuna tuning — LightGBM (50 trials × 3-fold CV) …\n")

LGBM_TRIALS = 50
LGBM_CV     = 3
_tscv_lgbm  = TimeSeriesSplit(n_splits=LGBM_CV)


def _lgbm_objective(trial: optuna.Trial) -> float:
    """
    Objective function for LightGBM Optuna search.

    Key parameters:
      num_leaves        — primary complexity knob (leaf-wise growth)
      min_child_samples — regulariser, min data per leaf
      n_estimators      — number of boosting rounds
      learning_rate     — step size (log scale)
      max_depth         — hard cap on tree depth
      colsample_bytree  — feature fraction per tree
    """
    params = dict(
        num_leaves        = trial.suggest_int("num_leaves",         20, 300),
        min_child_samples = trial.suggest_int("min_child_samples",   5, 100),
        n_estimators      = trial.suggest_int("n_estimators",       100, 600, step=50),
        learning_rate     = trial.suggest_float("learning_rate",    0.005, 0.3, log=True),
        max_depth         = trial.suggest_int("max_depth",            3,  12),
        colsample_bytree  = trial.suggest_float("colsample_bytree",  0.5, 1.0),
        subsample         = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        n_jobs            = -1,
        random_state      = 42,
        verbose           = -1,
    )
    fold_maes = []
    for tr_idx, va_idx in _tscv_lgbm.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(
            X_tr, y_tr,
            eval_set  = [(X_va, y_va)],
            callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False),
                         lgb.log_evaluation(period=-1)],
        )
        fold_maes.append(mean_absolute_error(y_va, mdl.predict(X_va)))
    return float(np.mean(fold_maes))


lgbm_study = optuna.create_study(direction="minimize",
                                   sampler=TPESampler(seed=42),
                                   study_name="lgbm_load_forecast")
lgbm_study.optimize(_lgbm_objective, n_trials=LGBM_TRIALS, show_progress_bar=True)

lgbm_best_params = lgbm_study.best_params
lgbm_best_cv_mae = lgbm_study.best_value

print(f"\n✅ LightGBM Optuna finished — best CV-MAE: {lgbm_best_cv_mae:.4f} MW")
print("   Best hyperparameters:")
for k, v in lgbm_best_params.items():
    print(f"     {k:<22} = {v}")

# Re-train tuned LightGBM on full training set
lgbm_tuned = lgb.LGBMRegressor(
    **lgbm_best_params,
    subsample=0.8, reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, random_state=42, verbose=-1,
)
lgbm_tuned.fit(X_train, y_train)
lgbm_tuned_preds = lgbm_tuned.predict(X_test)
m_lgbm_tuned     = compute_metrics("LightGBM (Tuned)", y_test, lgbm_tuned_preds)
all_metrics.append(m_lgbm_tuned)
all_preds["LightGBM (Tuned)"] = lgbm_tuned_preds

# Rebuild results
results_df = pd.DataFrame(all_metrics).set_index("Model").round(3)
results_df = results_df.sort_values("RMSE")
best_model = results_df.index[0]

print(f"\n📊 LightGBM default vs Tuned — Test 2024")
print(f"   {'Metric':<10}  {'Default':>10}  {'Tuned':>10}")
print(f"   {'-'*34}")
lgbm_default_m = compute_metrics("LightGBM", y_test, all_preds["LightGBM"])
for metric_key in ["MAE", "RMSE", "R²", "MAPE (%)"]:
    print(f"   {metric_key:<10}  {lgbm_default_m[metric_key]:>10.4f}"
          f"  {m_lgbm_tuned[metric_key]:>10.4f}")

# LightGBM Optuna convergence data calculation
lgbm_trial_nums  = [t.number + 1              for t in lgbm_study.trials]
lgbm_trial_maes  = [t.value                   for t in lgbm_study.trials]
lgbm_running_min = pd.Series(lgbm_trial_maes).cummin().tolist()



# ============================================================
# SECTION 8d: ENSEMBLE — XGBoost (Tuned) + LightGBM (Tuned)
# Strategy : Inverse-RMSE weighted average
# Why XGB+LGBM only?
#   • Different tree-growth strategies → partially uncorrelated errors
#   • Both outperform RF/others → blending lifts accuracy further
#   • Including weaker models would pull the ensemble down
# ============================================================
print("🔗 Building Ensemble: XGBoost (Tuned) + LightGBM (Tuned) …")

xgb_rmse  = results_df.loc["XGBoost (Tuned)",  "RMSE"]
lgbm_rmse = results_df.loc["LightGBM (Tuned)", "RMSE"]

# Inverse-RMSE: lower RMSE → higher weight (automatic, data-driven)
inv_xgb  = 1.0 / xgb_rmse
inv_lgbm = 1.0 / lgbm_rmse
total    = inv_xgb + inv_lgbm
w_xgb    = inv_xgb  / total
w_lgbm   = inv_lgbm / total

print(f"   XGBoost (Tuned)  weight : {w_xgb:.4f}  (RMSE={xgb_rmse:.2f} MW)")
print(f"   LightGBM (Tuned) weight : {w_lgbm:.4f}  (RMSE={lgbm_rmse:.2f} MW)")

ensemble_preds = w_xgb * tuned_preds + w_lgbm * lgbm_tuned_preds
m_ensemble     = compute_metrics("Ensemble (XGB+LGBM)", y_test, ensemble_preds)
all_metrics.append(m_ensemble)
all_preds["Ensemble (XGB+LGBM)"] = ensemble_preds

# Final results table
results_df = pd.DataFrame(all_metrics).set_index("Model").round(3)
results_df = results_df.sort_values("RMSE")
best_model = results_df.index[0]

print(f"\n📊 Component vs Ensemble — Test 2024")
print(f"   {'Model':<25}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}  {'MAPE%':>8}")
print(f"   {'-'*62}")
for mname in ["XGBoost (Tuned)", "LightGBM (Tuned)", "Ensemble (XGB+LGBM)"]:
    row    = results_df.loc[mname]
    marker = "  ← best" if mname == best_model else ""
    print(f"   {mname:<25}  {row['MAE']:>8.2f}  {row['RMSE']:>8.2f}"
          f"  {row['R²']:>8.4f}  {row['MAPE (%)']:>7.2f}%{marker}")


# ============================================================
# SECTION 8e: LSTM — Sequence-to-One Forecasting
# ─────────────────────────────────────────────────────────────
# Architecture : Input(lookback, n_features)
#                → LSTM(128, return_sequences=True)
#                → Dropout(0.2)
#                → LSTM(64, return_sequences=False)
#                → Dropout(0.2)
#                → Dense(32, relu)
#                → Dense(1)
#
# Lookback     : 24 steps  (2 hours of 5-min intervals)
#                Balances context vs sequence-matrix size.
#                Captures intraday load cycles visible at 2 h.
#
# Scaling      : MinMaxScaler on target + all features.
#                LSTMs are sensitive to scale; normalise to [0,1].
#
# Split        : Same chronological 2021-2023 train / 2024 test.
#
# Training     : Adam(lr=1e-3), MSE loss, batch=256, max 50 epochs
#                EarlyStopping(patience=5) restores best weights.
#                ReduceLROnPlateau(patience=3) halves lr on plateau.
#
# Inference    : Predict in a single vectorised call — no loop.
# ============================================================
print("\n🧠 Training LSTM …")

import gc
from sklearn.preprocessing import MinMaxScaler

LOOKBACK   = 24     # 24 × 5-min = 2-hour context window
BATCH_SIZE = 256
MAX_EPOCHS = 50

# ── 8e-1. Scale all features + target jointly ────────────────
# Fit scaler ONLY on training data to prevent leakage
lstm_feature_cols = FEATURES + [TARGET]   # scale target too

lstm_scaler = MinMaxScaler(feature_range=(0, 1))

# Build contiguous arrays aligned to the original df index
train_idx = df.index[df["datetime"] < SPLIT_DATE]
test_idx  = df.index[df["datetime"] >= SPLIT_DATE]

lstm_scaler.fit(df.loc[train_idx, lstm_feature_cols])
scaled_all = lstm_scaler.transform(df[lstm_feature_cols])

scaled_df       = pd.DataFrame(scaled_all, columns=lstm_feature_cols, index=df.index)
target_col_idx  = lstm_feature_cols.index(TARGET)   # position of 'load' in scaled array

# ── 8e-2. Build sequence matrices ────────────────────────────
def build_sequences(data: np.ndarray, lookback: int):
    """
    Returns (X_seq, y_seq) where
      X_seq.shape = (N - lookback, lookback, n_features)
      y_seq.shape = (N - lookback,)
    """
    X_seqs, y_seqs = [], []
    for i in range(lookback, len(data)):
        X_seqs.append(data[i - lookback : i, :])   # all features
        y_seqs.append(data[i, target_col_idx])      # scaled load at step i
    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)


# Separate scaled arrays for train / test
scaled_train = scaled_df.loc[train_idx].values
scaled_test_full = scaled_df.values   # need full prefix for test sequences

# Build train sequences from training portion only
X_lstm_train, y_lstm_train = build_sequences(scaled_train, LOOKBACK)

# Build test sequences: the first LOOKBACK steps of the test window
# require the tail of the training data as context
boundary = train_idx[-1]                          # last training row position
boundary_pos = df.index.get_loc(boundary)         # integer position in df
context_start = boundary_pos - LOOKBACK + 1       # include lookback context
scaled_ctx_and_test = scaled_df.values[context_start:]

X_lstm_test, y_lstm_test = build_sequences(scaled_ctx_and_test, LOOKBACK)

# y_lstm_test should align with y_test (both cover 2024 test period)
assert len(X_lstm_test) == len(y_test), (
    f"LSTM test length mismatch: {len(X_lstm_test)} vs {len(y_test)}")

n_features = X_lstm_train.shape[2]
print(f"   Train sequences : {X_lstm_train.shape}")
print(f"   Test  sequences : {X_lstm_test.shape}")
print(f"   Features        : {n_features}  |  Lookback: {LOOKBACK} steps")

# ── 8e-3. Build model ─────────────────────────────────────────
tf.random.set_seed(42)

lstm_model = Sequential([
    Input(shape=(LOOKBACK, n_features)),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64,  return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1),
], name="LSTM_LoadForecast")

lstm_model.compile(
    optimizer = Adam(learning_rate=1e-3),
    loss      = "mse",
)
lstm_model.summary()

# ── 8e-4. Callbacks ───────────────────────────────────────────
callbacks_lstm = [
    EarlyStopping(monitor="val_loss", patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=3, min_lr=1e-6, verbose=1),
]

# ── 8e-5. Train ───────────────────────────────────────────────
lstm_history = lstm_model.fit(
    X_lstm_train, y_lstm_train,
    validation_split = 0.1,
    epochs           = MAX_EPOCHS,
    batch_size       = BATCH_SIZE,
    callbacks        = callbacks_lstm,
    verbose          = 1,
)

print(f"\n   Stopped at epoch {len(lstm_history.history['loss'])}")

# ── 8e-6. Predict & inverse-transform ────────────────────────
lstm_scaled_preds = lstm_model.predict(X_lstm_test, batch_size=BATCH_SIZE, verbose=0).flatten()

# Inverse-transform: reconstruct a dummy full row with 0s, replace target column
dummy = np.zeros((len(lstm_scaled_preds), len(lstm_feature_cols)), dtype=np.float32)
dummy[:, target_col_idx] = lstm_scaled_preds
lstm_preds_mw = lstm_scaler.inverse_transform(dummy)[:, target_col_idx]

# ── 8e-7. Evaluate ───────────────────────────────────────────
m_lstm = compute_metrics("LSTM", y_test, lstm_preds_mw)
all_metrics.append(m_lstm)
all_preds["LSTM"] = lstm_preds_mw

results_df = pd.DataFrame(all_metrics).set_index("Model").round(3)
results_df = results_df.sort_values("RMSE")
best_model = results_df.index[0]

print(f"\n✅ LSTM — Test 2024")
print(f"   MAE  = {m_lstm['MAE']:.2f} MW")
print(f"   RMSE = {m_lstm['RMSE']:.2f} MW")
print(f"   R²   = {m_lstm['R²']:.4f}")
print(f"   MAPE = {m_lstm['MAPE (%)']:.2f}%\n")

# Free GPU memory
gc.collect()
tf.keras.backend.clear_session()


# ============================================================
# SECTION 9: FEATURE IMPORTANCE
# ============================================================
rf_model   = models["Random Forest"][0]
xgb_model  = models["XGBoost"][0]
lgbm_model = models["LightGBM"][0]

rf_imp   = pd.Series(rf_model.feature_importances_,   index=FEATURES).sort_values(ascending=False)
xgb_imp  = pd.Series(xgb_model.feature_importances_,  index=FEATURES).sort_values(ascending=False)
lgbm_imp = pd.Series(lgbm_model.feature_importances_, index=FEATURES).sort_values(ascending=False)

print("✅ Feature importances computed for RF, XGBoost, LightGBM.")


# ============================================================
# SECTION 10: VISUALIZATION — RESEARCH-STYLE PLOTS
# ============================================================
PLOT_PERIODS = 7 * 288    # last 7 days of test set (5-min intervals)
ZOOM         = 2 * 288    # last 2 days for zoom plots

plot_dates  = dates_test.values[-PLOT_PERIODS:]
plot_actual = y_test.values[-PLOT_PERIODS:]

COLORS = {
    "Naive Baseline":      "#95A5A6",
    "Linear Regression":   "#E74C3C",
    "Decision Tree":       "#F39C12",
    "Random Forest":       "#27AE60",
    "KNN":                 "#8E44AD",
    "Gradient Boosting":   "#2980B9",
    "XGBoost":             "#E67E22",
    "XGBoost (Tuned)":     "#C0392B",
    "LightGBM":            "#1ABC9C",
    "LightGBM (Tuned)":    "#148F77",
    "Ensemble (XGB+LGBM)": "#6C3483",
    "LSTM":                "#D4145A",   # deep pink — visually distinct
}


# ── Plot 1: All models — Actual vs Predicted (7 days) ────────
rows = []
for t, a in zip(plot_dates, plot_actual):
    rows.append({"datetime": t, "Power Demand (MW)": a, "Model": "Actual"})
for name, preds in all_preds.items():
    for t, p in zip(plot_dates, preds[-PLOT_PERIODS:]):
        rows.append({"datetime": t, "Power Demand (MW)": p, "Model": name})

plot1_df = pd.DataFrame(rows)
plot1_df["datetime"] = pd.to_datetime(plot1_df["datetime"])
color_map = {"Actual": "#2C3E50", **COLORS}

fig, ax = plt.subplots(figsize=(22, 6))
sns.lineplot(data=plot1_df, x="datetime", y="Power Demand (MW)",
             hue="Model", palette=color_map, linewidth=0.9, ax=ax)
for line in ax.get_lines():
    if line.get_label() == "Actual":
        line.set_linewidth(2.5)
        line.set_zorder(10)
ax.set_title("Actual vs Predicted Power Demand — Last 7 Days (2024)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("Power Demand (MW)")
ax.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.DayLocator())
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("plot1_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot1_actual_vs_predicted.png")


# ── Plot 2a: Residual Bivariate — Simple ML Models ───────────
# x = predicted load, y = residuals (actual − predicted)
# Each model: scatter + 2D KDE contours + marginal histograms
# Uses GridSpec to keep a consistent layout across models
SIMPLE_MODELS   = ["Linear Regression", "Decision Tree",
                   "Random Forest", "KNN", "Gradient Boosting"]
ADVANCED_MODELS = ["XGBoost (Tuned)", "LightGBM (Tuned)",
                   "Ensemble (XGB+LGBM)", "LSTM"]

def _bivariate_residual_grid(model_names, title, filename):
    """
    Draw a polished bivariate residual figure for a list of models.

    Each panel contains:
      • scatter of (predicted, residual) — semi-transparent dots
      • sns.kdeplot contour overlay (2D density)
      • horizontal zero-residual reference line
      • model name + MAE annotation
    All panels share the same x / y limits for fair comparison.
    """
    n   = len(model_names)
    # Compute global axis limits across all models in this group
    all_preds_grp = [all_preds[m] for m in model_names]
    all_resid_grp = [y_test.values - p for p in all_preds_grp]

    x_min = min(p.min() for p in all_preds_grp)
    x_max = max(p.max() for p in all_preds_grp)
    y_vals_concat = np.concatenate(all_resid_grp)
    y_lim_abs = np.percentile(np.abs(y_vals_concat), 98) * 1.15   # clip extreme outliers

    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 5.5 * nrows),
                             sharex=False, sharey=False)
    axes = np.array(axes).flatten()

    for i, mname in enumerate(model_names):
        ax       = axes[i]
        preds_m  = all_preds[mname]
        resid_m  = y_test.values - preds_m
        color    = COLORS.get(mname, "#AAB7B8")
        mae_val  = results_df.loc[mname, "MAE"]

        # ── scatter ───────────────────────────────────────────
        ax.scatter(preds_m, resid_m,
                   color=color, alpha=0.12, s=4, rasterized=True,
                   label="_nolegend_")

        # ── 2D KDE contours ───────────────────────────────────
        try:
            sns.kdeplot(x=preds_m, y=resid_m,
                        levels=6, linewidths=0.9,
                        color=color, alpha=0.75, ax=ax)
        except Exception:
            pass   # skip KDE if bandwidth fails on sparse data

        # ── zero-residual reference line ──────────────────────
        ax.axhline(0, color="#2C3E50", linewidth=1.2,
                   linestyle="--", alpha=0.8)

        # ── axis labels & title ───────────────────────────────
        ax.set_xlabel("Predicted Load (MW)", fontsize=10)
        ax.set_ylabel("Residual — Actual − Predicted (MW)", fontsize=10)
        ax.set_title(f"{mname}\nMAE = {mae_val:.2f} MW",
                     fontsize=11, fontweight="bold", pad=8)
        ax.set_xlim(x_min * 0.97, x_max * 1.03)
        ax.set_ylim(-y_lim_abs, y_lim_abs)
        ax.tick_params(labelsize=9)

        # ── mean residual annotation ───────────────────────────
        mean_res = np.mean(resid_m)
        ax.annotate(f"mean residual = {mean_res:+.1f} MW",
                    xy=(0.03, 0.05), xycoords="axes fraction",
                    fontsize=8.5, color="#555555",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", ec="none", alpha=0.7))

    # Hide any surplus axes
    for j in range(len(model_names), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: {filename}")


_bivariate_residual_grid(
    model_names = SIMPLE_MODELS,
    title       = "Residual Bivariate Plot — Simple ML Models\n"
                  "(x = Predicted Load,  y = Residual,  contours = 2-D KDE density)",
    filename    = "plot2a_residual_bivariate_simple.png",
)

# ── Plot 2b: Residual Bivariate — Advanced Models ────────────
_bivariate_residual_grid(
    model_names = ADVANCED_MODELS,
    title       = "Residual Bivariate Plot — Advanced Models\n"
                  "(x = Predicted Load,  y = Residual,  contours = 2-D KDE density)",
    filename    = "plot2b_residual_bivariate_advanced.png",
)


# ── Plot 3: Feature Importances — RF vs XGBoost vs LightGBM ──
fig, axes = plt.subplots(1, 3, figsize=(26, 8))
for ax, (title, imp, pal) in zip(axes, [
    ("Random Forest", rf_imp.head(15),   "Greens_r"),
    ("XGBoost",       xgb_imp.head(15),  "Oranges_r"),
    ("LightGBM",      lgbm_imp.head(15), "BuGn_r"),
]):
    imp_df = imp.reset_index()
    imp_df.columns = ["Feature", "Importance"]
    sns.barplot(data=imp_df, y="Feature", x="Importance",
                palette=pal, edgecolor="white", linewidth=0.6, ax=ax, orient="h")
    for bar, val in zip(ax.patches, imp_df["Importance"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8)
    ax.set_title(f"{title} — Top 15 Features", fontsize=11, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")

plt.suptitle("Feature Importances: Random Forest vs XGBoost vs LightGBM",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plot3_feature_importances.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot3_feature_importances.png")


# ── Plot 4: Model Comparison — MAE / RMSE / MAPE ─────────────
metrics_to_plot = ["MAE", "RMSE", "MAPE (%)"]
compare_long = results_df[metrics_to_plot].reset_index().melt(
    id_vars="Model", var_name="Metric", value_name="Value")

fig, axes = plt.subplots(1, 3, figsize=(24, 7))
for ax, metric in zip(axes, metrics_to_plot):
    sub        = compare_long[compare_long["Metric"] == metric].sort_values("Value")
    bar_colors = [COLORS.get(m, "#AAB7B8") for m in sub["Model"]]
    bars       = sns.barplot(data=sub, x="Model", y="Value",
                             palette=bar_colors, edgecolor="white",
                             linewidth=0.6, ax=ax)
    for bar, val in zip(ax.patches, sub["Value"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.patches[0].set_edgecolor("gold")
    ax.patches[0].set_linewidth(3)
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=40)

plt.suptitle("Model Comparison — Test Set 2024  (Gold border = best)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plot4_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot4_model_comparison.png")


# ── Plot 5: Zoom — XGBoost vs RF vs LightGBM (2 days) ────────
zoom_rows = []
for t, a in zip(plot_dates[-ZOOM:], plot_actual[-ZOOM:]):
    zoom_rows.append({"datetime": t, "Power Demand (MW)": a, "Model": "Actual"})
for name in ["XGBoost", "Random Forest", "LightGBM"]:
    for t, p in zip(plot_dates[-ZOOM:], all_preds[name][-ZOOM:]):
        zoom_rows.append({"datetime": t, "Power Demand (MW)": p, "Model": name})

zoom_df = pd.DataFrame(zoom_rows)
zoom_df["datetime"] = pd.to_datetime(zoom_df["datetime"])
zoom_pal = {"Actual": "#2C3E50", "XGBoost": "#E67E22",
            "Random Forest": "#27AE60", "LightGBM": "#1ABC9C"}

fig, ax = plt.subplots(figsize=(16, 6))
sns.lineplot(data=zoom_df, x="datetime", y="Power Demand (MW)",
             hue="Model", palette=zoom_pal, linewidth=1.2, ax=ax)
for line in ax.get_lines():
    if line.get_label() == "Actual":
        line.set_linewidth(2.5)
        line.set_zorder(10)
ax.set_title("XGBoost vs Random Forest vs LightGBM — Last 2 Days",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("Power Demand (MW)")
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("plot5_default_models_zoom.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot5_default_models_zoom.png")


# ── Plot 6: Ensemble vs Components — Zoom (2 days) ───────────
ens_rows = []
for t, a in zip(plot_dates[-ZOOM:], plot_actual[-ZOOM:]):
    ens_rows.append({"datetime": t, "Power Demand (MW)": a, "Model": "Actual"})
for mname in ["XGBoost (Tuned)", "LightGBM (Tuned)", "Ensemble (XGB+LGBM)"]:
    for t, p in zip(plot_dates[-ZOOM:], all_preds[mname][-ZOOM:]):
        ens_rows.append({"datetime": t, "Power Demand (MW)": p, "Model": mname})

ens_zoom_df = pd.DataFrame(ens_rows)
ens_zoom_df["datetime"] = pd.to_datetime(ens_zoom_df["datetime"])
ens_pal = {"Actual":                "#2C3E50",
           "XGBoost (Tuned)":       "#C0392B",
           "LightGBM (Tuned)":      "#148F77",
           "Ensemble (XGB+LGBM)":   "#6C3483"}

fig, ax = plt.subplots(figsize=(16, 6))
sns.lineplot(data=ens_zoom_df, x="datetime", y="Power Demand (MW)",
             hue="Model", palette=ens_pal, linewidth=1.2, ax=ax)
for line in ax.get_lines():
    if line.get_label() == "Actual":
        line.set_linewidth(2.5)
        line.set_zorder(10)
    if line.get_label() == "Ensemble (XGB+LGBM)":
        line.set_linewidth(2.0)
ax.set_title(
    f"Ensemble vs Components — Last 2 Days\n"
    f"Weights → XGB: {w_xgb:.3f}  |  LGBM: {w_lgbm:.3f}",
    fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("Power Demand (MW)")
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("plot6_ensemble_zoom.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot6_ensemble_zoom.png")


# ── Plot 7: Ensemble Residual Distribution ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, mname in zip(axes, ["XGBoost (Tuned)", "LightGBM (Tuned)", "Ensemble (XGB+LGBM)"]):
    residuals = y_test.values - all_preds[mname]
    color     = ens_pal.get(mname, "#AAB7B8")
    sns.histplot(residuals, bins=80, kde=True, color=color,
                 edgecolor="white", linewidth=0.4, ax=ax)
    ax.axvline(0,                    color="#2C3E50", linewidth=1.2, linestyle="--")
    ax.axvline(np.mean(residuals),   color="gold",   linewidth=1.2, linestyle="--",
               label=f"Mean = {np.mean(residuals):.1f}")
    ax.set_title(f"{mname}\nMAE = {results_df.loc[mname, 'MAE']:.2f} MW",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Residual (MW)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
plt.suptitle("Residual Distribution — XGBoost (Tuned) vs LightGBM (Tuned) vs Ensemble",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot7_ensemble_residual_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot7_ensemble_residual_distribution.png")


# ── Plot 8: XGBoost Optuna Convergence ───────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(xgb_trial_nums, xgb_trial_maes,
           color="#95A5A6", s=18, alpha=0.6, label="Trial MAE")
ax.plot(xgb_trial_nums, xgb_running_min,
        color="#E67E22", linewidth=2, label="Best so far")
ax.set_xlabel("Trial number")
ax.set_ylabel("CV-MAE (MW)")
ax.set_title("Optuna Convergence — XGBoost", fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("plot8_optuna_xgb_convergence.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot8_optuna_xgb_convergence.png")


# ── Plot 9: LightGBM Optuna Convergence ──────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(lgbm_trial_nums, lgbm_trial_maes,
           color="#95A5A6", s=18, alpha=0.6, label="Trial MAE")
ax.plot(lgbm_trial_nums, lgbm_running_min,
        color="#1ABC9C", linewidth=2, label="Best so far")
ax.set_xlabel("Trial number")
ax.set_ylabel("CV-MAE (MW)")
ax.set_title("Optuna Convergence — LightGBM", fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("plot9_optuna_lgbm_convergence.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot9_optuna_lgbm_convergence.png")


# ── Plot 10: R² Comparison — horizontal bar ───────────────────
r2_df = results_df[["R²"]].sort_values("R²", ascending=True).reset_index()
bar_colors = [COLORS.get(m, "#AAB7B8") for m in r2_df["Model"]]

fig, ax = plt.subplots(figsize=(10, 8))
bars = sns.barplot(data=r2_df, y="Model", x="R²",
                   palette=bar_colors, edgecolor="white",
                   linewidth=0.6, ax=ax, orient="h")
for bar, val in zip(ax.patches, r2_df["R²"]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9, fontweight="bold")
ax.set_title("R² Score — All Models (higher = better)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("R² Score")
ax.set_ylabel("")
ax.axvline(1.0, color="#2C3E50", linewidth=0.8, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plot10_r2_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot10_r2_comparison.png")


# ── Plot 11: Best 3 models — Zoomed 1 day ────────────────────
ZOOM_1D   = 288
top3      = results_df.index[:3].tolist()
top3_rows = []
for t, a in zip(plot_dates[-ZOOM_1D:], plot_actual[-ZOOM_1D:]):
    top3_rows.append({"datetime": t, "Power Demand (MW)": a, "Model": "Actual"})
for mname in top3:
    for t, p in zip(plot_dates[-ZOOM_1D:], all_preds[mname][-ZOOM_1D:]):
        top3_rows.append({"datetime": t, "Power Demand (MW)": p, "Model": mname})

top3_df  = pd.DataFrame(top3_rows)
top3_df["datetime"] = pd.to_datetime(top3_df["datetime"])
top3_pal = {"Actual": "#2C3E50",
            top3[0]: COLORS.get(top3[0], "#E67E22"),
            top3[1]: COLORS.get(top3[1], "#C0392B"),
            top3[2]: COLORS.get(top3[2], "#148F77")}

fig, ax = plt.subplots(figsize=(14, 5))
sns.lineplot(data=top3_df, x="datetime", y="Power Demand (MW)",
             hue="Model", palette=top3_pal, linewidth=1.3, ax=ax)
for line in ax.get_lines():
    if line.get_label() == "Actual":
        line.set_linewidth(2.5)
        line.set_zorder(10)
ax.set_title(f"Top 3 Models vs Actual — Last 24 Hours\n"
             f"1st: {top3[0]}  |  2nd: {top3[1]}  |  3rd: {top3[2]}",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("Power Demand (MW)")
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("plot11_top3_models_1day.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot11_top3_models_1day.png")


# ── Plot 12: LSTM vs Ensemble vs Actual — Last 2 Days ────────
lstm_rows = []
for t, a in zip(plot_dates[-ZOOM:], plot_actual[-ZOOM:]):
    lstm_rows.append({"datetime": t, "Power Demand (MW)": a, "Model": "Actual"})
for mname in ["LSTM", "Ensemble (XGB+LGBM)", "XGBoost (Tuned)"]:
    for t, p in zip(plot_dates[-ZOOM:], all_preds[mname][-ZOOM:]):
        lstm_rows.append({"datetime": t, "Power Demand (MW)": p, "Model": mname})

lstm_zoom_df = pd.DataFrame(lstm_rows)
lstm_zoom_df["datetime"] = pd.to_datetime(lstm_zoom_df["datetime"])
lstm_pal = {
    "Actual":               "#2C3E50",
    "LSTM":                 "#D4145A",
    "Ensemble (XGB+LGBM)":  "#6C3483",
    "XGBoost (Tuned)":      "#C0392B",
}

fig, ax = plt.subplots(figsize=(16, 6))
sns.lineplot(data=lstm_zoom_df, x="datetime", y="Power Demand (MW)",
             hue="Model", palette=lstm_pal, linewidth=1.2, ax=ax)
for line in ax.get_lines():
    if line.get_label() == "Actual":
        line.set_linewidth(2.5)
        line.set_zorder(10)
    if line.get_label() == "LSTM":
        line.set_linewidth(2.0)
        line.set_zorder(9)
ax.set_title(
    f"LSTM vs Ensemble vs XGBoost (Tuned) — Last 2 Days\n"
    f"LSTM  MAE={results_df.loc['LSTM','MAE']:.2f}  |  "
    f"Ensemble MAE={results_df.loc['Ensemble (XGB+LGBM)','MAE']:.2f}  |  "
    f"XGB(T) MAE={results_df.loc['XGBoost (Tuned)','MAE']:.2f}",
    fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("Power Demand (MW)")
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("plot12_lstm_vs_ensemble_zoom.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot12_lstm_vs_ensemble_zoom.png")


# ── Plot 13: LSTM Training History (Loss Curves) ─────────────
fig, ax = plt.subplots(figsize=(10, 4))
epochs_ran = range(1, len(lstm_history.history["loss"]) + 1)
ax.plot(epochs_ran, lstm_history.history["loss"],
        color="#D4145A", linewidth=2, label="Train Loss (MSE)")
ax.plot(epochs_ran, lstm_history.history["val_loss"],
        color="#2C3E50", linewidth=2, linestyle="--", label="Val Loss (MSE)")
best_ep = int(np.argmin(lstm_history.history["val_loss"])) + 1
ax.axvline(best_ep, color="gold", linewidth=1.5, linestyle=":",
           label=f"Best epoch = {best_ep}")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss (scaled)")
ax.set_title("LSTM Training History — Train vs Validation Loss",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("plot13_lstm_training_history.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot13_lstm_training_history.png")


# ── Plot 14: Correlation Heatmap — Selected Features ─────────
# Show only the most interpretable numeric features (drop cyclical sin/cos)
HEATMAP_FEATURES = [
    "temp", "dwpt", "rhum", "wspd", "pres",
    "lag_12", "lag_288", "lag_2016",
    "roll_mean_12", "roll_std_12", "roll_max_12", "roll_min_12",
    "temp_hour", "temp_x_peak",
    "weekday", "weekend", "is_peak_hour", "is_day",
    "load",
]
heatmap_cols = [c for c in HEATMAP_FEATURES if c in df.columns]
corr_matrix  = df[heatmap_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)   # upper-triangle mask

fig, ax = plt.subplots(figsize=(16, 13))
sns.heatmap(
    corr_matrix,
    mask      = mask,
    annot     = True,
    fmt       = ".2f",
    annot_kws = {"size": 7},
    cmap      = "RdYlBu_r",
    center    = 0,
    vmin      = -1, vmax = 1,
    linewidths= 0.4,
    linecolor = "white",
    square    = True,
    cbar_kws  = {"shrink": 0.75, "label": "Pearson r"},
    ax        = ax,
)
ax.set_title("Feature Correlation Heatmap (Pearson r)",
             fontsize=14, fontweight="bold", pad=14)
ax.tick_params(axis="x", rotation=45, labelsize=9)
ax.tick_params(axis="y", rotation=0,  labelsize=9)
plt.tight_layout()
plt.savefig("plot14_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot14_correlation_heatmap.png")


# ============================================================
# SECTION 11: FINAL SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("  ELECTRICITY LOAD FORECASTING v12 — FINAL SUMMARY (Test: 2024)")
print("=" * 72)
print(f"  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE%':>8}")
print("  " + "-" * 62)
for mname, row in results_df.iterrows():
    marker = "  🏆" if mname == best_model else ""
    print(f"  {mname:<25} {row['MAE']:>8.2f} {row['RMSE']:>8.2f}"
          f" {row['R²']:>8.4f} {row['MAPE (%)']:>7.2f}%{marker}")
print("=" * 72)

print("""
📋 v13 CHANGELOG:
  ✅ Removed console print loops for feature importances
       • Calculation preserved; top features visible in plots
  ✅ Replaced giant residual time-series subplot grid (Plot 2)
       with two polished bivariate residual plots:
       • Plot 2a : Simple ML models (LR, DT, RF, KNN, GB)
       • Plot 2b : Advanced models (XGB-T, LGBM-T, Ensemble, LSTM)
       • Each panel: scatter + 2-D KDE contours + zero-line
       • Consistent axis limits for fair visual comparison
  ✅ Added correlation heatmap (Plot 14)
       • Pearson r for weather, lag, rolling, and target features
  ✅ Cleaner section headers and console output
  ✅ All 13 original plots retained and refined
       (filenames: plot1 … plot13 + plot2a/2b + plot14)
""")


# ============================================================
# SECTION 12: EXPORT FOR INTERACTIVE DASHBOARD
# ============================================================
import os
import json
import shutil

export_dir = "dashboard_data"
os.makedirs(export_dir, exist_ok=True)

print("\n--- Starting Dashboard Export ---")

# 1. Save Test Predictions
preds_df = pd.DataFrame({"datetime": dates_test})
preds_df["Actual"] = y_test.values
for name, preds in all_preds.items():
    preds_df[name] = preds
preds_df.to_csv(f"{export_dir}/test_predictions.csv", index=False)
print("✅ Exported: test_predictions.csv")

# 2. Save Metrics
results_df.reset_index().to_csv(f"{export_dir}/model_metrics.csv", index=False)
print("✅ Exported: model_metrics.csv")

# 3. Save Feature Importances
importances_df = pd.DataFrame({
    "Feature": FEATURES,
    "Random Forest": rf_imp.reindex(FEATURES).values,
    "XGBoost": xgb_imp.reindex(FEATURES).values,
    "LightGBM": lgbm_imp.reindex(FEATURES).values
})
importances_df.to_csv(f"{export_dir}/feature_importances.csv", index=False)
print("✅ Exported: feature_importances.csv")

# 4. Save Optuna Histories
optuna_xgb_df = pd.DataFrame({
    "Trial": xgb_trial_nums,
    "MAE": xgb_trial_maes,
    "Running_Min": xgb_running_min
})
optuna_xgb_df.to_csv(f"{export_dir}/optuna_xgb_history.csv", index=False)

optuna_lgbm_df = pd.DataFrame({
    "Trial": lgbm_trial_nums,
    "MAE": lgbm_trial_maes,
    "Running_Min": lgbm_running_min
})
optuna_lgbm_df.to_csv(f"{export_dir}/optuna_lgbm_history.csv", index=False)
print("✅ Exported: Optuna histories")

# 5. Save LSTM History
lstm_hist_df = pd.DataFrame({
    "Epoch": range(1, len(lstm_history.history["loss"]) + 1),
    "Loss": lstm_history.history["loss"],
    "Val_Loss": lstm_history.history["val_loss"]
})
lstm_hist_df.to_csv(f"{export_dir}/lstm_history.csv", index=False)
print("✅ Exported: lstm_history.csv")

# 6. Save Extra Configs (weights, best model)
configs = {
    "w_xgb": float(w_xgb),
    "w_lgbm": float(w_lgbm),
    "best_model": str(best_model)
}
with open(f"{export_dir}/dashboard_config.json", "w") as f:
    json.dump(configs, f, indent=4)
print("✅ Exported: dashboard_config.json")

# 6b. Save Model Artifacts for FastAPI inference
import joblib
joblib.dump(scaler, f"{export_dir}/scaler.pkl")
joblib.dump(rf_model, f"{export_dir}/rf_model.pkl")
joblib.dump(xgb_tuned, f"{export_dir}/xgb_model.pkl")
joblib.dump(lgbm_tuned, f"{export_dir}/lgbm_model.pkl")
print("✅ Exported: scaler.pkl, rf_model.pkl, xgb_model.pkl, lgbm_model.pkl")

# 7. Zip everything for easy download
shutil.make_archive("electricity_forecast_dashboard_data", 'zip', export_dir)
print("📦 Created ZIP: electricity_forecast_dashboard_data.zip")
print("-----------------------------------")
print("📥 DOWNLOAD 'electricity_forecast_dashboard_data.zip' from Kaggle output and extract it locally!")