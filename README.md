# Short-Term Electricity Load Forecasting

This project implements a complete short-term electricity load forecasting pipeline for high-resolution power demand data. It compares classical machine learning models, boosting models, an inverse-RMSE weighted ensemble, and an LSTM network using 5-minute electricity demand and weather observations from Delhi, India.

The work is based on the research paper **"Comparative Analysis of Machine Learning and Deep Learning Models for Short Term Electricity Load Forecasting Using Weather and Temporal Features"** by Agrim Jain and Satya Narayan Agarwal.

## Overview

Short-term load forecasting helps utilities and grid operators estimate near-future electricity demand for generation scheduling, demand response, cost reduction, and grid stability. This project studies how well different ML and DL models forecast electricity load when given temporal, weather, lag, rolling-window, and interaction features.

The main finding from the research workflow is that strong feature engineering can make simpler machine learning models highly competitive. In the reported experiments, Linear Regression achieved the best overall test performance, while Random Forest, tuned LightGBM, and the XGBoost-LightGBM ensemble also produced strong results. The LSTM model underperformed the best ML models on this structured dataset.

## Key Features

- Loads and cleans 5-minute electricity demand data from 2021 to 2024.
- Uses weather variables such as temperature, dew point, relative humidity, wind speed, pressure, and wind direction.
- Builds temporal features using cyclical hour/month encodings, weekday/weekend flags, day/peak-hour flags, and feature interactions.
- Adds time-series dependency features including short-term lag, daily lag, weekly lag, and rolling statistics.
- Trains and compares:
  - Naive baseline
  - Linear Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - Tuned XGBoost with Optuna
  - Tuned LightGBM with Optuna
  - Weighted ensemble of tuned XGBoost and tuned LightGBM
  - LSTM
- Evaluates models using MAE, RMSE, MAPE, and R2.
- Generates visualizations for predictions, residuals, feature importance, Optuna convergence, model comparison, and LSTM training behavior.

## Repository Structure

```text
.
+-- datasets/
|   +-- powerdemand_5min_2021_to_2024_with weather.csv
+-- main project/
|   +-- electricity_load_forecastingV12.py
|   +-- electricity_load_forecasting_v12/
|       +-- config.py
|       +-- data.py
|       +-- ensemble.py
|       +-- environment.py
|       +-- lstm.py
|       +-- metrics.py
|       +-- models.py
|       +-- pipeline.py
|       +-- summary.py
|       +-- tuning.py
|       +-- visualization.py
+-- outputs/
|   +-- generated research plots
+-- project versions/
|   +-- previous script versions
+-- LICENSE
+-- README.md
```

## Dataset

The dataset used by the project is:

```text
datasets/powerdemand_5min_2021_to_2024_with weather.csv
```

Important columns include:

| Column | Description |
| --- | --- |
| `datetime` | Timestamp of each 5-minute observation |
| `Power demand` | Electricity load target |
| `temp` | Temperature |
| `dwpt` | Dew point temperature |
| `rhum` | Relative humidity |
| `wdir` | Wind direction |
| `wspd` | Wind speed |
| `pres` | Atmospheric pressure |
| `year`, `month`, `day`, `hour`, `minute` | Raw temporal fields |

The pipeline renames `Power demand` to `load`, sorts the time series chronologically, removes unused columns, and preserves temporal order for training and testing.

## Feature Engineering

The model feature set includes:

- Cyclical encodings: `hour_sin`, `hour_cos`, `month_sin`, `month_cos`
- Calendar indicators: `weekday`, `weekend`, `is_peak_hour`, `is_day`
- Weather variables: `temp`, `dwpt`, `rhum`, `wspd`, `pres`
- Wind direction encodings: `wdir_sin`, `wdir_cos`
- Interaction features: `temp_hour`, `temp_x_peak`
- Lag features: `lag_12`, `lag_288`, `lag_2016`
- Rolling features: `roll_mean_12`, `roll_std_12`, `roll_max_12`, `roll_min_12`

For 5-minute data, `lag_12` represents roughly 1 hour, `lag_288` represents the same time on the previous day, and `lag_2016` represents the same time in the previous week.

## Train-Test Split

The project uses a chronological split:

| Split | Date Range |
| --- | --- |
| Train | 2021-01-01 to 2023-12-31 |
| Test | 2024-01-01 onward |

This avoids leakage from future observations into the training set.

## Results Reported in the Research Paper

The paper reports the following test-set performance for 2024:

| Model | MAE | RMSE | MAPE (%) | R2 |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 37.11 | 87.70 | 0.87 | 0.9960 |
| Random Forest | 43.34 | 107.97 | 0.95 | 0.9940 |
| LightGBM (Tuned) | 53.08 | 123.07 | 1.10 | 0.9910 |
| Ensemble (XGB+LGBM) | 55.52 | 144.87 | 1.12 | 0.9890 |
| Decision Tree | 56.91 | 123.02 | 1.21 | 0.9920 |
| Gradient Boosting | 57.68 | 131.43 | 1.27 | 0.9920 |
| LightGBM | 57.77 | 131.81 | 1.22 | 0.9910 |
| XGBoost | 59.69 | 157.47 | 1.22 | 0.9870 |
| XGBoost (Tuned) | 61.18 | 173.56 | 1.28 | 0.9850 |
| LSTM | 210.41 | 269.61 | 4.31 | 0.9630 |
| KNN | 274.47 | 372.22 | 6.39 | 0.9290 |
| Naive Baseline | 598.43 | 903.01 | 15.19 | 0.5830 |

The strongest result came from Linear Regression, showing that well-designed lag, rolling, and temporal features captured much of the structure in the electricity demand series.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm optuna tensorflow
```

TensorFlow installation can vary by Python version and hardware. If the default install fails, install a TensorFlow build that matches your local environment.

## How to Run

From the repository root, run:

```bash
python "main project/electricity_load_forecastingV12.py"
```

The default configuration first checks the Kaggle dataset path defined in `config.py`. If that path is not available, it automatically falls back to the local CSV file in `datasets/`.

## Outputs

The `outputs/` folder contains generated figures from the research workflow, including:

- Actual vs predicted demand plots
- Model comparison plots
- Feature importance plots
- Residual analysis plots
- Optuna convergence plots
- LSTM training history
- Correlation heatmap

The modular V12 pipeline also saves plots during execution in the current working directory.

## Main Modules

| File | Purpose |
| --- | --- |
| `config.py` | Central configuration for paths, features, split date, plotting constants, and model settings |
| `data.py` | Dataset loading, cleaning, feature engineering, chronological split, and scaling |
| `models.py` | Default model definitions, training, time-series CV, and feature importance extraction |
| `tuning.py` | Optuna tuning for XGBoost and LightGBM |
| `ensemble.py` | Inverse-RMSE weighted ensemble of tuned XGBoost and tuned LightGBM |
| `lstm.py` | Sequence preparation and LSTM model training/evaluation |
| `visualization.py` | Plot generation |
| `metrics.py` | MAE, RMSE, MAPE, and R2 helpers |
| `pipeline.py` | Full orchestration of the forecasting workflow |
| `summary.py` | Final results table and workflow summary |

## Research Takeaways

- Lag and rolling statistical features are the most influential predictors.
- Temporal autocorrelation is very strong in the 5-minute electricity demand series.
- Tuned boosting models and ensembles provide stable forecasts with low residual bias.
- LSTM did not outperform engineered ML models on this structured dataset and showed signs of overfitting.
- Model explainability is important for practical smart-grid and utility forecasting use cases.

## Future Work

Possible extensions include:

- Adding holiday, festival, industrial activity, and economic indicators.
- Testing the pipeline across multiple cities or climate zones.
- Adding probabilistic forecasting and uncertainty intervals.
- Trying Transformer, Temporal Fusion Transformer, CNN-LSTM, or attention-based architectures.
- Adding explainable AI methods such as SHAP for model transparency.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
