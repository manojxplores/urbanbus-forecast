import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import holidays
import optuna
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

DATA_BASE_DIR = 'urbanbus_data'
OUTPUT_FILENAME = 'lgbm_model_metrics.csv'
FORECAST_OUTPUT_DIR = 'forecasts_lgbm'
os.makedirs(FORECAST_OUTPUT_DIR, exist_ok=True)

def feature_engineering(df):
    # Datetime features
    df['Ride_start_datetime'] = pd.to_datetime(df['Ride_start_datetime'], errors='coerce')
    df['hour'] = df['Ride_start_datetime'].dt.hour
    df['minute'] = df['Ride_start_datetime'].dt.minute
    df['day'] = df['Ride_start_datetime'].dt.day
    df['dayofweek'] = df['Ride_start_datetime'].dt.dayofweek
    df['month'] = df['Ride_start_datetime'].dt.month
    df['year'] = df['Ride_start_datetime'].dt.year

    # Cyclic encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Weekend and holiday flag
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    china_holidays = holidays.country_holidays('CN')
    df['is_holiday'] = df['Ride_start_datetime'].dt.date.isin(china_holidays).astype(int)

    # Peak hour flag
    peak_hours = df.groupby('hour')['Passenger_Count'].sum().nlargest(2).index.tolist()
    df['is_peak_hour'] = df['hour'].isin(peak_hours).astype(int)

    df = df.sort_values('Ride_start_datetime').reset_index(drop=True)

    # Lag and window features
    for lag in [1, 2, 3, 4, 8, 12, 24]:
        df[f'lag_{lag}'] = df['Passenger_Count'].shift(lag)

    for window in [4, 8, 12, 24]:
        shifted_data = df['Passenger_Count'].shift(1)
        df[f'rolling_mean_{window}'] = shifted_data.rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = shifted_data.rolling(window=window, min_periods=1).std()

    lag_roll_cols = [col for col in df.columns if col.startswith(('lag_', 'rolling_'))]
    df = df.dropna(subset=lag_roll_cols).reset_index(drop=True)
    df = df.fillna(0)
    return df

# Helper Functions
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0: return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if np.sum(mask) == 0: return 0.0
    return np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100

def evaluate_model(y_true, y_pred, set_name="Set"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    print(f"\n{set_name} Performance:")
    print(f"  MAE:   {mae:.4f}")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  R²:    {r2:.4f}")
    print(f"  MAPE:  {mape:.2f}%")
    print(f"  sMAPE: {smape:.2f}%")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'sMAPE': smape}

def get_fit_status(train_r2, val_r2):
    r2_gap = train_r2 - val_r2
    if train_r2 < 0.5:
        return "Underfitting"
    elif r2_gap > 0.15:
        return "Overfitting"
    elif r2_gap < -0.05:
        return "Check Data (Unusual)"
    else:
        return "Good Fit"

# Objective function for optuna
def objective(trial, X_train, y_train, X_val, y_val):
    params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 10, 30),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8), 
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 10.0, log=True), 
            "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 20.0, log=True),
            "random_state": 42
        }
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LGBMRegressor(**params))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    return mean_absolute_error(y_val, preds)

def tune_model(X_train, y_train, X_val, y_val, n_trials=30):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
    best_params = study.best_params

    print(f"\n✅ Best params for lightgbm: {best_params}")
    return best_params


def train_with_optuna(X_train, y_train, X_val, y_val, best_params):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LGBMRegressor(**best_params, random_state=42))
    ])

    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred_val = pipe.predict(X_val)

    print("\nTraining Set:")
    train_metrics = evaluate_model(y_train, y_pred_train, "Train")
    print("\nValidation Set:")
    validation_metrics = evaluate_model(y_val, y_pred_val, "Validation")

    return train_metrics, validation_metrics, y_pred_val

all_results = []
processed_routes = set()

if os.path.exists(OUTPUT_FILENAME):
    print(f"Found existing results file: {OUTPUT_FILENAME}")
    try:
        results_df = pd.read_csv(OUTPUT_FILENAME)
        all_results = results_df.to_dict('records')
        processed_routes = set(results_df['route_name'])
        print(f"Loaded {len(processed_routes)} previously processed routes.")
    except Exception as e:
        print(f"Warning: Could not read results file. Starting from scratch. Error: {e}")
        all_results = []
        processed_routes = set()
else:
    print("No existing results file found. Starting a new run.")


N_TRIALS = 30

csv_files = [f for f in os.listdir(DATA_BASE_DIR) if f.endswith('.csv')]
print(f"Found {len(csv_files)} routes to process...")

for file in csv_files:
    if file in processed_routes:
        print(f"Skipping {file}: Already processed.")
        continue

    print(f"\n{'='*20} Processing Route: {file} {'='*20}")
    file_path = os.path.join(DATA_BASE_DIR, file)

    try:
        df = pd.read_csv(file_path)
        df = df.groupby(["Ride_start_datetime", "Bus_Service_Number", "Direction"], as_index=False)["Passenger_Count"].sum()
        df['Ride_start_datetime'] = pd.to_datetime(df['Ride_start_datetime'], errors='coerce')

        # Check if all dates are NaT after coercion
        if df['Ride_start_datetime'].isnull().all():
            print(f"Skipping {file}: All dates are invalid after coercion.")
            continue

        df = df.sort_values('Ride_start_datetime').reset_index(drop=True)
        df = feature_engineering(df)

        max_date = df['Ride_start_datetime'].max()
        cutoff_date = max_date - timedelta(days=28)
        train_df = df[df['Ride_start_datetime'] < cutoff_date].copy()
        val_df = df[df['Ride_start_datetime'] >= cutoff_date].copy()

        if len(train_df) == 0 or len(val_df) == 0:
            print(f"Skipping {file}: Not enough data for train/val split.")
            continue

        num_cols = [
            'hour', 'minute', 'day', 'dayofweek', 'month', 'year',
            'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'dow_sin', 'dow_cos',
            'month_sin', 'month_cos', 'is_weekend', 'is_holiday', 'is_peak_hour'
        ]
        lag_roll_cols = [col for col in train_df.columns if col.startswith(('lag_', 'rolling_'))]
        features = num_cols + lag_roll_cols

        X_train = train_df[features].copy()
        y_train = train_df['Passenger_Count'].copy()
        X_val = val_df[features].copy()
        y_val = val_df['Passenger_Count'].copy()

        # LightGBM
        best_params_lgbm = tune_model(X_train, y_train, X_val, y_val, n_trials=N_TRIALS)
        train_metrics_lgbm, val_metrics_lgbm, y_pred_val = train_with_optuna(X_train, y_train, X_val, y_val, best_params_lgbm)
        fit_status = get_fit_status(train_metrics_lgbm['R2'], val_metrics_lgbm['R2'])

        forecast_df = pd.DataFrame({
            'Ride_start_datetime': val_df['Ride_start_datetime'],
            'Actual_Passenger_Count': y_val,                 
            'Predicted_Passenger_Count': y_pred_val          
        })
        forecast_filename = f"forecast_{file}"
        forecast_save_path = os.path.join(FORECAST_OUTPUT_DIR, forecast_filename)
        forecast_df.to_csv(forecast_save_path, index=False)

        res_lgbm = {
                'route_name': file,
                'model': 'LightGBM',
                'Fit_Status': fit_status,
                'train_R2': train_metrics_lgbm['R2'],
                'val_R2': val_metrics_lgbm['R2'],
                'train_MAE': train_metrics_lgbm['MAE'],
                'val_MAE': val_metrics_lgbm['MAE'],
                'train_RMSE': train_metrics_lgbm['RMSE'],
                'val_RMSE': val_metrics_lgbm['RMSE'],
                'val_MAPE': val_metrics_lgbm['MAPE'],
                'val_sMAPE': val_metrics_lgbm['sMAPE']
            }
        all_results.append(res_lgbm)

    except Exception as e:
        print(f"FAILED to process {file}. Error: {e}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        cols = [
                'route_name', 'model', 'Fit_Status',
                'train_R2', 'val_R2', 'train_MAE', 'val_MAE',
                'train_RMSE', 'val_RMSE', 'val_MAPE', 'val_sMAPE'
            ]
        existing_cols = [c for c in cols if c in results_df.columns]
        results_df = results_df[existing_cols]

        results_df.to_csv(OUTPUT_FILENAME, index=False)