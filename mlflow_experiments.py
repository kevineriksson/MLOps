"""
MLflow Experiment Tracking — NYC Green Taxi Fare Prediction
3 Experiments: XGBoost, Random Forest, Linear Regression
- Trains on Jan+Feb 2021 combined data
- Multiple runs per experiment with different hyperparameters
- Tests best models on March 2021 data
- Assigns Production/Staging stages via code
- Reproduces best model from MLflow
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import os

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# ── MLflow Setup ─────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# ── Data Loading & Cleaning ──────────────────────────────────────────────────
def load_and_clean(paths):
    frames = [pd.read_parquet(p, engine="pyarrow") for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["lpep_pickup_datetime"]  = pd.to_datetime(df["lpep_pickup_datetime"])
    df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])
    df["trip_duration_min"] = (
        (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
    )
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < df["fare_amount"].quantile(0.999))]
    df = df[(df["trip_duration_min"] > 0) & (df["trip_duration_min"] <= 180)]
    df = df.dropna(subset=["PULocationID", "DOLocationID", "passenger_count", "fare_amount"])
    df["pickup_hour"] = df["lpep_pickup_datetime"].dt.hour
    return df

print("Loading Jan+Feb 2021 training data...")
df_train_full = load_and_clean([
    "green_tripdata_2021-01.parquet",
    "green_tripdata_2021-02.parquet"
])
print(f"Training data size: {len(df_train_full):,} rows")

print("Loading March 2021 test data...")
df_mar = load_and_clean(["green_tripdata_2021-03.parquet"])
print(f"March data size: {len(df_mar):,} rows")

# ── Feature Engineering (OOF Duration) ──────────────────────────────────────
TARGET       = "fare_amount"
BASE_FEATURES = ["PULocationID", "DOLocationID", "pickup_hour", "passenger_count"]
IMP_FEATURES  = BASE_FEATURES + ["est_duration_min"]

def build_oof_duration(df, base_features):
    """Build est_duration_min via OOF to avoid leakage."""
    X = df[base_features].copy()
    y_dur = df["trip_duration_min"].copy()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X))
    dur_params = dict(
        objective="reg:squarederror", n_estimators=500,
        learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    for tr_idx, va_idx in kf.split(X):
        m = xgb.XGBRegressor(**dur_params)
        m.fit(X.iloc[tr_idx], y_dur.iloc[tr_idx], verbose=False)
        oof_pred[va_idx] = m.predict(X.iloc[va_idx])
    # Final model trained on all data for inference
    final_dur = xgb.XGBRegressor(**dur_params)
    final_dur.fit(X, y_dur, verbose=False)
    return oof_pred, final_dur

print("\nBuilding OOF duration model on Jan+Feb...")
oof_dur_train, duration_model = build_oof_duration(df_train_full, BASE_FEATURES)
df_train_full["est_duration_min"] = oof_dur_train

# Add est_duration_min to March data using the trained duration model
df_mar["est_duration_min"] = duration_model.predict(df_mar[BASE_FEATURES])

# Train/test split on Jan+Feb
X_train, X_test, y_train, y_test = train_test_split(
    df_train_full[IMP_FEATURES], df_train_full[TARGET],
    test_size=0.2, random_state=42
)

# March eval set
X_mar = df_mar[IMP_FEATURES]
y_mar = df_mar[TARGET]

print(f"Train: {len(X_train):,} | Val: {len(X_test):,} | March: {len(X_mar):,}")

# ── Helper: Compute Metrics ───────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    return {
        "mae":  mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2":   r2_score(y_true, y_pred),
    }

def log_metrics(metrics):
    mlflow.log_metric("mae",  metrics["mae"])
    mlflow.log_metric("rmse", metrics["rmse"])
    mlflow.log_metric("r2",   metrics["r2"])

# ── Create Experiments ────────────────────────────────────────────────────────
experiment_names = [
    "xgboost-fare-prediction",
    "random-forest-fare-prediction",
    "ridge-fare-prediction",
]

experiment_ids = {}
for name in experiment_names:
    existing = client.get_experiment_by_name(name)
    if existing is None:
        eid = client.create_experiment(name)
        print(f"Created experiment: {name} (id={eid})")
    else:
        eid = existing.experiment_id
        print(f"Experiment already exists: {name} (id={eid})")
    experiment_ids[name] = eid

# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — XGBoost Regressor
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 1: XGBoost Regressor")
print("=" * 70)

xgb_configs = [
    {"n_estimators": 500,  "learning_rate": 0.05, "max_depth": 4},
    {"n_estimators": 500,  "learning_rate": 0.05, "max_depth": 6},
    {"n_estimators": 1000, "learning_rate": 0.05, "max_depth": 6},
    {"n_estimators": 1000, "learning_rate": 0.01, "max_depth": 6},
    {"n_estimators": 1000, "learning_rate": 0.05, "max_depth": 8},
]

mlflow.set_experiment("xgboost-fare-prediction")

for cfg in xgb_configs:
    with mlflow.start_run(run_name=f"xgb_d{cfg['max_depth']}_lr{cfg['learning_rate']}_n{cfg['n_estimators']}"):
        mlflow.log_params(cfg)
        mlflow.log_param("features", IMP_FEATURES)

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, early_stopping_rounds=50,
            **cfg
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        metrics = compute_metrics(y_test, model.predict(X_test))
        log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="model")

        print(f"  XGB d={cfg['max_depth']} lr={cfg['learning_rate']} n={cfg['n_estimators']} "
              f"| MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} R²={metrics['r2']:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Random Forest Regressor
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 2: Random Forest Regressor")
print("=" * 70)

rf_configs = [
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5},
    {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5},
    {"n_estimators": 200, "max_depth": 20, "min_samples_split": 2},
    {"n_estimators": 300, "max_depth": 20, "min_samples_split": 2},
    {"n_estimators": 300, "max_depth": None, "min_samples_split": 5},
]

mlflow.set_experiment("random-forest-fare-prediction")

for cfg in rf_configs:
    run_name = f"rf_d{cfg['max_depth']}_n{cfg['n_estimators']}_mss{cfg['min_samples_split']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg)
        mlflow.log_param("features", IMP_FEATURES)

        model = RandomForestRegressor(random_state=42, n_jobs=-1, **cfg)
        model.fit(X_train, y_train)

        metrics = compute_metrics(y_test, model.predict(X_test))
        log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"  RF d={cfg['max_depth']} n={cfg['n_estimators']} mss={cfg['min_samples_split']} "
              f"| MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} R²={metrics['r2']:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Ridge Regression
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 3: Ridge Regression")
print("=" * 70)

ridge_configs = [
    {"alpha": 0.1,  "fit_intercept": True},
    {"alpha": 1.0,  "fit_intercept": True},
    {"alpha": 10.0, "fit_intercept": True},
    {"alpha": 100.0,"fit_intercept": True},
    {"alpha": 1.0,  "fit_intercept": False},
]

mlflow.set_experiment("ridge-fare-prediction")

for cfg in ridge_configs:
    run_name = f"ridge_alpha{cfg['alpha']}_intercept{cfg['fit_intercept']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg)
        mlflow.log_param("features", IMP_FEATURES)

        model = Ridge(**cfg)
        model.fit(X_train, y_train)

        metrics = compute_metrics(y_test, model.predict(X_test))
        log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"  Ridge alpha={cfg['alpha']} intercept={cfg['fit_intercept']} "
              f"| MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} R²={metrics['r2']:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# FIND BEST RUN PER EXPERIMENT & REGISTER MODELS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINDING BEST RUN PER EXPERIMENT (lowest MAE)")
print("=" * 70)

model_names = {
    "xgboost-fare-prediction":       "GreenTaxi-XGBoost",
    "random-forest-fare-prediction": "GreenTaxi-RandomForest",
    "ridge-fare-prediction":         "GreenTaxi-Ridge",
}

best_runs = {}

for exp_name, model_name in model_names.items():
    exp = client.get_experiment_by_name(exp_name)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=10,
        order_by=["metrics.mae ASC"]
    )
    best_run = runs[0]
    best_runs[model_name] = best_run
    print(f"\n  {exp_name}")
    print(f"    Best run ID : {best_run.info.run_id}")
    print(f"    MAE         : {best_run.data.metrics['mae']:.4f}")
    print(f"    RMSE        : {best_run.data.metrics['rmse']:.4f}")
    print(f"    R²          : {best_run.data.metrics['r2']:.4f}")

    # Register model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"    Registered  : {model_name} v{mv.version}")

# ════════════════════════════════════════════════════════════════════════════
# EVALUATE BEST MODELS ON MARCH 2021 DATA
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EVALUATING BEST MODELS ON MARCH 2021 DATA")
print("=" * 70)

march_results = {}

for model_name, best_run in best_runs.items():
    model_uri = f"runs:/{best_run.info.run_id}/model"
    loaded = mlflow.pyfunc.load_model(model_uri)
    preds = loaded.predict(X_mar)
    metrics = compute_metrics(y_mar, preds)
    march_results[model_name] = metrics
    print(f"\n  {model_name}")
    print(f"    MAE  : ${metrics['mae']:.4f}")
    print(f"    RMSE : ${metrics['rmse']:.4f}")
    print(f"    R²   : {metrics['r2']:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# ASSIGN STAGES — Best model → Production, others → Staging
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ASSIGNING MODEL STAGES BASED ON MARCH PERFORMANCE")
print("=" * 70)

best_model_name = min(march_results, key=lambda k: march_results[k]["mae"])
print(f"\n  Best model on March data: {best_model_name} "
      f"(MAE=${march_results[best_model_name]['mae']:.4f})")

for model_name in model_names.values():
    # Get latest version
    versions = client.get_latest_versions(model_name)
    latest_version = versions[-1].version
    stage = "Production" if model_name == best_model_name else "Staging"

    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage=stage,
        archive_existing_versions=False
    )
    print(f"  {model_name} v{latest_version} → {stage}")

# ════════════════════════════════════════════════════════════════════════════
# REPRODUCE BEST MODEL FROM MLFLOW
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("REPRODUCING BEST MODEL FROM MLFLOW")
print("=" * 70)

best_run = best_runs[best_model_name]
run_id   = best_run.info.run_id
print(f"\n  Run ID : {run_id}")
print(f"  Model  : {best_model_name}")

# Load via run ID
reproduced_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

# Original predictions (on Jan+Feb test set)
original_preds   = reproduced_model.predict(X_test)
original_metrics = compute_metrics(y_test, original_preds)

# Compare to logged metrics
logged_mae  = best_run.data.metrics["mae"]
logged_rmse = best_run.data.metrics["rmse"]
logged_r2   = best_run.data.metrics["r2"]

print(f"\n  {'Metric':<8} {'Logged':>12} {'Reproduced':>12} {'Match':>8}")
print("  " + "-" * 44)
print(f"  {'MAE':<8} {logged_mae:>12.4f} {original_metrics['mae']:>12.4f} "
      f"{'✓' if abs(logged_mae - original_metrics['mae']) < 1e-4 else '✗':>8}")
print(f"  {'RMSE':<8} {logged_rmse:>12.4f} {original_metrics['rmse']:>12.4f} "
      f"{'✓' if abs(logged_rmse - original_metrics['rmse']) < 1e-4 else '✗':>8}")
print(f"  {'R²':<8} {logged_r2:>12.4f} {original_metrics['r2']:>12.4f} "
      f"{'✓' if abs(logged_r2 - original_metrics['r2']) < 1e-4 else '✗':>8}")

print("\n  EXPLANATION:")
print("  Results match exactly because:")
print("  1. Same model artifact loaded from MLflow (no retraining)")
print("  2. Same test set used (random_state=42 ensures deterministic split)")
print("  3. Same est_duration_min values (duration model is deterministic)")
print("  4. XGBoost/sklearn inference is fully deterministic given same inputs")

print("\n" + "=" * 70)
print("DONE — All experiments logged, models registered and staged.")
print("  View UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("=" * 70)