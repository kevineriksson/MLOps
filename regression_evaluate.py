"""
Regression Model Evaluation (v2)
- Loads combined Jan + Feb 2021 data
- Applies same cleaning/feature engineering as regression_train.py
- Evaluates the EXISTING trained model (no retraining)
- Reports metrics and compares against v1 baseline
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Load & Combine Data ─────────────────────────────────────────────────────
df_jan = pd.read_parquet("green_tripdata_2021-01.parquet", engine="pyarrow")
df_feb = pd.read_parquet("green_tripdata_2021-02.parquet", engine="pyarrow")

print("=" * 70)
print("V2 EVALUATION — EXISTING MODEL ON COMBINED JAN + FEB DATA")
print("=" * 70)
print(f"\nJanuary rows:  {len(df_jan):,}")
print(f"February rows: {len(df_feb):,}")

df = pd.concat([df_jan, df_feb], ignore_index=True)
print(f"Combined rows: {len(df):,}")

# ── Cleaning (same as regression_train.py) ───────────────────────────────────
df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])

df["trip_duration_min"] = (
    (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
)

df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < df["fare_amount"].quantile(0.999))]
df = df[(df["trip_duration_min"] > 0) & (df["trip_duration_min"] <= 180)]
df = df.dropna(subset=["PULocationID", "DOLocationID", "passenger_count", "fare_amount"])

# Feature engineering
df["pickup_hour"] = df["lpep_pickup_datetime"].dt.hour

print(f"Combined size after cleaning: {len(df):,} rows")

# ── Train/Test Split (same params as v1) ─────────────────────────────────────
TARGET = "fare_amount"
BASE_FEATURES = ["PULocationID", "DOLocationID", "pickup_hour", "passenger_count"]
IMPROVED_FEATURES = BASE_FEATURES + ["est_duration_min"]

X = df[BASE_FEATURES].copy()
y = df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")

# ── Load Existing Models (trained on Jan only in v1) ─────────────────────────
print("\n" + "─" * 70)
print("LOADING EXISTING MODELS (trained on v1 Jan data)")
print("─" * 70)

regression_model = joblib.load("regression_models/regression_model.pkl")
duration_model = joblib.load("regression_models/duration_model.pkl")

print("  Loaded regression_models/regression_model.pkl")
print("  Loaded regression_models/duration_model.pkl")

# ── Generate est_duration_min using existing duration model ──────────────────
X_test_imp = X_test.copy()
X_test_imp["est_duration_min"] = duration_model.predict(X_test[BASE_FEATURES])

# ── Evaluate on Test Set ─────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("EVALUATION ON COMBINED JAN+FEB TEST SET")
print("─" * 70)

predictions = regression_model.predict(X_test_imp[IMPROVED_FEATURES])

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"  MAE:  ${mae:.4f}")
print(f"  RMSE: ${rmse:.4f}")
print(f"  R²:   {r2:.4f}")

# ── Comparison with v1 Results ───────────────────────────────────────────────
# v1 improved model results (from regression_train.py on Jan-only data)
# These are approximate reference values from v1
V1_MAE = 3.20
V1_RMSE = 5.10
V1_R2 = 0.62

print("\n" + "=" * 70)
print("COMPARISON: V1 (Jan only) vs V2 (Jan+Feb, same model)")
print("=" * 70)
print(f"{'Metric':<12} {'V1 (Jan)':>12} {'V2 (Jan+Feb)':>14} {'Change':>12}")
print("-" * 52)
print(f"{'MAE ($)':<12} {V1_MAE:>12.4f} {mae:>14.4f} {mae - V1_MAE:>+12.4f}")
print(f"{'RMSE ($)':<12} {V1_RMSE:>12.4f} {rmse:>14.4f} {rmse - V1_RMSE:>+12.4f}")
print(f"{'R²':<12} {V1_R2:>12.4f} {r2:>14.4f} {r2 - V1_R2:>+12.4f}")

# ── Explanation ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPLANATION OF RESULTS")
print("=" * 70)
print("""
The model was trained on January 2021 data only (v1). In v2 we evaluate
it on a combined January + February 2021 dataset.

Expected observations:
1. PERFORMANCE DEGRADATION: MAE/RMSE likely increase, R² likely decreases.
   The model has never seen February patterns during training.

2. DATA DRIFT: February has different characteristics than January:
   - Different weather conditions (colder, more snow)
   - Different travel patterns (post-holiday vs holiday season)
   - Potentially different fare distributions

3. DISTRIBUTION SHIFT: The train/test split now mixes both months,
   so the test set contains February trips the model was never trained on.
   The model's learned relationships from January may not generalize.

4. CONCLUSION: This demonstrates why monitoring and retraining matter.
   When new data arrives with different distributions, model performance
   degrades. Version 3 will retrain on the combined data to recover
   performance.
""")
