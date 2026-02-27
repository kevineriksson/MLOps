"""
EDA for NYC Green Taxi Fare Prediction (Regression)
Produces regression_plots in regression_plots/ directory and prints conclusions to stdout.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Load & Clean ─────────────────────────────────────────────────────────────
df = pd.read_parquet("green_tripdata_2021-01.parquet", engine="pyarrow")

df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])

df["trip_duration_min"] = (
    (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
)

# Filtering
df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < df["fare_amount"].quantile(0.999))]
df = df[(df["trip_duration_min"] > 0) & (df["trip_duration_min"] <= 180)]

# Feature engineering
df["pickup_hour"] = df["lpep_pickup_datetime"].dt.hour

# OOF est_duration_min (simple proxy for EDA: use actual duration as stand-in for scatter)
# In regression_train.py this is computed properly via OOF to avoid leakage
from sklearn.model_selection import KFold
import xgboost as xgb

base_features = ["PULocationID", "DOLocationID", "pickup_hour", "passenger_count"]
X_dur = df[base_features].copy()
y_dur = df["trip_duration_min"].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(df))
for train_idx, val_idx in kf.split(X_dur):
    m = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    m.fit(X_dur.iloc[train_idx], y_dur.iloc[train_idx], verbose=False)
    oof_pred[val_idx] = m.predict(X_dur.iloc[val_idx])

df["est_duration_min"] = oof_pred

os.makedirs("regression_plots", exist_ok=True)

# ── Plot 1: fare_amount histogram + log-histogram ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["fare_amount"], bins=80, edgecolor="black", alpha=0.7)
axes[0].set_title("Fare Amount Distribution")
axes[0].set_xlabel("Fare ($)")
axes[0].set_ylabel("Count")

axes[1].hist(np.log1p(df["fare_amount"]), bins=80, edgecolor="black", alpha=0.7, color="orange")
axes[1].set_title("Log(1 + Fare Amount) Distribution")
axes[1].set_xlabel("log(1 + Fare)")
axes[1].set_ylabel("Count")
plt.tight_layout()
plt.savefig("regression_plots/fare_histogram.png", dpi=150)
plt.close()

# ── Plot 2: trip_duration_min histogram ──────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.hist(df["trip_duration_min"], bins=80, edgecolor="black", alpha=0.7, color="green")
plt.title("Trip Duration Distribution")
plt.xlabel("Duration (min)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("regression_plots/trip_duration_histogram.png", dpi=150)
plt.close()

# ── Plot 3: % missing per column ────────────────────────────────────────────
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]

plt.figure(figsize=(10, 5))
missing_pct.plot(kind="bar", color="salmon", edgecolor="black")
plt.title("Percentage of Missing Values per Column")
plt.ylabel("% Missing")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("regression_plots/missing_values.png", dpi=150)
plt.close()

# ── Plot 4: Boxplots of fare_amount and trip_duration_min ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].boxplot(df["fare_amount"].dropna(), vert=True)
axes[0].set_title("Fare Amount Boxplot")
axes[0].set_ylabel("Fare ($)")

axes[1].boxplot(df["trip_duration_min"].dropna(), vert=True)
axes[1].set_title("Trip Duration Boxplot")
axes[1].set_ylabel("Duration (min)")
plt.tight_layout()
plt.savefig("regression_plots/boxplots.png", dpi=150)
plt.close()

# ── Plot 5: Scatter fare_amount vs trip_distance (post-trip context) ─────────
sample = df.sample(n=min(5000, len(df)), random_state=42)
plt.figure(figsize=(8, 5))
plt.scatter(sample["trip_distance"], sample["fare_amount"], alpha=0.3, s=10)
plt.title("Fare Amount vs Trip Distance (post-trip context)")
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Fare ($)")
plt.tight_layout()
plt.savefig("regression_plots/fare_vs_distance.png", dpi=150)
plt.close()

# ── Plot 6: Scatter fare_amount vs est_duration_min ──────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(sample["est_duration_min"], sample["fare_amount"], alpha=0.3, s=10, color="purple")
plt.title("Fare Amount vs Estimated Duration")
plt.xlabel("Estimated Duration (min)")
plt.ylabel("Fare ($)")
plt.tight_layout()
plt.savefig("regression_plots/fare_vs_est_duration.png", dpi=150)
plt.close()

# ── Plot 7: Avg fare by pickup_hour and by RatecodeID ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

avg_by_hour = df.groupby("pickup_hour")["fare_amount"].mean()
axes[0].bar(avg_by_hour.index, avg_by_hour.values, color="steelblue", edgecolor="black")
axes[0].set_title("Average Fare by Pickup Hour")
axes[0].set_xlabel("Pickup Hour")
axes[0].set_ylabel("Avg Fare ($)")

avg_by_rate = df.groupby("RatecodeID")["fare_amount"].mean().sort_values(ascending=False)
axes[1].bar(avg_by_rate.index.astype(str), avg_by_rate.values, color="coral", edgecolor="black")
axes[1].set_title("Average Fare by RatecodeID")
axes[1].set_xlabel("RatecodeID")
axes[1].set_ylabel("Avg Fare ($)")
plt.tight_layout()
plt.savefig("regression_plots/avg_fare_by_hour_and_ratecode.png", dpi=150)
plt.close()
