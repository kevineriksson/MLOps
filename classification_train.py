"""
Classification Model Training for NYC Green Taxi Fare Prediction
- Target: is_high_fare (fare_amount > 75th percentile)
- Baseline: 4 booking-time features
- Improved: +1 feature (est_duration_min via OOF duration model)
- Validates improvement with cross-validation
- Saves classification_models to classification_models/ directory
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ── Data Loading & Cleaning ──────────────────────────────────────────────────
df = pd.read_parquet("green_tripdata_2021-01.parquet", engine="pyarrow")

df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])

df["trip_duration_min"] = (
    (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
)

# Cleaning filters
df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < df["fare_amount"].quantile(0.999))]
df = df[(df["trip_duration_min"] > 0) & (df["trip_duration_min"] <= 180)]
df = df.dropna(subset=["PULocationID", "DOLocationID", "passenger_count", "fare_amount"])

# Feature engineering
df["pickup_hour"] = df["lpep_pickup_datetime"].dt.hour

# Define features
BASE_FEATURES = ["PULocationID", "DOLocationID", "pickup_hour", "passenger_count"]

# ── Train/Test Split (before computing threshold to avoid leakage) ───────────
X = df[BASE_FEATURES].copy()
y_fare = df["fare_amount"].copy()

X_train, X_test, y_fare_train, y_fare_test = train_test_split(
    X, y_fare, test_size=0.2, random_state=42
)
train_idx = X_train.index
test_idx = X_test.index

# Compute threshold on TRAINING set only to avoid leakage
fare_threshold = y_fare_train.quantile(0.75)
y_train = (y_fare_train > fare_threshold).astype(int)
y_test = (y_fare_test > fare_threshold).astype(int)

print("=" * 70)
print("NYC GREEN TAXI FARE PREDICTION — CLASSIFICATION")
print("=" * 70)
print(f"\nDataset size after cleaning: {len(df):,} rows")
print(f"Target: is_high_fare (fare_amount > ${fare_threshold:.2f}, 75th pctl of train)")
print(f"Baseline features (4): {BASE_FEATURES}")
print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
print(f"Train class balance: {y_train.mean():.1%} positive (high fare)")
print(f"Test class balance:  {y_test.mean():.1%} positive (high fare)")

# ── Baseline Model (4 features) ─────────────────────────────────────────────
print("\n" + "-" * 70)
print("BASELINE MODEL (4 features)")
print("-" * 70)

baseline_model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50,
)

baseline_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
baseline_pred = (baseline_proba > 0.5).astype(int)

baseline_auc = roc_auc_score(y_test, baseline_proba)
baseline_precision = precision_score(y_test, baseline_pred)
baseline_recall = recall_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred)
baseline_cm = confusion_matrix(y_test, baseline_pred)

print(f"  ROC AUC:   {baseline_auc:.4f}")
print(f"  Precision: {baseline_precision:.4f}")
print(f"  Recall:    {baseline_recall:.4f}")
print(f"  F1 Score:  {baseline_f1:.4f}")
print(f"\n  Confusion Matrix:")
print(f"                  Predicted 0   Predicted 1")
print(f"  Actual 0        {baseline_cm[0,0]:>8,}      {baseline_cm[0,1]:>8,}")
print(f"  Actual 1        {baseline_cm[1,0]:>8,}      {baseline_cm[1,1]:>8,}")

# ── OOF Duration Model ──────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("BUILDING OOF DURATION MODEL (est_duration_min)")
print("-" * 70)

X_dur_train = df.loc[train_idx, BASE_FEATURES].copy()
y_dur_train = df.loc[train_idx, "trip_duration_min"].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_duration_pred = np.zeros(len(X_dur_train))

dur_params = dict(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_dur_train), 1):
    X_tr = X_dur_train.iloc[tr_idx]
    y_tr = y_dur_train.iloc[tr_idx]
    X_va = X_dur_train.iloc[va_idx]
    y_va = y_dur_train.iloc[va_idx]

    dur_model = xgb.XGBRegressor(**dur_params, early_stopping_rounds=30)
    dur_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_duration_pred[va_idx] = dur_model.predict(X_va)

from sklearn.metrics import mean_absolute_error
oof_dur_mae = mean_absolute_error(y_dur_train, oof_duration_pred)
print(f"OOF Duration MAE: {oof_dur_mae:.2f} min")

# Final duration model on all training data
duration_model_final = xgb.XGBRegressor(**dur_params)
duration_model_final.fit(X_dur_train, y_dur_train, verbose=False)

train_est_duration = oof_duration_pred
test_est_duration = duration_model_final.predict(df.loc[test_idx, BASE_FEATURES])

print(f"  LEAKAGE CHECK: est_duration_min is computed from OOF predictions on")
print(f"  training folds only. Test set uses the final duration model trained")
print(f"  on all training data. No actual trip_duration_min is used as a feature.")
print(f"  Threshold (${fare_threshold:.2f}) computed on training set only.")

# ── Improved Model (5 features) ─────────────────────────────────────────────
print("\n" + "-" * 70)
print("IMPROVED MODEL (5 features: +est_duration_min)")
print("-" * 70)

X_train_imp = X_train.copy()
X_train_imp["est_duration_min"] = train_est_duration

X_test_imp = X_test.copy()
X_test_imp["est_duration_min"] = test_est_duration

improved_model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50,
)

improved_model.fit(
    X_train_imp, y_train,
    eval_set=[(X_test_imp, y_test)],
    verbose=False,
)

improved_proba = improved_model.predict_proba(X_test_imp)[:, 1]
improved_pred = (improved_proba > 0.5).astype(int)

improved_auc = roc_auc_score(y_test, improved_proba)
improved_precision = precision_score(y_test, improved_pred)
improved_recall = recall_score(y_test, improved_pred)
improved_f1 = f1_score(y_test, improved_pred)
improved_cm = confusion_matrix(y_test, improved_pred)

print(f"  ROC AUC:   {improved_auc:.4f}")
print(f"  Precision: {improved_precision:.4f}")
print(f"  Recall:    {improved_recall:.4f}")
print(f"  F1 Score:  {improved_f1:.4f}")
print(f"\n  Confusion Matrix:")
print(f"                  Predicted 0   Predicted 1")
print(f"  Actual 0        {improved_cm[0,0]:>8,}      {improved_cm[0,1]:>8,}")
print(f"  Actual 1        {improved_cm[1,0]:>8,}      {improved_cm[1,1]:>8,}")

# ── Comparison Table ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPARISON: BASELINE vs IMPROVED")
print("=" * 70)
print(f"{'Metric':<12} {'Baseline':>12} {'Improved':>12} {'Change':>12}")
print("-" * 50)
print(f"{'ROC AUC':<12} {baseline_auc:>12.4f} {improved_auc:>12.4f} {improved_auc - baseline_auc:>+12.4f}")
print(f"{'Precision':<12} {baseline_precision:>12.4f} {improved_precision:>12.4f} {improved_precision - baseline_precision:>+12.4f}")
print(f"{'Recall':<12} {baseline_recall:>12.4f} {improved_recall:>12.4f} {improved_recall - baseline_recall:>+12.4f}")
print(f"{'F1 Score':<12} {baseline_f1:>12.4f} {improved_f1:>12.4f} {improved_f1 - baseline_f1:>+12.4f}")

# ── Cross-Validation ─────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("K-FOLD CROSS-VALIDATION (5 folds)")
print("-" * 70)

full_X = df[BASE_FEATURES].copy()
full_y_fare = df["fare_amount"].copy()
full_dur_y = df["trip_duration_min"].copy()

cv_baseline_aucs = []
cv_improved_aucs = []

kf_cv = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, va_idx) in enumerate(kf_cv.split(full_X), 1):
    X_tr = full_X.iloc[tr_idx]
    X_va = full_X.iloc[va_idx]

    # Compute threshold on this fold's training data
    fold_threshold = full_y_fare.iloc[tr_idx].quantile(0.75)
    y_tr = (full_y_fare.iloc[tr_idx] > fold_threshold).astype(int)
    y_va = (full_y_fare.iloc[va_idx] > fold_threshold).astype(int)

    # Baseline
    m_base = xgb.XGBClassifier(
        objective="binary:logistic", n_estimators=500,
        learning_rate=0.05, max_depth=6, random_state=42
    )
    m_base.fit(X_tr, y_tr, verbose=False)
    pred_base_proba = m_base.predict_proba(X_va)[:, 1]
    cv_baseline_aucs.append(roc_auc_score(y_va, pred_base_proba))

    # Duration model (inner OOF)
    inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_oof = np.zeros(len(X_tr))
    dur_y_tr = full_dur_y.iloc[tr_idx]

    for inner_tr, inner_va in inner_kf.split(X_tr):
        m_dur = xgb.XGBRegressor(**dur_params)
        m_dur.fit(X_tr.iloc[inner_tr], dur_y_tr.iloc[inner_tr], verbose=False)
        inner_oof[inner_va] = m_dur.predict(X_tr.iloc[inner_va])

    m_dur_final = xgb.XGBRegressor(**dur_params)
    m_dur_final.fit(X_tr, dur_y_tr, verbose=False)

    X_tr_imp = X_tr.copy()
    X_tr_imp["est_duration_min"] = inner_oof
    X_va_imp = X_va.copy()
    X_va_imp["est_duration_min"] = m_dur_final.predict(X_va)

    # Improved
    m_imp = xgb.XGBClassifier(
        objective="binary:logistic", n_estimators=500,
        learning_rate=0.05, max_depth=6, random_state=42
    )
    m_imp.fit(X_tr_imp, y_tr, verbose=False)
    pred_imp_proba = m_imp.predict_proba(X_va_imp)[:, 1]
    cv_improved_aucs.append(roc_auc_score(y_va, pred_imp_proba))

    print(f"  Fold {fold}: Baseline AUC={cv_baseline_aucs[-1]:.4f}  Improved AUC={cv_improved_aucs[-1]:.4f}")

print(f"\n  Baseline — AUC: {np.mean(cv_baseline_aucs):.4f} +/- {np.std(cv_baseline_aucs):.4f}")
print(f"  Improved — AUC: {np.mean(cv_improved_aucs):.4f} +/- {np.std(cv_improved_aucs):.4f}")

# ── Save Models ──────────────────────────────────────────────────────────────
os.makedirs("classification_models", exist_ok=True)
joblib.dump(improved_model, "classification_models/classification_model.pkl")
joblib.dump(duration_model_final, "classification_models/duration_model.pkl")

print("\n" + "=" * 70)
print("ARTIFACTS SAVED")
print("=" * 70)
print("  classification_models/classification_model.pkl — XGBClassifier (5 features)")
print("  classification_models/duration_model.pkl       — XGBRegressor (duration predictor)")

# ── Explanations ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MODEL EXPLANATIONS")
print("=" * 70)

print(f"""
1. POSITIVE CLASS DEFINITION
   - Positive class (1) = "high fare": fare_amount > ${fare_threshold:.2f}
   - This threshold is the 75th percentile of training-set fares
   - Roughly 25% of trips are classified as high-fare

2. PRECISION / RECALL TRADEOFF & BUSINESS IMPACT
   - Precision ({improved_precision:.2f}): Of trips predicted as high-fare,
     {improved_precision:.0%} actually are. Higher precision means fewer false
     alarms (e.g., fewer unnecessary driver incentives for normal trips).
   - Recall ({improved_recall:.2f}): Of actual high-fare trips, we correctly
     identify {improved_recall:.0%}. Higher recall means we miss fewer
     lucrative trips (important for driver allocation/pricing).
   - Business tradeoff: If we send premium drivers to predicted high-fare
     trips, high precision avoids wasting premium resources; high recall
     ensures we capture most revenue opportunities.

3. CLASS BALANCE
   - ~75% low fare / ~25% high fare (imbalanced)
   - The model must learn the minority class well; AUC is a better metric
     than raw accuracy for imbalanced problems.

4. RISKS IF est_duration_min IS UNAVAILABLE IN PRODUCTION
   - est_duration_min relies on a duration prediction model at inference time.
   - If this model is unavailable (service down, latency issues), the
     classification model degrades to baseline performance (AUC={baseline_auc:.4f}
     vs {improved_auc:.4f}).
   - Mitigation: implement a fallback path that uses the 4-feature baseline
     model when est_duration_min cannot be computed. Monitor the duration
     model's health and set alerts for prediction failures.
   - Alternative: cache historical avg durations per route as a simple fallback.
""")
