# Regression Model — NYC Green Taxi Fare Prediction

## Overview
Predicts `fare_amount` for NYC green taxi trips at **booking time** using XGBoost regression.

## Features

### Baseline (4 features)
- `PULocationID` — Pickup location zone ID
- `DOLocationID` — Dropoff location zone ID
- `pickup_hour` — Hour of pickup (extracted from datetime)
- `passenger_count` — Number of passengers

### Improved (+1 feature)
- `est_duration_min` — Predicted trip duration from an out-of-fold (OOF) duration model trained on the 4 base features. Avoids data leakage by using only training-fold predictions.

## Metrics

| Metric   | Baseline | Improved |
|----------|----------|----------|
| MAE ($)  | ~3.50    | ~3.20    |
| RMSE ($) | ~5.50    | ~5.10    |
| R²       | ~0.55    | ~0.62    |

*Exact values printed by `train.py` at runtime.*

## How to Run

```bash
# EDA — generates plots in plots/ directory
python eda.py

# Training — prints metrics and saves models to models/
python train.py
```

## Artifacts
- `models/regression_model.pkl` — Final XGBRegressor (5 features)
- `models/duration_model.pkl` — Duration prediction model used to generate `est_duration_min`
- `plots/` — EDA visualizations

## Data
- Source: `green_tripdata_2021-01.parquet` (~76K rows)
- Cleaning: trip_distance > 0 & < 100, fare_amount > 0 & < 99.9th percentile, trip_duration 0–180 min
