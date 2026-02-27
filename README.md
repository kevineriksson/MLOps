# Classification Model — NYC Green Taxi High-Fare Prediction

## Overview
Predicts whether a NYC green taxi trip will be a **high-fare trip** (`fare_amount > 75th percentile`) at **booking time** using XGBoost classification.

## Features

### Baseline (4 features)
- `PULocationID` — Pickup location zone ID
- `DOLocationID` — Dropoff location zone ID
- `pickup_hour` — Hour of pickup (extracted from datetime)
- `passenger_count` — Number of passengers

### Improved (+1 feature)
- `est_duration_min` — Predicted trip duration from an out-of-fold (OOF) duration model trained on the 4 base features. Avoids data leakage by using only training-fold predictions.

## Metrics

| Metric    | Baseline | Improved |
|-----------|----------|----------|
| ROC AUC   | ~0.87    | ~0.88    |
| Precision | ~0.74    | ~0.76    |
| Recall    | ~0.65    | ~0.67    |
| F1 Score  | ~0.69    | ~0.71    |

*Exact values printed by `train.py` at runtime.*

## How to Run

```bash
# EDA — generates plots in plots/ directory
python eda.py

# Training — prints metrics and saves models to models/
python train.py
```

## Artifacts
- `models/classification_model.pkl` — Final XGBClassifier (5 features)
- `models/duration_model.pkl` — Duration prediction model used to generate `est_duration_min`
- `plots/` — EDA visualizations (including class balance chart)

## Data
- Source: `green_tripdata_2021-01.parquet` (~76K rows)
- Cleaning: trip_distance > 0 & < 100, fare_amount > 0 & < 99.9th percentile, trip_duration 0–180 min
- Target threshold computed on training set only (no leakage)
