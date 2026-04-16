"""
Flask Prediction App — NYC Green Taxi Fare Prediction
Loads the Production model from MLflow Model Registry and a duration model
from the shared PVC to serve fare predictions via a web form.
"""

import os
import time
import pandas as pd
import numpy as np
import mlflow
import joblib
from flask import Flask, request, render_template_string

app = Flask(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
DATA_DIR = os.environ.get("DATA_DIR", "/mlflow-data")
MODEL_NAME = "GreenTaxi-XGBoost"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

fare_model = None
duration_model = None

def load_models():
    global fare_model, duration_model

    # Load fare prediction model from MLflow (Production stage)
    for attempt in range(30):
        try:
            model_uri = f"models:/{MODEL_NAME}/Production"
            fare_model = mlflow.pyfunc.load_model(model_uri)
            print(f"Loaded {MODEL_NAME} (Production) from MLflow")
            break
        except Exception as e:
            print(f"Waiting for model... attempt {attempt + 1}/30 ({e})")
            time.sleep(10)
    else:
        raise RuntimeError(f"Could not load {MODEL_NAME} from MLflow after 30 attempts")

    # Load duration model from shared PVC
    dur_path = os.path.join(DATA_DIR, "duration_model.pkl")
    for attempt in range(10):
        if os.path.exists(dur_path):
            duration_model = joblib.load(dur_path)
            print(f"Loaded duration model from {dur_path}")
            return
        print(f"Waiting for duration_model.pkl... attempt {attempt + 1}/10")
        time.sleep(5)

    raise RuntimeError(f"duration_model.pkl not found at {dur_path}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NYC Green Taxi — Fare Prediction</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: Arial, sans-serif; background: #f4f4f4; color: #111; }
  .header { background: #1a1a2e; color: #fff; padding: 20px 30px; }
  .header h1 { font-size: 22px; font-weight: 600; }
  .header p { font-size: 13px; color: #aaa; margin-top: 4px; }
  .container { max-width: 600px; margin: 30px auto; padding: 0 20px; }
  .card { background: #fff; border-radius: 8px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
  .card h2 { font-size: 16px; margin-bottom: 16px; color: #333; }
  label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 4px; color: #444; }
  .hint { font-size: 11px; color: #888; margin-bottom: 8px; }
  input[type="number"] {
    width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px;
    font-size: 14px; margin-bottom: 4px;
  }
  input:focus { outline: none; border-color: #1a1a2e; }
  .row { display: flex; gap: 16px; }
  .row > div { flex: 1; }
  .btn {
    display: block; width: 100%; margin-top: 20px; padding: 12px;
    background: #1a1a2e; color: #fff; border: none; border-radius: 4px;
    font-size: 15px; font-weight: 600; cursor: pointer;
  }
  .btn:hover { background: #16213e; }
  .result {
    margin-top: 20px; padding: 20px; border-radius: 8px; text-align: center;
    background: #d4edda; border: 1px solid #b7d7be;
  }
  .result .fare { font-size: 36px; font-weight: 700; color: #1a7a1a; }
  .result .label { font-size: 13px; color: #555; margin-bottom: 4px; }
  .result .details { font-size: 12px; color: #777; margin-top: 8px; }
  .footer { text-align: center; font-size: 11px; color: #999; margin: 30px 0; }
</style>
</head>
<body>

<div class="header">
  <h1>NYC Green Taxi Fare Prediction</h1>
  <p>Model: {{ model_name }} (Production) &mdash; Powered by MLflow + Kubernetes</p>
</div>

<div class="container">
  <div class="card">
    <h2>Enter Trip Details</h2>
    <form method="POST">
      <div class="row">
        <div>
          <label for="pu_location">Pickup Location ID</label>
          <div class="hint">NYC taxi zone (1&ndash;265)</div>
          <input type="number" id="pu_location" name="pu_location"
                 min="1" max="265" value="{{ pu_location or 43 }}" required>
        </div>
        <div>
          <label for="do_location">Dropoff Location ID</label>
          <div class="hint">NYC taxi zone (1&ndash;265)</div>
          <input type="number" id="do_location" name="do_location"
                 min="1" max="265" value="{{ do_location or 238 }}" required>
        </div>
      </div>
      <div class="row" style="margin-top: 12px;">
        <div>
          <label for="pickup_hour">Pickup Hour</label>
          <div class="hint">Hour of day (0&ndash;23)</div>
          <input type="number" id="pickup_hour" name="pickup_hour"
                 min="0" max="23" value="{{ pickup_hour or 14 }}" required>
        </div>
        <div>
          <label for="passenger_count">Passengers</label>
          <div class="hint">Number of passengers (1&ndash;9)</div>
          <input type="number" id="passenger_count" name="passenger_count"
                 min="1" max="9" value="{{ passenger_count or 1 }}" required>
        </div>
      </div>
      <button type="submit" class="btn">Predict Fare</button>
    </form>

    {% if prediction is not none %}
    <div class="result">
      <div class="label">Predicted Fare</div>
      <div class="fare">${{ "%.2f"|format(prediction) }}</div>
      <div class="details">
        Estimated trip duration: {{ "%.1f"|format(est_duration) }} min
      </div>
    </div>
    {% endif %}
  </div>
</div>

<div class="footer">
  MLflow Tracking: {{ mlflow_uri }} &bull; Model: {{ model_name }}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    est_duration = None
    pu_location = None
    do_location = None
    pickup_hour = None
    passenger_count = None

    if request.method == "POST":
        pu_location = float(request.form["pu_location"])
        do_location = float(request.form["do_location"])
        pickup_hour = float(request.form["pickup_hour"])
        passenger_count = float(request.form["passenger_count"])

        base_features = pd.DataFrame({
            "PULocationID": [pu_location],
            "DOLocationID": [do_location],
            "pickup_hour": [pickup_hour],
            "passenger_count": [passenger_count],
        })

        est_duration = float(duration_model.predict(base_features)[0])

        features = base_features.copy()
        features["est_duration_min"] = est_duration

        prediction = float(fare_model.predict(features)[0])

    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction,
        est_duration=est_duration,
        pu_location=pu_location,
        do_location=do_location,
        pickup_hour=pickup_hour,
        passenger_count=passenger_count,
        model_name=MODEL_NAME,
        mlflow_uri=MLFLOW_TRACKING_URI,
    )


@app.route("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=8080)
