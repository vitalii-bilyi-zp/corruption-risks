import os
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, ValidationError

# -------- Config --------
MODEL_PATH = os.getenv("MODEL_PATH", "restoration_model.pkl")
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "feature_columns.json")

# API key: must be passed in X-API-Key header
API_KEY = os.getenv("API_KEY")


def check_key(req) -> bool:
    return bool(API_KEY) and req.headers.get("X-API-Key") == API_KEY


# -------- Input schema --------
class Payload(BaseModel):
    area: float = Field(..., ge=0, description="Building area in m²")
    floors: int = Field(..., ge=0, description="Number of floors")
    building_type: str
    damage_level: str
    region: str
    repair_type: str


# -------- App --------
app = Flask(__name__)

# Load model and feature columns
with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
    FEATURE_COLUMNS = json.load(f)
MODEL = joblib.load(MODEL_PATH)

CATEGORICAL = ["building_type", "damage_level", "region", "repair_type"]


def preprocess_input(data_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([data_dict])
    df = pd.get_dummies(df, columns=CATEGORICAL)
    # Fill missing columns with 0 and reorder
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_COLUMNS]
    return df


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    if not check_key(request):
        return jsonify({"error": "unauthorized"}), 401
    try:
        payload = Payload.model_validate(request.get_json(force=True))
        X = preprocess_input(payload.model_dump())
        y_hat = float(MODEL.predict(X)[0])
        return jsonify({
            "predicted_cost": y_hat,
            "currency": "UAH",
            "model": "XGBoostRegressor"
        })
    except ValidationError as ve:
        return jsonify({"error": ve.errors()}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
