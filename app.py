import os
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, ValidationError
from pathlib import Path

from encoding import preprocess_for_inference

# -------- Configuration --------
MODEL_PATH = os.getenv("MODEL_PATH", "restoration_model.pkl")
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "feature_columns.json")
FEATURE_GROUPS_PATH = os.getenv("FEATURE_GROUPS_PATH", "feature_groups.json")

# API key must be passed in X-API-Key header
API_KEY = os.getenv("API_KEY")


def check_key(req) -> bool:
    """Check API key for authentication"""
    return bool(API_KEY) and req.headers.get("X-API-Key") == API_KEY


# -------- Input schema --------
class Payload(BaseModel):
    area: float = Field(..., ge=0, description="Building area in m²")
    floors: int = Field(..., ge=0, description="Number of floors")
    building_type: str
    damage_level: str
    region: str
    repair_type: str


# -------- Flask App --------
app = Flask(__name__)

# Load model, feature columns and groups at startup
with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
    FEATURE_COLUMNS = json.load(f)

FEATURE_GROUPS = {}
if Path(FEATURE_GROUPS_PATH).exists():
    with open(FEATURE_GROUPS_PATH, "r", encoding="utf-8") as f:
        FEATURE_GROUPS = json.load(f)

MODEL = joblib.load(MODEL_PATH)

# Визначаємо версію кодування з метаданих
ENCODING_VERSION = FEATURE_GROUPS.get("version", "1.0")


def preprocess_input(data_dict: dict) -> pd.DataFrame:
    """Convert input JSON into DataFrame with the same structure as during training.
    Автоматично визначає версію кодування (v1.0: one-hot, v2.0: fuzzy).
    """
    return preprocess_for_inference(data_dict, FEATURE_COLUMNS, FEATURE_GROUPS)


def get_feature_groups_for_aggregation():
    """
    Повертає групи ознак для агрегації SHAP-значень.
    Виключає службові поля (version).
    """
    return {k: v for k, v in FEATURE_GROUPS.items() if isinstance(v, list)}


def aggregate_local_contrib(shap_vec, feature_columns, feature_groups):
    """
    Aggregate local SHAP values into human-readable feature groups.
    Returns:
      - ordered: list of tuples (group, contribution)
      - per_group_pct: percentage impact per group
    """
    per_feature = {feat: float(val) for feat, val in zip(feature_columns, shap_vec)}
    per_group = {}
    for g, cols in feature_groups.items():
        if not isinstance(cols, list):
            continue
        per_group[g] = sum(per_feature.get(c, 0.0) for c in cols)

    # Compute relative impact in %
    denom = sum(abs(v) for v in per_group.values()) or 1.0
    per_group_pct = {g: (abs(v) / denom) * 100.0 for g, v in per_group.items()}

    ordered = sorted(per_group.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return ordered, per_group_pct


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    """Basic endpoint: returns only the predicted cost"""
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


@app.post("/predict_explain")
def predict_explain():
    """
    Extended endpoint: returns prediction + explainability info (SHAP or fallback)
    Response fields:
      - predicted_cost
      - base_value (if SHAP available)
      - contributions: list of {group, contribution, percent}
    """
    if not check_key(request):
        return jsonify({"error": "unauthorized"}), 401
    try:
        payload = Payload.model_validate(request.get_json(force=True))
        X = preprocess_input(payload.model_dump())
        y_hat = float(MODEL.predict(X)[0])

        feature_groups = get_feature_groups_for_aggregation()

        base_value = None
        contributions = []

        try:
            import shap
            explainer = shap.TreeExplainer(MODEL)
            shap_values = explainer.shap_values(X)  # shape: (1, n_features)
            local = shap_values[0]
            base_value = float(explainer.expected_value)

            ordered, per_group_pct = aggregate_local_contrib(local, FEATURE_COLUMNS, feature_groups)

            for g, val in ordered:
                contributions.append({
                    "group": g,
                    "contribution": float(val),        # impact on cost (+/-)
                    "percent": float(per_group_pct[g]) # relative importance (%)
                })

        except Exception as shap_err:
            # Fallback: use global feature importance (gain) from XGBoost
            booster = MODEL.get_booster()
            gain = booster.get_score(importance_type="gain")
            per_group = {}
            for g, cols in feature_groups.items():
                if not isinstance(cols, list):
                    continue
                per_group[g] = sum(float(gain.get(col, 0.0)) for col in cols)
            total = sum(per_group.values()) or 1.0
            for g, v in sorted(per_group.items(), key=lambda kv: kv[1], reverse=True):
                pct = (v / total) * 100.0
                contributions.append({
                    "group": g,
                    "contribution": None,
                    "percent": float(pct)
                })

        return jsonify({
            "predicted_cost": y_hat,
            "currency": "UAH",
            "model": "XGBoostRegressor",
            "base_value": base_value,
            "contributions": contributions
        })

    except ValidationError as ve:
        return jsonify({"error": ve.errors()}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
