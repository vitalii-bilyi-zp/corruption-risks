import os
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pathlib import Path
from typing import Optional, List

from encoding import preprocess_for_inference
from inflation import InflationService, BASE_YEAR, BASE_MONTH

# -------- Configuration --------
MODEL_PATH = os.getenv("MODEL_PATH", "restoration_model.pkl")
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "feature_columns.json")
FEATURE_GROUPS_PATH = os.getenv("FEATURE_GROUPS_PATH", "feature_groups.json")

# API key must be passed in X-API-Key header
API_KEY = os.getenv("API_KEY")


def check_key(req) -> bool:
    """Check API key for authentication"""
    return bool(API_KEY) and req.headers.get("X-API-Key") == API_KEY


def _serializable_errors(ve: ValidationError) -> list:
    """
    Pydantic v2 включає ctx.error як об'єкт Exception, який не серіалізується
    стандартним JSON-енкодером Flask. Конвертуємо його у рядок.
    """
    result = []
    for err in ve.errors(include_url=False):
        entry = dict(err)
        if "ctx" in entry and isinstance(entry["ctx"].get("error"), Exception):
            entry["ctx"] = {**entry["ctx"], "error": str(entry["ctx"]["error"])}
        result.append(entry)
    return result


# -------- Input schema --------
class InflationIndexRow(BaseModel):
    year:        int
    month:       int
    index_value: float


ALLOWED_DAMAGE_LEVELS = {"Легке", "Середнє", "Тяжке"}
ALLOWED_REPAIR_TYPES  = {"Поточний ремонт", "Капітальний ремонт", "Реставрація", "Знесення з новим будівництвом", "Консервація"}


class Payload(BaseModel):
    area: float = Field(..., ge=0, description="Building area in m²")
    floors: int = Field(..., ge=0, description="Number of floors")
    building_type: str
    damage_level: str
    region: str
    repair_type: str
    work_year:  Optional[int] = Field(default=None, ge=2020, le=2040)
    work_month: Optional[int] = Field(default=None, ge=1, le=12)
    inflation_indices: Optional[List[InflationIndexRow]] = None

    @field_validator("damage_level")
    @classmethod
    def validate_damage_level(cls, v: str) -> str:
        if v not in ALLOWED_DAMAGE_LEVELS:
            raise ValueError(
                f"Недопустимий рівень пошкодження: '{v}'. "
                f"Допустимі значення: {sorted(ALLOWED_DAMAGE_LEVELS)}"
            )
        return v

    @field_validator("repair_type")
    @classmethod
    def validate_repair_type(cls, v: str) -> str:
        if v not in ALLOWED_REPAIR_TYPES:
            raise ValueError(
                f"Недопустимий тип ремонту: '{v}'. "
                f"Допустимі значення: {sorted(ALLOWED_REPAIR_TYPES)}"
            )
        return v

    @model_validator(mode='after')
    def check_year_month_together(self):
        year_given  = self.work_year  is not None
        month_given = self.work_month is not None
        if year_given != month_given:
            raise ValueError(
                'work_year та work_month мають передаватись разом або не передаватись взагалі'
            )
        return self


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

inflation_svc = InflationService()


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
        input_data = payload.model_dump(exclude={"work_year", "work_month", "inflation_indices"})
        X = preprocess_input(input_data)
        y_hat = float(MODEL.predict(X)[0])
        try:
            inflation = inflation_svc.apply(
                y_hat,
                payload.work_year,
                payload.work_month,
                payload.inflation_indices,
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({
            "predicted_cost": round(y_hat, 2),
            "adjusted_cost":  inflation["adjusted_cost"],
            "inflation_k":    inflation["inflation_k"],
            "base_year":      inflation["base_year"],
            "base_month":     inflation["base_month"],
            "work_year":      inflation["work_year"],
            "work_month":     inflation["work_month"],
            "currency":       "UAH",
            "model":          "XGBoostRegressor",
        })
    except ValidationError as ve:
        return jsonify({"error": _serializable_errors(ve)}), 400
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
        input_data = payload.model_dump(exclude={"work_year", "work_month", "inflation_indices"})
        X = preprocess_input(input_data)
        y_hat = float(MODEL.predict(X)[0])
        try:
            inflation = inflation_svc.apply(
                y_hat,
                payload.work_year,
                payload.work_month,
                payload.inflation_indices,
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

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
            "predicted_cost": round(y_hat, 2),
            "adjusted_cost":  inflation["adjusted_cost"],
            "inflation_k":    inflation["inflation_k"],
            "base_year":      inflation["base_year"],
            "base_month":     inflation["base_month"],
            "work_year":      inflation["work_year"],
            "work_month":     inflation["work_month"],
            "currency":       "UAH",
            "model":          "XGBoostRegressor",
            "base_value":     base_value,
            "contributions":  contributions,
        })

    except ValidationError as ve:
        return jsonify({"error": _serializable_errors(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
