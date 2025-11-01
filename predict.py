import pandas as pd
import joblib
import json
from pathlib import Path

MODEL_PATH = "restoration_model.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.json"
FEATURE_GROUPS_PATH = "feature_groups.json"

CATEGORICAL = ["building_type", "damage_level", "region", "repair_type"]


def preprocess_input(data_dict, feature_columns):
    """Prepare the input data into the same structure used during model training."""
    df = pd.DataFrame([data_dict])
    df = pd.get_dummies(df, columns=CATEGORICAL)
    # Add missing columns and reorder to match the training feature order
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return df


def load_json(path):
    """Load JSON file and return the parsed object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_local_shap(shap_vec, feature_columns, feature_groups):
    """Aggregate local SHAP values by human-readable feature groups (e.g., area, floors, repair_type)."""
    per_feature = {feat: float(val) for feat, val in zip(feature_columns, shap_vec)}
    per_group = {}
    for g, cols in feature_groups.items():
        per_group[g] = sum(per_feature.get(c, 0.0) for c in cols)

    # Compute absolute sum to calculate relative importance (% impact)
    denom = sum(abs(v) for v in per_group.values()) or 1.0
    per_group_pct = {g: abs(v) / denom * 100.0 for g, v in per_group.items()}

    # Sort for convenience by absolute contribution
    ordered = sorted(per_group.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return ordered, per_group_pct


def main():
    # 1) Load model and metadata
    model = joblib.load(MODEL_PATH)
    feature_columns = load_json(FEATURE_COLUMNS_PATH)
    if Path(FEATURE_GROUPS_PATH).exists():
        feature_groups = load_json(FEATURE_GROUPS_PATH)
    else:
        # Fallback: build feature groups by prefix (for older models)
        feature_groups = {
            "area": ["area"], "floors": ["floors"],
            "building_type": [c for c in feature_columns if c.startswith("building_type_")],
            "damage_level": [c for c in feature_columns if c.startswith("damage_level_")],
            "region": [c for c in feature_columns if c.startswith("region_")],
            "repair_type": [c for c in feature_columns if c.startswith("repair_type_")],
        }

    # 2) Example input (replace with your own JSON or live input)
    input_data = {
        "area": 7200,
        "floors": 9,
        "building_type": "multi-storey",
        "damage_level": "medium",
        "region": "Lviv",
        "repair_type": "capital"
    }

    X = preprocess_input(input_data, feature_columns)
    y_hat = float(model.predict(X)[0])

    print(f"\nPredicted restoration cost: {y_hat:,.2f} UAH")

    # 3) Local SHAP explainability
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)  # shape: (1, n_features)
        base_value = float(explainer.expected_value)
        local = shap_values[0]

        ordered, per_group_pct = aggregate_local_shap(local, feature_columns, feature_groups)

        print("\nFeature group contributions (local for this object):")
        for g, val in ordered:
            sign = "+" if val >= 0 else "−"
            pct = per_group_pct.get(g, 0.0)
            print(f"  {g:14s}: {sign}{abs(val):,.2f} (≈ {pct:5.1f}%)")

        # Validate decomposition
        recon = base_value + local.sum()
        print(f"\nDecomposition check: base_value + ΣSHAP ≈ prediction")
        print(f"  base_value = {base_value:,.2f}")
        print(f"  ΣSHAP      = {local.sum():,.2f}")
        print(f"  recon      = {recon:,.2f}")
        print(f"  y_hat      = {y_hat:,.2f}")

    except Exception as e:
        print(f"\n[WARN] SHAP not available ({e}). Explainability disabled.")
        # Optional: implement fallback using global feature importance (gain),
        # but note that it reflects global importance, not local instance-specific impact.


if __name__ == "__main__":
    main()
