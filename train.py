import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import joblib
import json

from encoding import preprocess_for_training

# ---- Output artifact constants ----
MODEL_PATH = "restoration_model.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.json"
FEATURE_GROUPS_PATH = "feature_groups.json"
GLOBAL_IMPORTANCE_PATH = "global_importance.json"


def safe_dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def main():
    # --- Step 1: Load and prepare data ---
    df = pd.read_csv("restoration_data.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Використовуємо централізований модуль кодування (fuzzy для порядкових ознак)
    y = df["restoration_cost"]
    X_raw = df.drop(columns=["restoration_cost"])
    X, encoder = preprocess_for_training(X_raw)

    # Save feature list and groups
    encoder.save_metadata(FEATURE_COLUMNS_PATH, FEATURE_GROUPS_PATH)
    feature_columns = encoder.get_feature_columns()
    feature_groups = encoder.get_feature_groups()

    # --- Step 2: Split target and features ---
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # --- Step 3: Train model ---
    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=1)
    model.fit(x_train, y_train)

    # --- Step 4: Evaluate ---
    mae = mean_absolute_error(y_test, model.predict(x_test))
    print(f"MAE: {mae:.2f} грн")

    # --- Step 5: Persist model ---
    joblib.dump(model, MODEL_PATH)

    # --- Step 6: Global importance (SHAP → fallback: XGBoost gain) ---
    global_importance = {}
    try:
        import shap  # pip install shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)  # shape: (n_samples, n_features)
        abs_mean = (abs(shap_values)).mean(axis=0)   # mean absolute contribution per feature

        # Map to dict: feature -> score
        per_feature = {feat: float(score) for feat, score in zip(feature_columns, abs_mean)}

        # Aggregate by groups: sum of absolute SHAP over group columns
        per_group = {}
        for g, cols in feature_groups.items():
            if g == "version" or not isinstance(cols, list):
                continue
            per_group[g] = float(sum(per_feature.get(c, 0.0) for c in cols))

        # Normalize to 100%
        total = sum(per_group.values()) or 1.0
        per_group_pct = {g: (v / total) * 100.0 for g, v in per_group.items()}

        # Sorted dict
        global_importance = dict(sorted(per_group_pct.items(), key=lambda kv: kv[1], reverse=True))
        print("\nGlobal group importance (SHAP, %):")
        for k, v in global_importance.items():
            print(f"  {k}: {v:.1f}%")
    except Exception as e:
        print(f"\n[WARN] SHAP not available ({e}). Falling back to XGBoost gain.")
        booster = model.get_booster()
        gain = booster.get_score(importance_type="gain")  # dict: feature -> gain
        # Expand to full dict
        per_feature = {feat: float(gain.get(feat, 0.0)) for feat in feature_columns}
        per_group = {}
        for g, cols in feature_groups.items():
            if g == "version" or not isinstance(cols, list):
                continue
            per_group[g] = float(sum(per_feature.get(c, 0.0) for c in cols))
        total = sum(per_group.values()) or 1.0
        global_importance = {g: (v / total) * 100.0 for g, v in per_group.items()}
        global_importance = dict(sorted(global_importance.items(), key=lambda kv: kv[1], reverse=True))
        print("Global group importance (gain, %):")
        for k, v in global_importance.items():
            print(f"  {k}: {v:.1f}%")

    safe_dump_json(global_importance, GLOBAL_IMPORTANCE_PATH)

    # --- (Optional) dump first tree text ---
    booster = model.get_booster()
    tree_text = booster.get_dump(with_stats=False)[0]
    print("\nXGBoost tree textual dump (first tree):")
    print(tree_text)


if __name__ == "__main__":
    main()
