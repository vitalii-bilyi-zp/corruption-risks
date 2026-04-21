"""
compare_encodings.py - A/B порівняння методів кодування ознак.

Порівнює:
- One-Hot кодування (v1.0)
- Fuzzy кодування для порядкових ознак (v2.0)

Метрики:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (коефіцієнт детермінації)
- MAPE (Mean Absolute Percentage Error)

Методологія:
- K-Fold крос-валідація (k=5)
- Paired t-test для статистичної значущості
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import xgboost as xgb
from encoding import FeatureEncoder, FuzzyEncodingConfig

# Константи
N_FOLDS = 5
RANDOM_STATE = 42
CATEGORICAL = ["building_type", "damage_level", "region", "repair_type"]


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def encode_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot кодування (v1.0 baseline)."""
    df_encoded = pd.get_dummies(df.drop(columns=["restoration_cost"]), columns=CATEGORICAL)
    return df_encoded


def encode_fuzzy(df: pd.DataFrame) -> pd.DataFrame:
    """Fuzzy кодування для порядкових ознак (v2.0)."""
    encoder = FeatureEncoder(FuzzyEncodingConfig.default())
    return encoder.fit_transform(df.drop(columns=["restoration_cost"]))


def evaluate_encoding(
    df: pd.DataFrame,
    encode_func,
    encoding_name: str,
    n_folds: int = N_FOLDS
) -> dict:
    """
    Оцінює метод кодування з використанням K-Fold крос-валідації.

    Returns:
        dict з метриками по кожному фолду та середніми значеннями
    """
    y = df["restoration_cost"].values
    X = encode_func(df)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    results = {
        "encoding": encoding_name,
        "mae_folds": [],
        "rmse_folds": [],
        "r2_folds": [],
        "mape_folds": [],
        "predictions": [],
        "actuals": []
    }

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results["mae_folds"].append(mean_absolute_error(y_test, y_pred))
        results["rmse_folds"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        results["r2_folds"].append(r2_score(y_test, y_pred))
        results["mape_folds"].append(mape(y_test, y_pred))
        results["predictions"].extend(y_pred.tolist())
        results["actuals"].extend(y_test.tolist())

    # Середні значення
    results["mae_mean"] = np.mean(results["mae_folds"])
    results["mae_std"] = np.std(results["mae_folds"])
    results["rmse_mean"] = np.mean(results["rmse_folds"])
    results["rmse_std"] = np.std(results["rmse_folds"])
    results["r2_mean"] = np.mean(results["r2_folds"])
    results["r2_std"] = np.std(results["r2_folds"])
    results["mape_mean"] = np.mean(results["mape_folds"])
    results["mape_std"] = np.std(results["mape_folds"])

    return results


def test_monotonicity(df: pd.DataFrame, encode_func) -> dict:
    """
    Перевіряє монотонність передбачень по порядкових ознаках.

    Для damage_level: Легке < Середнє < Тяжке має давати зростаючу вартість.
    """
    X = encode_func(df)
    y = df["restoration_cost"].values

    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE)
    model.fit(X, y)

    # Базовий тестовий випадок
    base_sample = {
        "area": 5000,
        "floors": 5,
        "building_type": "Багатоповерховий будинок",
        "region": "м. Київ",
        "repair_type": "Капітальний ремонт"
    }

    damage_levels = ["Легке", "Середнє", "Тяжке"]
    predictions = []

    for level in damage_levels:
        sample = base_sample.copy()
        sample["damage_level"] = level
        sample_df = pd.DataFrame([sample])

        if encode_func == encode_onehot:
            X_sample = pd.get_dummies(sample_df, columns=CATEGORICAL)
            for col in X.columns:
                if col not in X_sample.columns:
                    X_sample[col] = 0
            X_sample = X_sample[X.columns]
        else:
            encoder = FeatureEncoder(FuzzyEncodingConfig.default())
            encoder._fitted_columns = list(X.columns)
            X_sample = encoder.transform(sample_df)

        pred = model.predict(X_sample)[0]
        predictions.append(pred)

    # Перевірка монотонності
    is_monotonic = predictions[0] <= predictions[1] <= predictions[2]

    return {
        "damage_level_predictions": dict(zip(damage_levels, predictions)),
        "is_monotonic": is_monotonic,
        "monotonicity_score": 1.0 if is_monotonic else 0.0
    }


def format_currency(value: float) -> str:
    """Форматує число як валюту (грн)."""
    return f"{value:,.0f}".replace(",", " ")


def print_comparison_report(onehot_results: dict, fuzzy_results: dict):
    """Виводить відформатований звіт порівняння."""
    def delta_pct(v1, v2):
        if v1 == 0:
            return 0
        return ((v2 - v1) / abs(v1)) * 100

    print("\n" + "=" * 70)
    print("COMPARISON OF FEATURE ENCODING METHODS")
    print("=" * 70)

    headers = ["Metrika", "One-Hot (v1.0)", "Fuzzy (v2.0)", "Delta%"]
    row_format = "| {:<18} | {:>17} | {:>14} | {:>8} |"
    separator = "+" + "-" * 20 + "+" + "-" * 19 + "+" + "-" * 16 + "+" + "-" * 10 + "+"

    print("+" + "-" * 20 + "+" + "-" * 19 + "+" + "-" * 16 + "+" + "-" * 10 + "+")
    print(row_format.format(*headers))
    print(separator)

    # MAE
    mae_delta = delta_pct(onehot_results["mae_mean"], fuzzy_results["mae_mean"])
    print(row_format.format(
        "MAE (UAH)",
        format_currency(onehot_results["mae_mean"]),
        format_currency(fuzzy_results["mae_mean"]),
        f"{mae_delta:+.1f}%"
    ))

    # RMSE
    rmse_delta = delta_pct(onehot_results["rmse_mean"], fuzzy_results["rmse_mean"])
    print(row_format.format(
        "RMSE (UAH)",
        format_currency(onehot_results["rmse_mean"]),
        format_currency(fuzzy_results["rmse_mean"]),
        f"{rmse_delta:+.1f}%"
    ))

    # R2
    r2_delta = delta_pct(onehot_results["r2_mean"], fuzzy_results["r2_mean"])
    print(row_format.format(
        "R2",
        f"{onehot_results['r2_mean']:.4f}",
        f"{fuzzy_results['r2_mean']:.4f}",
        f"{r2_delta:+.1f}%"
    ))

    # MAPE
    mape_delta = delta_pct(onehot_results["mape_mean"], fuzzy_results["mape_mean"])
    print(row_format.format(
        "MAPE (%)",
        f"{onehot_results['mape_mean']:.1f}%",
        f"{fuzzy_results['mape_mean']:.1f}%",
        f"{mape_delta:+.1f}%"
    ))

    print("+" + "-" * 20 + "+" + "-" * 19 + "+" + "-" * 16 + "+" + "-" * 10 + "+")

    # Paired t-test for MAE
    t_stat, p_value = stats.ttest_rel(
        onehot_results["mae_folds"],
        fuzzy_results["mae_folds"]
    )

    print(f"\nStatistical analysis (paired t-test on MAE):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Statistically significant (alpha=0.05): {'Yes' if p_value < 0.05 else 'No'}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)

    if mae_delta < 0:
        print(f"[+] Fuzzy encoding reduces MAE by {abs(mae_delta):.1f}%")
    else:
        print(f"[-] Fuzzy encoding increases MAE by {mae_delta:.1f}%")

    if r2_delta > 0:
        print(f"[+] Fuzzy encoding improves R2 by {r2_delta:.1f}%")
    else:
        print(f"[-] Fuzzy encoding reduces R2 by {abs(r2_delta):.1f}%")

    if p_value < 0.05:
        print("[+] Difference is statistically significant (p < 0.05)")
    else:
        print("[o] Difference is NOT statistically significant (p >= 0.05)")


def main():
    print("Loading data...")
    df = pd.read_csv("restoration_data.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    print(f"Dataset size: {len(df)} records")
    print(f"Cross-validation: {N_FOLDS} folds")

    print("\n[1/4] Evaluating One-Hot encoding (v1.0)...")
    onehot_results = evaluate_encoding(df, encode_onehot, "One-Hot (v1.0)")

    print("[2/4] Evaluating Fuzzy encoding (v2.0)...")
    fuzzy_results = evaluate_encoding(df, encode_fuzzy, "Fuzzy (v2.0)")

    print("[3/4] Testing prediction monotonicity...")
    onehot_mono = test_monotonicity(df, encode_onehot)
    fuzzy_mono = test_monotonicity(df, encode_fuzzy)

    print("[4/4] Generating report...")
    print_comparison_report(onehot_results, fuzzy_results)

    # Monotonicity
    print("\n" + "=" * 70)
    print("PREDICTION MONOTONICITY (damage_level)")
    print("=" * 70)

    print("\nOne-Hot encoding:")
    for level, pred in onehot_mono["damage_level_predictions"].items():
        print(f"  {level}: {format_currency(pred)} UAH")
    print(f"  Monotonic: {'Yes' if onehot_mono['is_monotonic'] else 'No'}")

    print("\nFuzzy encoding:")
    for level, pred in fuzzy_mono["damage_level_predictions"].items():
        print(f"  {level}: {format_currency(pred)} UAH")
    print(f"  Monotonic: {'Yes' if fuzzy_mono['is_monotonic'] else 'No'}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    mae_improvement = ((onehot_results["mae_mean"] - fuzzy_results["mae_mean"]) /
                       onehot_results["mae_mean"]) * 100

    if mae_improvement > 0:
        print(f"\nFuzzy encoding is RECOMMENDED for implementation:")
        print(f"  - MAE reduction: {mae_improvement:.1f}%")
        print(f"  - Prediction error savings: ~{format_currency(onehot_results['mae_mean'] - fuzzy_results['mae_mean'])} UAH/object")
    else:
        print(f"\nFuzzy encoding does NOT provide accuracy advantages.")
        print(f"  - MAE change: {mae_improvement:.1f}%")


if __name__ == "__main__":
    main()
