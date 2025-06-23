import pandas as pd
import joblib
import json

MODEL_PATH = "restoration_model_rf.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.json"


def preprocess_input(data_dict, feature_columns):
    df = pd.DataFrame([data_dict])
    df = pd.get_dummies(df, columns=["тип_здания", "степень_повреждения", "регион"])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]
    return df


def main():
    # Завантажуємо модель
    model = joblib.load(MODEL_PATH)

    # Завантажуємо список колонок
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    # Приклад вхідних даних
    input_data = {
        "площадь": 7200,
        "этажность": 9,
        "тип_здания": "багатоповерховий",
        "степень_повреждения": "середнє",
        "регион": "Львів"
    }

    x_new = preprocess_input(input_data, feature_columns)

    predicted_cost = model.predict(x_new)[0]

    print(f"Передбачена вартість відновлення: {predicted_cost:.2f} грн")


if __name__ == "__main__":
    main()
