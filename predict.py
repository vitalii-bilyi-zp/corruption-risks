import pandas as pd
import joblib
import json

# Путь к сохранённой модели
MODEL_PATH = "restoration_model.pkl"
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
    # Загружаем модель
    model = joblib.load(MODEL_PATH)

    # Загружаем список колонок из JSON
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    # Пример входных данных
    input_data = {
        "площадь": 70,
        "этажность": 1,
        "тип_здания": "частный",
        "степень_повреждения": "средняя",
        "регион": "Львов"
    }

    x_new = preprocess_input(input_data, feature_columns)

    predicted_cost = model.predict(x_new)[0]

    print(f"Предсказанная стоимость восстановления: {predicted_cost:.2f} грн")


if __name__ == "__main__":
    main()
