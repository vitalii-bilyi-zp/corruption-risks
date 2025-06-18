import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import json


def save_columns(feature_columns):
    feature_columns.remove("стоимость_восстановления")  # убираем целевую переменную

    # Запишем список колонок в JSON-файл
    with open("feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=4)


def main():
    # --- Шаг 1: Загрузка и подготовка данных ---
    df = pd.read_csv("restoration_data.csv")
    first_row = df.iloc[[0]]
    shuffled_rest = df.iloc[1:].sample(frac=1).reset_index(drop=True)
    df = pd.concat([first_row, shuffled_rest], ignore_index=True)
    df.columns = df.columns.str.strip()

    # Кодируем категориальные признаки с помощью get_dummies
    df = pd.get_dummies(df, columns=["тип_здания", "степень_повреждения", "регион"])

    save_columns(df.columns.tolist())

    # Разделяем целевую переменную и признаки
    y = df["стоимость_восстановления"]
    x = df.drop(columns=["стоимость_восстановления"])

    # Делим на обучающую и тестовую выборки
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # --- Шаг 2: Обучение модели ---
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(x_train, y_train)

    # --- Шаг 3: Оценка модели (по желанию) ---
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae:.2f} грн")

    # plt.scatter(y_test, y_pred, alpha=0.5)
    # plt.xlabel("Фактическая стоимость")
    # plt.ylabel("Предсказанная стоимость")
    # plt.title("Факт vs Предсказание")
    # plt.grid(True)
    # plt.show()

    # --- Шаг 4: Сохраняем модель через joblib ---
    joblib.dump(model, "restoration_model.pkl")


if __name__ == '__main__':
    main()
