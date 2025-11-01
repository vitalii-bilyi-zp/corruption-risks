import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import joblib
import json


def save_columns(feature_columns):
    feature_columns.remove("restoration_cost")  # приховуємо цільову змінну

    # Запишемо список колонок у JSON-файл
    with open("feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=4)


def encode_ordinal(df, columns):
    for col in columns:
        df[col] = df[col].astype("category").cat.codes + 1
        # +1 – щоб кодування починалося з 1, а не з 0
    return df


def main():
    # --- Крок 1: Завантаження та підготовка даних ---
    df = pd.read_csv("restoration_data.csv", encoding="utf-8-sig")

    # shuffle in random order
    # first_row = df.iloc[[0]]
    # shuffled_rest = df.iloc[1:].sample(frac=1).reset_index(drop=True)
    # df = pd.concat([first_row, shuffled_rest], ignore_index=True)

    df.columns = df.columns.str.strip()

    # Застосовуємо порядкове кодування для категоріальних ознак
    categorical_cols = ["building_type", "damage_level", "region", "repair_type"]
    df = encode_ordinal(df, categorical_cols)

    save_columns(df.columns.tolist())

    # Розділяємо цільову змінну та ознаки
    y = df["restoration_cost"]
    x = df.drop(columns=["restoration_cost"])

    # Ділимо на навчальну та тестову вибірки
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # --- Крок 2: Навчання моделі ---
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Оцінка якості (за бажанням)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae:.2f} грн")

    # --- Крок 3: Зберігаємо модель ---
    joblib.dump(model, "restoration_model_lr.pkl")

    # --- Крок 4: Вивід коефіцієнтів (АНАЛІТИЧНИЙ ВИРАЗ) ---
    feature_names = x_train.columns
    coefs = model.coef_
    intercept = model.intercept_

    print("\nАналітичний вираз лінійної регресії (з порядковим кодуванням):")
    print(f"V = {intercept:.2f}", end=" ")
    for name, coef in zip(feature_names, coefs):
        sign = "+" if coef >= 0 else "-"
        print(f"{sign} {abs(coef):.2f} * {name}", end=" ")
    print("\n")


if __name__ == '__main__':
    main()
