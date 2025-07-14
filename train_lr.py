import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import joblib
import json


def save_columns(feature_columns):
    feature_columns.remove("вартість_відновлення")  # приховуємо цільову змінну

    # Запишемо список колонок у JSON-файл
    with open("feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=4)


def main():
    # --- Крок 1: Завантаження та підготовка даних ---
    df = pd.read_csv("restoration_data.csv")
    first_row = df.iloc[[0]]
    shuffled_rest = df.iloc[1:].sample(frac=1).reset_index(drop=True)
    df = pd.concat([first_row, shuffled_rest], ignore_index=True)
    df.columns = df.columns.str.strip()

    # Кодуємо категоріальні ознаки за допомогою get_dummies
    df = pd.get_dummies(df, columns=["тип_будівлі", "ступінь_пошкодження", "регіон"])

    save_columns(df.columns.tolist())

    # Розділяємо цільову змінну та ознаки
    y = df["вартість_відновлення"]
    x = df.drop(columns=["вартість_відновлення"])

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


if __name__ == '__main__':
    main()
