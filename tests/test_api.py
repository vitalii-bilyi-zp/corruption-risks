"""
tests/test_api.py — pytest-тести для Flask API прогнозування вартості відновлення.

Запуск:
    pytest tests/test_api.py -v

Якщо файли моделі відсутні — тести, що потребують реальної моделі,
автоматично пропускаються (pytest.mark.skipif).
"""

import os
import sys
import pytest

# Додаємо корінь проєкту до шляху пошуку модулів
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Перевірка наявності артефактів моделі (потрібна для більшості тестів)
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "restoration_model.pkl")
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "feature_columns.json")
FEATURE_GROUPS_PATH = os.getenv("FEATURE_GROUPS_PATH", "feature_groups.json")

_model_files_exist = (
    os.path.exists(MODEL_PATH)
    and os.path.exists(FEATURE_COLUMNS_PATH)
    and os.path.exists(FEATURE_GROUPS_PATH)
)

requires_model = pytest.mark.skipif(
    not _model_files_exist,
    reason=(
        f"Артефакти моделі не знайдено: {MODEL_PATH}, "
        f"{FEATURE_COLUMNS_PATH}, {FEATURE_GROUPS_PATH}. "
        "Спочатку виконайте: python train.py"
    ),
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
TEST_API_KEY = "test-secret-key-123"


@pytest.fixture(scope="session")
def app():
    """Створює Flask-застосунок з тестовим API-ключем."""
    os.environ["API_KEY"] = TEST_API_KEY
    from app import app as flask_app
    flask_app.config["TESTING"] = True
    yield flask_app


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def auth_headers():
    """Заголовки з валідним API-ключем."""
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def valid_payload():
    """Мінімально коректний payload без інфляційного коригування."""
    return {
        "area": 150.5,
        "floors": 3,
        "building_type": "Багатоповерховий будинок",
        "damage_level": "Середнє",
        "region": "м. Запоріжжя",
        "repair_type": "Капітальний",
    }


@pytest.fixture
def valid_payload_with_inflation(valid_payload):
    """Payload із параметрами інфляційного коригування."""
    return {
        **valid_payload,
        "work_year": 2026,
        "work_month": 4,
        "inflation_indices": [
            {"year": 2022, "month": 1, "index_value": 0.75},
            {"year": 2024, "month": 1, "index_value": 1.0},
            {"year": 2026, "month": 4, "index_value": 1.24},
        ],
    }


# ===========================================================================
# TestAuthentication
# ===========================================================================
@requires_model
class TestAuthentication:
    def test_predict_returns_401_without_api_key(self, client, valid_payload):
        """Запит без заголовка X-API-Key повинен повернути 401."""
        resp = client.post("/predict", json=valid_payload)
        assert resp.status_code == 401

    def test_predict_returns_401_with_invalid_api_key(self, client, valid_payload):
        """Запит з невалідним API-ключем повинен повернути 401."""
        resp = client.post(
            "/predict",
            json=valid_payload,
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_predict_returns_200_with_valid_api_key(self, client, auth_headers, valid_payload):
        """Запит з валідним API-ключем повинен повернути 200."""
        resp = client.post("/predict", json=valid_payload, headers=auth_headers)
        assert resp.status_code == 200

    def test_predict_explain_returns_401_without_api_key(self, client, valid_payload):
        """predict_explain без заголовка X-API-Key повинен повернути 401."""
        resp = client.post("/predict_explain", json=valid_payload)
        assert resp.status_code == 401


# ===========================================================================
# TestValidation
# ===========================================================================
@requires_model
class TestValidation:
    def test_missing_required_field_area_returns_400(self, client, auth_headers, valid_payload):
        """Відсутнє обов'язкове поле area → 400."""
        payload = {k: v for k, v in valid_payload.items() if k != "area"}
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 400

    def test_invalid_type_for_area_returns_400(self, client, auth_headers, valid_payload):
        """Некоректний тип area (рядок замість числа) → 400."""
        payload = {**valid_payload, "area": "abc"}
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 400

    def test_invalid_building_type_returns_400(self, client, auth_headers, valid_payload):
        """Неприпустиме значення building_type → 400."""
        payload = {**valid_payload, "building_type": "Невідомий тип будівлі"}
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 400

    def test_work_year_without_work_month_returns_400(self, client, auth_headers, valid_payload):
        """work_year без work_month → 400 (мають передаватись разом)."""
        payload = {**valid_payload, "work_year": 2026}
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 400

    def test_work_month_without_work_year_returns_400(self, client, auth_headers, valid_payload):
        """work_month без work_year → 400 (мають передаватись разом)."""
        payload = {**valid_payload, "work_month": 4}
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 400

    def test_work_year_month_without_inflation_indices_returns_400(
        self, client, auth_headers, valid_payload
    ):
        """work_year + work_month без inflation_indices → 400."""
        payload = {**valid_payload, "work_year": 2026, "work_month": 4}
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 400

    def test_negative_area_returns_400(self, client, auth_headers, valid_payload):
        """Від'ємна площа (area < 0) → 400."""
        payload = {**valid_payload, "area": -10.0}
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 400

    def test_negative_floors_returns_400(self, client, auth_headers, valid_payload):
        """Від'ємна кількість поверхів (floors < 0) → 400."""
        payload = {**valid_payload, "floors": -1}
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 400


# ===========================================================================
# TestPredict
# ===========================================================================
@requires_model
class TestPredict:
    def test_basic_predict_returns_positive_cost(self, client, auth_headers, valid_payload):
        """Базовий прогноз без інфляційного коригування: predicted_cost > 0."""
        resp = client.post("/predict", json=valid_payload, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "predicted_cost" in data
        assert data["predicted_cost"] > 0

    def test_basic_predict_response_has_required_fields(
        self, client, auth_headers, valid_payload
    ):
        """Відповідь /predict містить всі обов'язкові поля."""
        resp = client.post("/predict", json=valid_payload, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        required_fields = {
            "predicted_cost",
            "adjusted_cost",
            "inflation_k",
            "base_year",
            "base_month",
            "work_year",
            "work_month",
            "currency",
            "model",
        }
        assert required_fields.issubset(data.keys()), (
            f"Відсутні поля: {required_fields - data.keys()}"
        )

    def test_basic_predict_currency_is_uah(self, client, auth_headers, valid_payload):
        """Валюта у відповіді має бути UAH."""
        resp = client.post("/predict", json=valid_payload, headers=auth_headers)
        assert resp.get_json()["currency"] == "UAH"

    def test_predict_without_inflation_adjusted_cost_equals_predicted(
        self, client, auth_headers, valid_payload
    ):
        """Без інфляційного коригування adjusted_cost == predicted_cost."""
        resp = client.post("/predict", json=valid_payload, headers=auth_headers)
        data = resp.get_json()
        assert data["adjusted_cost"] == data["predicted_cost"]
        assert data["inflation_k"] == 1.0

    def test_predict_with_inflation_returns_adjusted_cost(
        self, client, auth_headers, valid_payload_with_inflation
    ):
        """Прогноз з інфляцією > 1: adjusted_cost >= predicted_cost."""
        resp = client.post(
            "/predict", json=valid_payload_with_inflation, headers=auth_headers
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "adjusted_cost" in data
        # index_value=1.24 > 1 → скоригована вартість більша або рівна базовій
        assert data["adjusted_cost"] >= data["predicted_cost"]

    def test_predict_with_inflation_coefficient_matches_payload(
        self, client, auth_headers, valid_payload_with_inflation
    ):
        """Коефіцієнт інфляції у відповіді відповідає переданому inflation_indices."""
        resp = client.post(
            "/predict", json=valid_payload_with_inflation, headers=auth_headers
        )
        data = resp.get_json()
        assert abs(data["inflation_k"] - 1.24) < 1e-6
        assert data["work_year"] == 2026
        assert data["work_month"] == 4

    def test_predict_minimal_values_returns_200(self, client, auth_headers):
        """Мінімальні значення (area=1, floors=1) → 200, predicted_cost > 0."""
        payload = {
            "area": 1.0,
            "floors": 1,
            "building_type": "Інше",
            "damage_level": "Легке",
            "region": "м. Київ",
            "repair_type": "Поточний",
        }
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.get_json()["predicted_cost"] > 0

    def test_predict_large_values_returns_200(self, client, auth_headers):
        """Великі значення (area=100000, floors=50) → 200, predicted_cost > 0."""
        payload = {
            "area": 100000.0,
            "floors": 50,
            "building_type": "Адміністративна будівля",
            "damage_level": "Тяжке",
            "region": "Київська обл.",
            "repair_type": "Повна реконструкція",
        }
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.get_json()["predicted_cost"] > 0


# ===========================================================================
# TestPredictExplain
# ===========================================================================
@requires_model
class TestPredictExplain:
    def test_predict_explain_returns_200(self, client, auth_headers, valid_payload):
        """/predict_explain повертає 200."""
        resp = client.post("/predict_explain", json=valid_payload, headers=auth_headers)
        assert resp.status_code == 200

    def test_predict_explain_has_contributions_field(
        self, client, auth_headers, valid_payload
    ):
        """Відповідь /predict_explain містить поле contributions."""
        resp = client.post("/predict_explain", json=valid_payload, headers=auth_headers)
        data = resp.get_json()
        assert "contributions" in data
        assert isinstance(data["contributions"], list)
        assert len(data["contributions"]) > 0

    def test_predict_explain_contributions_have_correct_structure(
        self, client, auth_headers, valid_payload
    ):
        """Кожен елемент contributions має поля group, contribution, percent."""
        resp = client.post("/predict_explain", json=valid_payload, headers=auth_headers)
        data = resp.get_json()
        for item in data["contributions"]:
            assert "group" in item, f"Відсутнє поле 'group' у {item}"
            assert "contribution" in item, f"Відсутнє поле 'contribution' у {item}"
            assert "percent" in item, f"Відсутнє поле 'percent' у {item}"

    def test_predict_explain_percentages_sum_to_100(
        self, client, auth_headers, valid_payload
    ):
        """Сума percent у contributions ≈ 100% (з допуском ±0.1)."""
        resp = client.post("/predict_explain", json=valid_payload, headers=auth_headers)
        data = resp.get_json()
        total_pct = sum(item["percent"] for item in data["contributions"])
        assert abs(total_pct - 100.0) < 0.1, (
            f"Сума відсотків {total_pct:.4f}% ≠ 100%"
        )

    def test_predict_explain_shap_decomposition_property(
        self, client, auth_headers, valid_payload
    ):
        """
        Базова властивість SHAP: base_value + Σ(contributions) ≈ predicted_cost.
        Перевіряємо лише якщо base_value не None (SHAP доступний, не fallback).
        Допуск: 1% від predicted_cost.
        """
        resp = client.post("/predict_explain", json=valid_payload, headers=auth_headers)
        data = resp.get_json()

        if data.get("base_value") is None:
            pytest.skip("SHAP недоступний, використовується fallback (gain). Пропускаємо.")

        base_value = data["base_value"]
        shap_sum = sum(
            item["contribution"]
            for item in data["contributions"]
            if item["contribution"] is not None
        )
        predicted = data["predicted_cost"]
        reconstructed = base_value + shap_sum

        tolerance = predicted * 0.01  # 1% допуск
        assert abs(reconstructed - predicted) < tolerance, (
            f"SHAP декомпозиція: base_value({base_value:.2f}) + "
            f"Σcontributions({shap_sum:.2f}) = {reconstructed:.2f}, "
            f"але predicted_cost = {predicted:.2f} (різниця > 1%)"
        )

    def test_predict_explain_has_base_value_field(
        self, client, auth_headers, valid_payload
    ):
        """Відповідь /predict_explain містить поле base_value (може бути None при fallback)."""
        resp = client.post("/predict_explain", json=valid_payload, headers=auth_headers)
        data = resp.get_json()
        assert "base_value" in data

    def test_predict_explain_predicted_cost_matches_predict(
        self, client, auth_headers, valid_payload
    ):
        """predicted_cost у /predict_explain збігається з /predict для того самого payload."""
        resp_predict = client.post("/predict", json=valid_payload, headers=auth_headers)
        resp_explain = client.post("/predict_explain", json=valid_payload, headers=auth_headers)

        cost_predict = resp_predict.get_json()["predicted_cost"]
        cost_explain = resp_explain.get_json()["predicted_cost"]

        assert cost_predict == cost_explain, (
            f"/predict: {cost_predict}, /predict_explain: {cost_explain}"
        )

    def test_predict_explain_with_inflation(
        self, client, auth_headers, valid_payload_with_inflation
    ):
        """/predict_explain з інфляцією повертає 200 і contributions."""
        resp = client.post(
            "/predict_explain", json=valid_payload_with_inflation, headers=auth_headers
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["contributions"]) > 0
        assert data["adjusted_cost"] >= data["predicted_cost"]


# ===========================================================================
# TestHealth
# ===========================================================================
class TestHealth:
    """Тести /health не потребують моделі."""

    def test_health_returns_200(self, client):
        """/health повертає 200."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok_status(self, client):
        """/health повертає {"status": "ok"}."""
        resp = client.get("/health")
        data = resp.get_json()
        assert data == {"status": "ok"}
