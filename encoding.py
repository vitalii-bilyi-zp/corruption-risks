"""
encoding.py - Централізований модуль кодування ознак для прогнозування вартості відновлення.

Надає:
- Fuzzy-кодування для порядкових категоріальних ознак (damage_level, repair_type)
- One-hot кодування для номінальних ознак (building_type, region)
- Зворотна сумісність з моделями v1.0 (чистий one-hot)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import json


# ============== Конфігурація ==============

@dataclass
class OrdinalFeatureConfig:
    """Конфігурація для fuzzy-кодування порядкової ознаки."""
    name: str
    categories: List[str]  # Впорядковані від низького до високого
    positions: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.positions:
            n = len(self.categories)
            self.positions = [i / (n - 1) for i in range(n)] if n > 1 else [0.5]


@dataclass
class FuzzyEncodingConfig:
    """Повна конфігурація системи кодування."""
    version: str = "2.0"
    ordinal_features: Dict[str, OrdinalFeatureConfig] = field(default_factory=dict)
    nominal_features: List[str] = field(default_factory=list)

    @classmethod
    def default(cls) -> "FuzzyEncodingConfig":
        """Конфігурація за замовчуванням для моделі вартості відновлення."""
        return cls(
            ordinal_features={
                "damage_level": OrdinalFeatureConfig(
                    name="damage_level",
                    categories=["Легке", "Середнє", "Тяжке"]
                ),
                "repair_type": OrdinalFeatureConfig(
                    name="repair_type",
                    categories=["Поточний", "Капітальний", "Повна реконструкція"]
                )
            },
            nominal_features=["building_type", "region"]
        )


# ============== Трикутні функції належності ==============

def triangular_mf(x: float, a: float, b: float, c: float) -> float:
    """
    Трикутна функція належності.

    Args:
        x: Вхідне значення
        a: Ліва межа (μ=0)
        b: Вершина (μ=1)
        c: Права межа (μ=0)

    Returns:
        Ступінь належності в [0, 1]
    """
    if x <= a or x >= c:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    else:
        return (c - x) / (c - b) if c != b else 1.0


def compute_fuzzy_params(positions: List[float]) -> List[Tuple[float, float, float]]:
    """
    Обчислює параметри трикутних функцій належності для впорядкованих категорій.

    Args:
        positions: Числові позиції категорій на [0, 1]

    Returns:
        Список кортежів (a, b, c) для кожної нечіткої множини
    """
    params = []
    n = len(positions)

    for i, center in enumerate(positions):
        if n == 1:
            params.append((center - 0.5, center, center + 0.5))
        elif i == 0:
            # Перша категорія: розширюємо ліву межу
            right_dist = positions[1] - center
            params.append((center - right_dist, center, positions[1]))
        elif i == n - 1:
            # Остання категорія: розширюємо праву межу
            left_dist = center - positions[i - 1]
            params.append((positions[i - 1], center, center + left_dist))
        else:
            # Середні категорії
            params.append((positions[i - 1], center, positions[i + 1]))

    return params


# ============== Класи кодування ==============

class FuzzyEncoder:
    """Кодувальник для порядкових ознак з використанням fuzzy-множин."""

    def __init__(self, config: OrdinalFeatureConfig):
        self.config = config
        self.category_to_position = {
            cat: pos for cat, pos in zip(config.categories, config.positions)
        }
        self.fuzzy_params = compute_fuzzy_params(config.positions)
        self.output_columns = [f"{config.name}_{cat}" for cat in config.categories]

    def encode_value(self, category: str) -> np.ndarray:
        """Кодує одне категоріальне значення у вектор ступенів належності."""
        if category not in self.category_to_position:
            raise ValueError(
                f"Невідома категорія '{category}' для ознаки '{self.config.name}'. "
                f"Допустимі: {self.config.categories}"
            )

        x = self.category_to_position[category]
        memberships = [triangular_mf(x, *params) for params in self.fuzzy_params]
        return np.array(memberships)

    def encode_series(self, series: pd.Series) -> pd.DataFrame:
        """Кодує pandas Series у DataFrame з fuzzy-колонками."""
        encoded = np.vstack([self.encode_value(v) for v in series])
        return pd.DataFrame(encoded, columns=self.output_columns, index=series.index)


class FeatureEncoder:
    """
    Повний пайплайн кодування ознак з підтримкою fuzzy (порядкові)
    та one-hot (номінальні) кодування.
    """

    def __init__(self, config: Optional[FuzzyEncodingConfig] = None):
        self.config = config or FuzzyEncodingConfig.default()
        self.fuzzy_encoders = {
            name: FuzzyEncoder(feat_config)
            for name, feat_config in self.config.ordinal_features.items()
        }
        self._fitted_columns: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        """Підганяє кодувальник до навчальних даних (запам'ятовує порядок колонок)."""
        encoded = self._transform_internal(df)
        self._fitted_columns = list(encoded.columns)
        return self

    def _transform_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Внутрішня трансформація без вирівнювання по fitted_columns."""
        result_parts = []

        # 1. Числові колонки
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in ("restoration_cost",):
                result_parts.append(df[[col]])

        # 2. Fuzzy-кодування для порядкових ознак
        for feat_name, encoder in self.fuzzy_encoders.items():
            if feat_name in df.columns:
                result_parts.append(encoder.encode_series(df[feat_name]))

        # 3. One-hot кодування для номінальних ознак
        nominal_present = [c for c in self.config.nominal_features if c in df.columns]
        if nominal_present:
            dummies = pd.get_dummies(df[nominal_present], columns=nominal_present)
            result_parts.append(dummies)

        return pd.concat(result_parts, axis=1) if result_parts else pd.DataFrame()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Трансформує DataFrame з вирівнюванням по навчених колонках."""
        result = self._transform_internal(df)

        if self._fitted_columns is not None:
            for col in self._fitted_columns:
                if col not in result.columns:
                    result[col] = 0.0
            result = result[self._fitted_columns]

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Підгонка та трансформація за один крок."""
        return self.fit(df).transform(df)

    def get_feature_columns(self) -> List[str]:
        """Повертає список імен колонок після кодування."""
        return self._fitted_columns.copy() if self._fitted_columns else []

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Повертає маппінг оригінальних ознак на закодовані колонки."""
        groups = {"version": self.config.version}

        # Числові ознаки
        groups["area"] = ["area"]
        groups["floors"] = ["floors"]

        # Порядкові ознаки (fuzzy)
        for name, encoder in self.fuzzy_encoders.items():
            groups[name] = encoder.output_columns.copy()

        # Номінальні ознаки (one-hot)
        if self._fitted_columns:
            for feat in self.config.nominal_features:
                prefix = f"{feat}_"
                groups[feat] = [c for c in self._fitted_columns if c.startswith(prefix)]

        return groups

    def save_metadata(self, columns_path: str, groups_path: str):
        """Зберігає колонки та групи у JSON файли."""
        with open(columns_path, "w", encoding="utf-8") as f:
            json.dump(self.get_feature_columns(), f, ensure_ascii=False, indent=4)

        with open(groups_path, "w", encoding="utf-8") as f:
            json.dump(self.get_feature_groups(), f, ensure_ascii=False, indent=4)


# ============== Legacy One-Hot кодування (v1.0) ==============

def legacy_onehot_preprocess(data_dict: dict, feature_columns: List[str]) -> pd.DataFrame:
    """
    Legacy-препроцесинг з one-hot кодуванням (для сумісності з v1.0 моделями).
    """
    categorical = ["building_type", "damage_level", "region", "repair_type"]
    df = pd.DataFrame([data_dict])
    df = pd.get_dummies(df, columns=categorical)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]


# ============== Зручні функції ==============

def create_default_encoder() -> FeatureEncoder:
    """Створює кодувальник з конфігурацією за замовчуванням."""
    return FeatureEncoder(FuzzyEncodingConfig.default())


def preprocess_for_training(df: pd.DataFrame) -> Tuple[pd.DataFrame, FeatureEncoder]:
    """
    Препроцесинг навчальних даних.

    Returns:
        (X_encoded, encoder) - закодовані ознаки та навчений кодувальник
    """
    encoder = create_default_encoder()
    X = encoder.fit_transform(df)
    return X, encoder


def preprocess_for_inference(
    data: dict,
    feature_columns: List[str],
    feature_groups: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Препроцесинг для інференсу з автовизначенням версії кодування.

    Args:
        data: Вхідний словник з сирими значеннями ознак
        feature_columns: Очікуваний порядок колонок з навчання
        feature_groups: Метадані груп (опціонально, для визначення версії)

    Returns:
        DataFrame, готовий для model.predict()
    """
    version = "1.0"
    if feature_groups:
        version = feature_groups.get("version", "1.0")

    if version == "1.0":
        return legacy_onehot_preprocess(data, feature_columns)

    # v2.0: fuzzy encoding
    encoder = FeatureEncoder(FuzzyEncodingConfig.default())
    encoder._fitted_columns = feature_columns
    df = pd.DataFrame([data])
    return encoder.transform(df)
