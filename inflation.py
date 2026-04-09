"""
inflation.py — коефіцієнти інфляційного коригування вартості відновлення.

Логіка:
  BASE_YEAR = 2024, BASE_MONTH = 1 — базовий період навчання ML-моделі.
  Прогноз V_base відповідає цінам 2024-01. Скоригована вартість:
  V(y, m) = V_base * k(y, m).

  Коефіцієнти передаються Laravel-ом з БД у полі inflation_indices кожного запиту.
  Якщо коефіцієнт для запитаної пари (year, month) не знайдено — викидається
  ValueError, що конвертується у HTTP 400.
"""

from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)

BASE_YEAR: int = 2024
BASE_MONTH: int = 1


class InflationService:
    """
    Сервіс інфляційного коригування вартості відновлення.

    Коефіцієнти отримуються виключно з payload (передаються Laravel з БД).
    Якщо коефіцієнт для запитаної пари (year, month) не знайдено —
    викидається ValueError, що конвертується у HTTP 400.
    """

    def get_coefficient(
        self,
        work_year: Optional[int],
        work_month: Optional[int],
        indices: Optional[list] = None,
    ) -> float:
        """
        Повертає коефіцієнт k для пари (work_year, work_month).

        Raises:
            ValueError: якщо work_year/work_month передано, але коефіцієнт
                        не знайдено у indices.
        """
        if work_year is None and work_month is None:
            return 1.0

        if not indices:
            raise ValueError(
                f"Не передано таблицю інфляційних коефіцієнтів (inflation_indices). "
                f"Неможливо розрахувати коригування для {work_year}-{work_month:02d}."
            )

        for row in indices:
            y = row['year']        if isinstance(row, dict) else row.year
            m = row['month']       if isinstance(row, dict) else row.month
            v = row['index_value'] if isinstance(row, dict) else row.index_value
            if y == work_year and m == work_month:
                logger.info(f"Inflation k={v} for {work_year}-{work_month:02d}")
                return float(v)

        raise ValueError(
            f"Коефіцієнт інфляції для {work_year}-{work_month:02d} відсутній "
            f"у переданій таблиці. Оновіть таблицю inflation_indices у системі."
        )

    def apply(
        self,
        base_cost: float,
        work_year: Optional[int],
        work_month: Optional[int],
        indices: Optional[list] = None,
    ) -> dict:
        """
        Застосовує коефіцієнт до базового прогнозу.

        Raises:
            ValueError: пробрасується з get_coefficient якщо коефіцієнт не знайдено.
        """
        k = self.get_coefficient(work_year, work_month, indices)
        return {
            "adjusted_cost": round(base_cost * k, 2),
            "inflation_k":   k,
            "base_year":     BASE_YEAR,
            "base_month":    BASE_MONTH,
            "work_year":     work_year  if work_year  is not None else BASE_YEAR,
            "work_month":    work_month if work_month is not None else BASE_MONTH,
        }
