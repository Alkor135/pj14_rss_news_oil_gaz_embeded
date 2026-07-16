"""Тесты настроек и входных данных адаптивного walk-forward."""

import unittest

import pandas as pd

from research.adaptive_regime_walk_forward import (
    RegimeDataError,
    load_walk_forward_config,
    validate_source_results,
)


class InputValidationTests(unittest.TestCase):
    """Проверки настроек и входного DataFrame."""

    def test_config_rejects_span_not_smaller_than_training_window(self):
        """Период индикатора обязан помещаться в обучающее окно."""
        settings = {
            "regime_walk_forward": {
                "train_window_days": 5,
                "score_spans": [5],
                "confidence_thresholds": [0.1],
                "min_regime_days": [1],
                "drawdown_penalty": 1.0,
            }
        }

        with self.assertRaisesRegex(RegimeDataError, "score_spans"):
            load_walk_forward_config(settings)

    def test_validation_sorts_dates(self):
        """Корректные уникальные даты должны сортироваться по возрастанию."""
        source = pd.DataFrame(
            {
                "TRADEDATE": ["2026-01-02", "2026-01-01"],
                "P/L": [1.0, 2.0],
                "max": [5, 6],
            }
        )

        validated = validate_source_results(source, train_window_days=1)

        self.assertTrue(validated["TRADEDATE"].is_monotonic_increasing)

    def test_validation_rejects_duplicate_dates(self):
        """Повторяющиеся торговые даты нельзя исправлять скрытно."""
        source = pd.DataFrame(
            {
                "TRADEDATE": ["2026-01-01", "2026-01-01"],
                "P/L": [1.0, 2.0],
                "max": [5, 6],
            }
        )

        with self.assertRaisesRegex(RegimeDataError, "повтор"):
            validate_source_results(source, train_window_days=1)

    def test_validation_requires_one_oos_day(self):
        """После обучения должен оставаться минимум один тестовый день."""
        source = pd.DataFrame(
            {"TRADEDATE": ["2026-01-01"], "P/L": [1.0], "max": [5]}
        )

        with self.assertRaisesRegex(RegimeDataError, "недостаточно"):
            validate_source_results(source, train_window_days=1)


if __name__ == "__main__":
    unittest.main()
