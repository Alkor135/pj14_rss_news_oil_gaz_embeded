"""Тесты внешнего цикла адаптивного walk-forward."""

import unittest

import pandas as pd

from research.adaptive_regime_walk_forward import (
    REGIME_COUNTERTREND,
    REGIME_MOMENTUM,
    WalkForwardConfig,
    generate_candidates,
    run_walk_forward,
)


class WalkForwardTests(unittest.TestCase):
    """Проверки вложенного ежедневного walk-forward."""

    def setUp(self):
        """Создаёт компактную конфигурацию для синтетических историй."""
        self.config = WalkForwardConfig(
            train_window_days=6,
            score_spans=(1, 2),
            confidence_thresholds=(0.0, 0.2),
            min_regime_days=(1, 2),
            drawdown_penalty=1.0,
        )

    def test_generate_candidates_is_deterministic(self):
        """Сетка кандидатов должна иметь стабильный порядок."""
        first = generate_candidates(self.config)
        second = generate_candidates(self.config)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)

    def test_current_outcome_does_not_change_current_decision(self):
        """P/L тестируемого дня не должен влиять на его режим и score."""
        source = pd.DataFrame(
            {
                "TRADEDATE": pd.date_range("2026-01-01", periods=12),
                "P/L": [-1.0] * 12,
                "max": [5] * 12,
            }
        )
        changed_source = source.copy()
        changed_source.loc[11, "P/L"] = 1000.0

        original = run_walk_forward(source, self.config)
        changed = run_walk_forward(changed_source, self.config)

        self.assertEqual(
            original.iloc[-1]["applied_mode"],
            changed.iloc[-1]["applied_mode"],
        )
        self.assertEqual(
            original.iloc[-1]["regime_score"],
            changed.iloc[-1]["regime_score"],
        )

    def test_walk_forward_changes_from_countertrend_to_momentum(self):
        """После смены знака устойчивого P/L режим должен адаптироваться."""
        source = pd.DataFrame(
            {
                "TRADEDATE": pd.date_range("2026-01-01", periods=30),
                "P/L": [1.0] * 15 + [-1.0] * 15,
                "max": [5] * 30,
            }
        )

        result = run_walk_forward(source, self.config)

        self.assertIn(REGIME_COUNTERTREND, result["applied_mode"].iloc[:10].tolist())
        self.assertIn(REGIME_MOMENTUM, result["applied_mode"].iloc[-10:].tolist())

    def test_first_training_window_is_excluded(self):
        """Результат должен начинаться с первого дня после полного обучения."""
        source = pd.DataFrame(
            {
                "TRADEDATE": pd.date_range("2026-01-01", periods=10),
                "P/L": [1.0] * 10,
                "max": [5] * 10,
            }
        )

        result = run_walk_forward(source, self.config)

        self.assertEqual(len(result), 4)
        self.assertEqual(result.iloc[0]["TRADEDATE"], source.iloc[6]["TRADEDATE"])


if __name__ == "__main__":
    unittest.main()
