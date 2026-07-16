"""Тесты адаптивного режимного walk-forward бэктеста."""

import unittest

import pandas as pd

from research.adaptive_regime_walk_forward import (
    REGIME_COUNTERTREND,
    REGIME_FLAT,
    REGIME_MOMENTUM,
    build_payoff_columns,
    calculate_max_drawdown,
    calculate_objective,
    classify_score,
)


class BasicCalculationsTests(unittest.TestCase):
    """Проверки базовых расчётов режимной стратегии."""

    def test_build_payoff_columns_uses_opposite_signs(self):
        """Импульсный P/L должен быть зеркалом контртрендового."""
        source = pd.DataFrame({"P/L": [10.0, -4.0]})

        result = build_payoff_columns(source)

        self.assertEqual(result["countertrend_pl"].tolist(), [10.0, -4.0])
        self.assertEqual(result["momentum_pl"].tolist(), [-10.0, 4.0])
        self.assertEqual(result["advantage"].tolist(), [-20.0, 8.0])

    def test_classify_score_has_neutral_zone(self):
        """Значения внутри порога должны приводить к пропуску сделки."""
        self.assertEqual(classify_score(0.21, 0.20), REGIME_MOMENTUM)
        self.assertEqual(classify_score(-0.21, 0.20), REGIME_COUNTERTREND)
        self.assertEqual(classify_score(0.20, 0.20), REGIME_FLAT)

    def test_max_drawdown_is_positive_magnitude(self):
        """Просадка должна измеряться положительной глубиной от пика."""
        result = calculate_max_drawdown(pd.Series([5.0, -2.0, -4.0, 3.0]))

        self.assertEqual(result, 6.0)

    def test_objective_penalizes_drawdown(self):
        """Целевая функция должна штрафовать максимальную просадку."""
        self.assertEqual(calculate_objective(12.0, 5.0, 1.5), 4.5)


if __name__ == "__main__":
    unittest.main()
