"""Тесты причинной симуляции режимного кандидата."""

import unittest

import pandas as pd

from research.adaptive_regime_walk_forward import (
    REGIME_MOMENTUM,
    RegimeParameters,
    build_payoff_columns,
    compute_causal_scores,
    simulate_candidate,
)


class CandidateSimulationTests(unittest.TestCase):
    """Проверки причинной симуляции кандидата."""

    def test_score_does_not_use_current_day(self):
        """Результат текущего дня не должен попадать в его индикатор."""
        original = compute_causal_scores(pd.Series([2.0, 2.0, -100.0]), span=2)
        changed = compute_causal_scores(pd.Series([2.0, 2.0, 100.0]), span=2)

        self.assertEqual(original.iloc[2], changed.iloc[2])

    def test_candidate_holds_active_regime_for_minimum_days(self):
        """После входа режим должен удерживаться заданное число дней."""
        frame = build_payoff_columns(
            pd.DataFrame({"P/L": [-1.0, -1.0, 10.0, 10.0, 10.0]})
        )
        result = simulate_candidate(
            frame,
            RegimeParameters(1, 0.0, 3),
            drawdown_penalty=1.0,
        )

        self.assertEqual(
            result.daily.loc[1:3, "applied_mode"].tolist(),
            [REGIME_MOMENTUM, REGIME_MOMENTUM, REGIME_MOMENTUM],
        )

    def test_candidate_daily_pl_matches_applied_mode(self):
        """Активный P/L должен соответствовать фактически применённому режиму."""
        frame = build_payoff_columns(pd.DataFrame({"P/L": [-2.0, -2.0, -2.0]}))
        result = simulate_candidate(
            frame,
            RegimeParameters(1, 0.0, 1),
            drawdown_penalty=0.0,
        )
        active = result.daily[
            result.daily["applied_mode"] == REGIME_MOMENTUM
        ]

        self.assertTrue((active["strategy_pl"] == active["momentum_pl"]).all())


if __name__ == "__main__":
    unittest.main()
