"""Тесты общего запуска адаптивного walk-forward."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import pandas as pd
import yaml

from run_adaptive_regime_walk_forward import (
    exit_code_for,
    parse_args,
    run_many,
)


class RunnerTests(unittest.TestCase):
    """Проверки общего запускного сценария."""

    def test_partial_success_does_not_stop_other_tickers(self):
        """Отсутствующий Excel одного тикера не должен прерывать остальные."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "rts"
            missing = root / "mix"
            valid.mkdir()
            missing.mkdir()
            settings = {
                "ticker": "RTS",
                "regime_walk_forward": {
                    "train_window_days": 2,
                    "score_spans": [1],
                    "confidence_thresholds": [0.0],
                    "min_regime_days": [1],
                    "drawdown_penalty": 0.0,
                },
            }
            for path in (valid, missing):
                (path / "settings.yaml").write_text(
                    yaml.safe_dump(settings, allow_unicode=True),
                    encoding="utf-8",
                )
            pd.DataFrame(
                {
                    "TRADEDATE": pd.date_range("2026-01-01", periods=5),
                    "P/L": [1.0, 1.0, -1.0, -1.0, -1.0],
                    "max": [5] * 5,
                }
            ).to_excel(valid / "df_rez_output.xlsx", index=False)

            statuses = run_many(("rts", "mix"), root)

            self.assertEqual(statuses[0].status, "success")
            self.assertEqual(statuses[1].status, "skipped")
            self.assertEqual(exit_code_for(statuses), 0)
            self.assertEqual(exit_code_for([statuses[1]]), 1)

    def test_ticker_argument_is_case_insensitive(self):
        """Имя тикера в CLI должно приниматься без учёта регистра."""
        self.assertEqual(parse_args(["--ticker", "RTS"]).ticker, "rts")

    def test_ticker_and_all_are_mutually_exclusive(self):
        """Нельзя одновременно запросить один тикер и полный запуск."""
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--ticker", "rts", "--all"])


if __name__ == "__main__":
    unittest.main()
