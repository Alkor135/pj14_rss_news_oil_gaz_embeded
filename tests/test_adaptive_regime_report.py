"""Тесты отчётов адаптивного режимного walk-forward."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from research.adaptive_regime_report import (
    build_summary,
    write_global_summary,
    write_html_report,
    write_results_workbook,
)


class ReportTests(unittest.TestCase):
    """Проверки метрик и исследовательских артефактов."""

    def setUp(self):
        """Создаёт компактный пример дневных результатов."""
        self.daily = pd.DataFrame(
            {
                "TRADEDATE": pd.date_range("2026-01-01", periods=4),
                "max": [5, 5, 6, 6],
                "countertrend_pl": [2.0, -1.0, 3.0, -2.0],
                "momentum_pl": [-2.0, 1.0, -3.0, 2.0],
                "proposed_mode": [
                    "countertrend",
                    "flat",
                    "countertrend",
                    "momentum",
                ],
                "applied_mode": [
                    "countertrend",
                    "flat",
                    "countertrend",
                    "momentum",
                ],
                "regime_score": [-0.4, 0.0, -0.5, 0.6],
                "score_span": [5, pd.NA, 5, 8],
                "confidence_threshold": [0.2, pd.NA, 0.2, 0.2],
                "min_regime_days": [1, pd.NA, 1, 1],
                "train_objective": [2.0, 0.0, 3.0, 2.0],
                "train_total_pl": [3.0, 0.0, 4.0, 3.0],
                "train_max_drawdown": [1.0, 0.0, 1.0, 1.0],
                "train_switches": [1, 0, 1, 2],
                "strategy_pl": [2.0, 0.0, 3.0, 2.0],
                "cumulative_pl": [2.0, 2.0, 5.0, 7.0],
                "drawdown": [0.0, 0.0, 0.0, 0.0],
            }
        )

    def test_summary_contains_strategy_and_benchmarks(self):
        """Сводка должна сопоставлять три стратегии на одном периоде."""
        summary = build_summary(self.daily, "RTS")

        self.assertEqual(summary.loc["adaptive", "total_pl"], 7.0)
        self.assertIn("countertrend", summary.index)
        self.assertIn("momentum", summary.index)
        self.assertAlmostEqual(summary.loc["adaptive", "mode_share_flat"], 0.25)

    def test_writers_create_expected_artifacts(self):
        """Excel и HTML должны иметь стабильные имена и структуру."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = build_summary(self.daily, "RTS")

            workbook = write_results_workbook(
                self.daily,
                summary,
                {"train_window_days": 63},
                output,
            )
            html = write_html_report(self.daily, summary, "RTS", output)

            with pd.ExcelFile(workbook) as excel_file:
                sheet_names = excel_file.sheet_names
            self.assertEqual(sheet_names, ["daily", "summary", "config"])
            self.assertIn("RTS", html.read_text(encoding="utf-8"))

    def test_global_summary_uses_fixed_columns(self):
        """Общая книга должна содержать статусы всех тикеров."""
        rows = [
            {
                "ticker": "rts",
                "status": "success",
                "message": "ok",
                "total_pl": 7.0,
                "max_drawdown": 2.0,
                "output_dir": "rts/research",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.xlsx"

            result = write_global_summary(rows, path)
            loaded = pd.read_excel(result)

            self.assertEqual(
                loaded.columns.tolist(),
                [
                    "ticker",
                    "status",
                    "message",
                    "total_pl",
                    "max_drawdown",
                    "output_dir",
                ],
            )


if __name__ == "__main__":
    unittest.main()
