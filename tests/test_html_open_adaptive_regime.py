"""Тесты открытия адаптивных HTML-отчётов."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock

from html_open_adaptive_regime import (
    TICKERS,
    collect_report_paths,
    launch_reports,
    main,
)


class AdaptiveHtmlOpenTests(unittest.TestCase):
    """Проверки поиска отчётов и единственного запуска Chrome."""

    def test_collect_report_paths_preserves_ticker_order(self):
        """Существующие и отсутствующие пути должны идти в порядке TICKERS."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for ticker in ("rts", "ng"):
                path = root / ticker / "research" / "adaptive_regime_walk_forward"
                path.mkdir(parents=True)
                (path / "report.html").write_text(
                    "<html></html>", encoding="utf-8"
                )

            found, missing = collect_report_paths(root)

            self.assertEqual([path.parts[-4] for path in found], ["rts", "ng"])
            self.assertEqual(
                [path.parts[-4] for path in missing],
                [ticker for ticker in TICKERS if ticker not in {"rts", "ng"}],
            )

    def test_launch_reports_uses_one_new_chrome_window(self):
        """Все отчёты должны передаваться одному вызову Chrome."""
        launcher = Mock()
        chrome = Path("C:/Chrome/chrome.exe")
        reports = [Path("C:/reports/rts.html"), Path("C:/reports/mix.html")]

        launch_reports(chrome, reports, launcher)

        launcher.assert_called_once_with(
            [str(chrome), "--new-window", str(reports[0]), str(reports[1])]
        )

    def test_main_returns_one_without_chrome(self):
        """При отсутствующем Chrome запуск процесса запрещён."""
        with tempfile.TemporaryDirectory() as directory:
            launcher = Mock()

            with redirect_stdout(io.StringIO()):
                code = main(Path(directory), Path(directory) / "chrome.exe", launcher)

            self.assertEqual(code, 1)
            launcher.assert_not_called()

    def test_main_opens_available_reports_and_returns_zero(self):
        """Наличие Chrome и одного отчёта достаточно для успешного запуска."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chrome = root / "chrome.exe"
            chrome.touch()
            report = (
                root
                / "rts"
                / "research"
                / "adaptive_regime_walk_forward"
                / "report.html"
            )
            report.parent.mkdir(parents=True)
            report.write_text("<html></html>", encoding="utf-8")
            launcher = Mock()

            with redirect_stdout(io.StringIO()):
                code = main(root, chrome, launcher)

            self.assertEqual(code, 0)
            launcher.assert_called_once()


if __name__ == "__main__":
    unittest.main()
