"""Тесты правил Git для локальных walk-forward отчётов."""

import subprocess
import unittest
from pathlib import Path


class GeneratedFilesIgnoreTests(unittest.TestCase):
    """Проверки исключения тяжёлых локальных артефактов из Git."""

    def test_generated_research_reports_are_ignored(self):
        """Общая книга и тикерные отчёты не должны попадать в коммиты."""
        root = Path(__file__).resolve().parents[1]
        paths = (
            "research/adaptive_regime_walk_forward_summary.xlsx",
            "rts/research/adaptive_regime_walk_forward/results.xlsx",
            "rts/research/adaptive_regime_walk_forward/report.html",
        )
        for path in paths:
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "check-ignore", "--quiet", path],
                    cwd=root,
                    check=False,
                )
                self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
