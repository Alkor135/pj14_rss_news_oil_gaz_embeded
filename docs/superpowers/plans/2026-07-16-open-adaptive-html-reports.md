# Open Adaptive HTML Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить один Python-скрипт, открывающий семь локальных walk-forward `report.html` в одном новом окне Chrome.

**Architecture:** Корневой сценарий строит пути относительно собственного файла, разделяет найденные и отсутствующие отчёты и выполняет один `subprocess.Popen`. Функция запуска процесса передаётся как зависимость, поэтому `unittest` проверяет аргументы без открытия реального Chrome.

**Tech Stack:** Python 3, pathlib, subprocess, стандартный unittest.

---

## Карта файлов

- Create: `html_open_adaptive_regime.py` — сбор путей, диагностика и открытие Chrome.
- Create: `tests/test_html_open_adaptive_regime.py` — проверки путей и запуска без GUI.

### Task 1: Тесты поведения и реализация сценария

**Files:**
- Create: `tests/test_html_open_adaptive_regime.py`
- Create: `html_open_adaptive_regime.py`

- [ ] **Step 1: Написать падающие тесты до создания сценария**

```python
"""Тесты открытия адаптивных HTML-отчётов."""

import tempfile
import unittest
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
                (path / "report.html").write_text("<html></html>", encoding="utf-8")

            found, missing = collect_report_paths(root)

            self.assertEqual([path.parts[-4] for path in found], ["rts", "ng"])
            self.assertEqual([path.parts[-4] for path in missing], [
                ticker for ticker in TICKERS if ticker not in {"rts", "ng"}
            ])

    def test_launch_reports_uses_one_new_chrome_window(self):
        """Все отчёты должны передаваться одному вызову Chrome."""
        launcher = Mock()
        chrome = Path("C:/Chrome/chrome.exe")
        reports = [Path("C:/reports/rts.html"), Path("C:/reports/mix.html")]

        launch_reports(chrome, reports, launcher)

        launcher.assert_called_once_with([
            str(chrome), "--new-window", str(reports[0]), str(reports[1])
        ])

    def test_main_returns_one_without_chrome(self):
        """При отсутствующем Chrome запуск процесса запрещён."""
        with tempfile.TemporaryDirectory() as directory:
            launcher = Mock()

            code = main(Path(directory), Path(directory) / "chrome.exe", launcher)

            self.assertEqual(code, 1)
            launcher.assert_not_called()

    def test_main_opens_available_reports_and_returns_zero(self):
        """Наличие Chrome и одного отчёта достаточно для успешного запуска."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chrome = root / "chrome.exe"
            chrome.touch()
            report = root / "rts" / "research" / "adaptive_regime_walk_forward" / "report.html"
            report.parent.mkdir(parents=True)
            report.write_text("<html></html>", encoding="utf-8")
            launcher = Mock()

            code = main(root, chrome, launcher)

            self.assertEqual(code, 0)
            launcher.assert_called_once()
```

- [ ] **Step 2: Запустить тест и подтвердить ожидаемое падение импорта**

Run: `.venv\Scripts\python.exe -m unittest tests.test_html_open_adaptive_regime -v`

Expected: `ERROR` с `ModuleNotFoundError: No module named 'html_open_adaptive_regime'`.

- [ ] **Step 3: Реализовать минимальный сценарий**

```python
"""Открывает семь HTML-отчётов адаптивного walk-forward в Google Chrome.

Пример запуска:
    python html_open_adaptive_regime.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Sequence

TICKERS = ("rts", "mix", "ng", "br", "gold", "si", "spyf")
CHROME_PATH = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
Launcher = Callable[[list[str]], object]


def report_path(project_root: Path, ticker: str) -> Path:
    """Возвращает путь к walk-forward HTML-отчёту тикера."""
    return project_root / ticker / "research" / "adaptive_regime_walk_forward" / "report.html"


def collect_report_paths(project_root: Path) -> tuple[list[Path], list[Path]]:
    """Разделяет семь ожидаемых отчётов на найденные и отсутствующие."""
    expected = [report_path(project_root, ticker) for ticker in TICKERS]
    return (
        [path for path in expected if path.is_file()],
        [path for path in expected if not path.is_file()],
    )


def launch_reports(
    chrome_path: Path,
    reports: Sequence[Path],
    launcher: Launcher = subprocess.Popen,
) -> None:
    """Открывает все переданные отчёты одним новым окном Chrome."""
    launcher([str(chrome_path), "--new-window", *(str(path) for path in reports)])


def main(
    project_root: Path | None = None,
    chrome_path: Path = CHROME_PATH,
    launcher: Launcher = subprocess.Popen,
) -> int:
    """Проверяет файлы, выводит диагностику и запускает найденные отчёты."""
    root = project_root or Path(__file__).resolve().parent
    if not chrome_path.is_file():
        print(f"[ERROR] Не найден Google Chrome: {chrome_path}")
        return 1
    found, missing = collect_report_paths(root)
    for path in missing:
        print(f"[WARNING] Не найден отчёт: {path}")
    if not found:
        print("[ERROR] Не найден ни один walk-forward HTML-отчёт")
        return 1
    for path in found:
        print(f"[OPEN] {path}")
    launch_reports(chrome_path, found, launcher)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Запустить новые тесты и весь набор**

Run: `.venv\Scripts\python.exe -m unittest tests.test_html_open_adaptive_regime -v`

Expected: `Ran 4 tests ... OK`.

Run: `.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: все тесты завершаются `OK`.

- [ ] **Step 5: Проверить реальный запуск**

Run: `.venv\Scripts\python.exe html_open_adaptive_regime.py`

Expected: открывается одно новое окно Chrome с семью вкладками, код возврата `0`.

- [ ] **Step 6: Зафиксировать и отправить изменения**

```powershell
git add -- html_open_adaptive_regime.py tests/test_html_open_adaptive_regime.py
git commit -m "Добавить открытие walk-forward отчётов в Chrome"
git push origin main
```
