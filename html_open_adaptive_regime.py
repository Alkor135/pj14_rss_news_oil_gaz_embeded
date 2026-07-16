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
    return (
        project_root
        / ticker
        / "research"
        / "adaptive_regime_walk_forward"
        / "report.html"
    )


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
    launcher(
        [str(chrome_path), "--new-window", *(str(path) for path in reports)]
    )


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
