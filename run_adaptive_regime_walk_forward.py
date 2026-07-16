"""Запускает исследовательский адаптивный walk-forward тест.

Сценарий читает готовые df_rez_output.xlsx и не меняет текущие прогнозы.

Примеры запуска:
    python run_adaptive_regime_walk_forward.py --ticker rts
    python run_adaptive_regime_walk_forward.py --all
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from research.adaptive_regime_report import (
    build_summary,
    write_global_summary,
    write_html_report,
    write_results_workbook,
)
from research.adaptive_regime_walk_forward import (
    RegimeDataError,
    load_source_results,
    load_walk_forward_config,
    run_walk_forward,
)

TICKERS = ("rts", "mix", "ng", "br", "gold", "si", "spyf")


@dataclass(frozen=True)
class TickerStatus:
    """Итог обработки одного тикера."""

    ticker: str
    status: str
    message: str
    total_pl: float | None = None
    max_drawdown: float | None = None
    output_dir: str | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Разбирает выбор одного тикера или полного запуска."""
    parser = argparse.ArgumentParser(
        description="Исследовательский адаптивный walk-forward"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", type=str.lower, choices=TICKERS)
    group.add_argument("--all", action="store_true")
    return parser.parse_args(argv)


def _load_settings(path: Path) -> Mapping[str, object]:
    """Загружает YAML тикера и проверяет корневую структуру."""
    if not path.is_file():
        raise RegimeDataError(f"Не найден файл настроек: {path}")
    try:
        settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RegimeDataError(f"Не удалось прочитать {path}: {error}") from error
    if not isinstance(settings, Mapping):
        raise RegimeDataError(f"Корень {path} должен быть YAML-словарём")
    return settings


def run_ticker(ticker_lc: str, project_root: Path) -> TickerStatus:
    """Рассчитывает стратегию и отчёты одного тикера."""
    ticker_dir = project_root / ticker_lc
    settings_path = ticker_dir / "settings.yaml"
    settings = _load_settings(settings_path)
    config = load_walk_forward_config(settings)
    source_path = ticker_dir / "df_rez_output.xlsx"
    source = load_source_results(source_path, config.train_window_days)
    daily = run_walk_forward(source, config)
    ticker = str(settings.get("ticker", ticker_lc.upper()))
    summary = build_summary(daily, ticker)
    output_dir = ticker_dir / "research" / "adaptive_regime_walk_forward"
    config_snapshot = {
        "input_path": str(source_path),
        "train_window_days": config.train_window_days,
        "score_spans": config.score_spans,
        "confidence_thresholds": config.confidence_thresholds,
        "min_regime_days": config.min_regime_days,
        "drawdown_penalty": config.drawdown_penalty,
    }
    write_results_workbook(daily, summary, config_snapshot, output_dir)
    write_html_report(daily, summary, ticker, output_dir)
    adaptive = summary.loc["adaptive"]
    return TickerStatus(
        ticker=ticker_lc,
        status="success",
        message="Расчёт завершён",
        total_pl=float(adaptive["total_pl"]),
        max_drawdown=float(adaptive["max_drawdown"]),
        output_dir=str(output_dir),
    )


def run_many(
    tickers: Sequence[str],
    project_root: Path,
) -> list[TickerStatus]:
    """Обрабатывает тикеры независимо и сохраняет причины сбоев."""
    statuses: list[TickerStatus] = []
    for ticker in tickers:
        try:
            statuses.append(run_ticker(ticker, project_root))
        except RegimeDataError as error:
            statuses.append(TickerStatus(ticker, "skipped", str(error)))
        except Exception as error:
            statuses.append(
                TickerStatus(
                    ticker,
                    "error",
                    f"{type(error).__name__}: {error}",
                )
            )
    return statuses


def exit_code_for(statuses: Sequence[TickerStatus]) -> int:
    """Возвращает успех, если рассчитан хотя бы один тикер."""
    return 0 if any(item.status == "success" for item in statuses) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Запускает исследование и записывает общую сводку."""
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parent
    tickers = TICKERS if args.all else (args.ticker,)
    statuses = run_many(tickers, project_root)
    write_global_summary(
        [asdict(item) for item in statuses],
        project_root / "research" / "adaptive_regime_walk_forward_summary.xlsx",
    )
    for item in statuses:
        print(f"{item.ticker}: {item.status} — {item.message}")
    return exit_code_for(statuses)


if __name__ == "__main__":
    raise SystemExit(main())
