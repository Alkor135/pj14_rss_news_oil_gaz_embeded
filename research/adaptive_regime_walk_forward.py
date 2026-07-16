"""Считает адаптивную импульсно-контртрендовую walk-forward стратегию.

Модуль не изменяет текущие симуляторы и торговые прогнозы.

Примеры запуска через общий сценарий:
    python run_adaptive_regime_walk_forward.py --ticker rts
    python run_adaptive_regime_walk_forward.py --all
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

REGIME_MOMENTUM = "momentum"
REGIME_COUNTERTREND = "countertrend"
REGIME_FLAT = "flat"
EPSILON = 1e-12


class RegimeDataError(ValueError):
    """Ошибка входных данных или настроек walk-forward исследования."""


@dataclass(frozen=True, order=True)
class RegimeParameters:
    """Параметры одного кандидата режимного фильтра."""

    score_span: int
    confidence_threshold: float
    min_regime_days: int


@dataclass(frozen=True)
class WalkForwardConfig:
    """Проверенные настройки walk-forward исследования."""

    train_window_days: int
    score_spans: tuple[int, ...]
    confidence_thresholds: tuple[float, ...]
    min_regime_days: tuple[int, ...]
    drawdown_penalty: float


@dataclass(frozen=True)
class CandidateSimulation:
    """Результат внутренней симуляции одного набора параметров."""

    parameters: RegimeParameters
    daily: pd.DataFrame
    total_pl: float
    max_drawdown: float
    objective: float
    switches: int


def build_payoff_columns(source: pd.DataFrame) -> pd.DataFrame:
    """Добавляет P/L контртренда, импульса и их разность."""
    result = source.copy()
    result["countertrend_pl"] = pd.to_numeric(
        result["P/L"], errors="raise"
    ).astype(float)
    result["momentum_pl"] = -result["countertrend_pl"]
    result["advantage"] = result["momentum_pl"] - result["countertrend_pl"]
    return result


def classify_score(score: float, threshold: float) -> str:
    """Преобразует нормированный индикатор в один из трёх режимов."""
    if score > threshold:
        return REGIME_MOMENTUM
    if score < -threshold:
        return REGIME_COUNTERTREND
    return REGIME_FLAT


def calculate_max_drawdown(daily_pl: pd.Series) -> float:
    """Возвращает максимальную просадку как положительную величину."""
    equity = daily_pl.astype(float).cumsum()
    with_origin = pd.concat(
        [pd.Series([0.0]), equity.reset_index(drop=True)], ignore_index=True
    )
    return float((with_origin.cummax() - with_origin).max())


def calculate_objective(
    total_pl: float,
    max_drawdown: float,
    penalty: float,
) -> float:
    """Считает прибыль за вычетом штрафа за максимальную просадку."""
    return float(total_pl - penalty * max_drawdown)


def compute_causal_scores(advantage: pd.Series, span: int) -> pd.Series:
    """Считает нормированный EWM-индикатор только по предыдущим дням."""
    known = advantage.astype(float).shift(1)
    mean = known.ewm(span=span, adjust=False, min_periods=span).mean()
    scale = known.abs().ewm(span=span, adjust=False, min_periods=span).mean()
    return mean / scale.clip(lower=EPSILON)


def apply_minimum_regime_days(
    proposed: Sequence[str],
    minimum_days: int,
) -> list[str]:
    """Не разрешает покинуть активный режим раньше заданного срока."""
    applied: list[str] = []
    current = REGIME_FLAT
    locked_days_left = 0
    for suggestion in proposed:
        if current != REGIME_FLAT and locked_days_left > 0:
            applied.append(current)
            locked_days_left -= 1
            continue
        if suggestion != current:
            current = suggestion
            locked_days_left = minimum_days - 1 if current != REGIME_FLAT else 0
        applied.append(current)
    return applied


def count_switches(modes: Sequence[str]) -> int:
    """Считает фактические изменения режима между соседними днями."""
    return sum(left != right for left, right in zip(modes, modes[1:]))


def simulate_candidate(
    training: pd.DataFrame,
    parameters: RegimeParameters,
    drawdown_penalty: float,
) -> CandidateSimulation:
    """Причинно тестирует одного кандидата внутри обучающего окна."""
    daily = training.copy()
    daily["regime_score"] = compute_causal_scores(
        daily["advantage"], parameters.score_span
    )
    proposed = [
        classify_score(value, parameters.confidence_threshold)
        if pd.notna(value)
        else REGIME_FLAT
        for value in daily["regime_score"]
    ]
    daily["proposed_mode"] = proposed
    daily["applied_mode"] = apply_minimum_regime_days(
        proposed, parameters.min_regime_days
    )
    daily["strategy_pl"] = np.select(
        [
            daily["applied_mode"].eq(REGIME_MOMENTUM),
            daily["applied_mode"].eq(REGIME_COUNTERTREND),
        ],
        [daily["momentum_pl"], daily["countertrend_pl"]],
        default=0.0,
    )
    total_pl = float(daily["strategy_pl"].sum())
    max_drawdown = calculate_max_drawdown(daily["strategy_pl"])
    switches = count_switches(daily["applied_mode"].tolist())
    return CandidateSimulation(
        parameters=parameters,
        daily=daily,
        total_pl=total_pl,
        max_drawdown=max_drawdown,
        objective=calculate_objective(
            total_pl, max_drawdown, drawdown_penalty
        ),
        switches=switches,
    )


def generate_candidates(config: WalkForwardConfig) -> list[RegimeParameters]:
    """Строит стабильную декартову сетку параметров."""
    return [
        RegimeParameters(span, threshold, minimum)
        for span in sorted(config.score_spans)
        for threshold in sorted(config.confidence_thresholds)
        for minimum in sorted(config.min_regime_days)
    ]


def select_candidate(
    training: pd.DataFrame,
    config: WalkForwardConfig,
) -> CandidateSimulation | None:
    """Выбирает лучший обучающий результат или нейтральный эталон."""
    simulations = [
        simulate_candidate(training, candidate, config.drawdown_penalty)
        for candidate in generate_candidates(config)
    ]
    simulations.sort(
        key=lambda item: (
            -item.objective,
            item.max_drawdown,
            item.switches,
            item.parameters,
        )
    )
    best = simulations[0]
    return best if best.objective > 0.0 else None


def compute_next_score(training: pd.DataFrame, span: int) -> float:
    """Считает индикатор следующего дня по известному обучающему окну."""
    advantage = training["advantage"].astype(float)
    mean = advantage.ewm(
        span=span, adjust=False, min_periods=span
    ).mean().iloc[-1]
    scale = advantage.abs().ewm(
        span=span, adjust=False, min_periods=span
    ).mean().iloc[-1]
    if pd.isna(mean) or pd.isna(scale):
        return float("nan")
    return float(mean / max(float(scale), EPSILON))


def run_walk_forward(
    source: pd.DataFrame,
    config: WalkForwardConfig,
) -> pd.DataFrame:
    """Выполняет ежедневный вложенный walk-forward без утечки будущего."""
    prepared = build_payoff_columns(source.reset_index(drop=True))
    current_mode = REGIME_FLAT
    locked_days_left = 0
    rows: list[dict[str, object]] = []

    for position in range(config.train_window_days, len(prepared)):
        training = prepared.iloc[
            position - config.train_window_days : position
        ].copy()
        selected = select_candidate(training, config)
        if selected is None:
            score = float("nan")
            proposed_mode = REGIME_FLAT
            parameters = None
        else:
            parameters = selected.parameters
            score = compute_next_score(training, parameters.score_span)
            proposed_mode = classify_score(
                score, parameters.confidence_threshold
            )

        if current_mode != REGIME_FLAT and locked_days_left > 0:
            applied_mode = current_mode
            locked_days_left -= 1
        else:
            applied_mode = proposed_mode
            if applied_mode != current_mode:
                current_mode = applied_mode
                locked_days_left = (
                    parameters.min_regime_days - 1
                    if parameters is not None and current_mode != REGIME_FLAT
                    else 0
                )

        day = prepared.iloc[position]
        strategy_pl = (
            float(day["momentum_pl"])
            if applied_mode == REGIME_MOMENTUM
            else float(day["countertrend_pl"])
            if applied_mode == REGIME_COUNTERTREND
            else 0.0
        )
        rows.append(
            {
                "TRADEDATE": day["TRADEDATE"],
                "max": day["max"],
                "countertrend_pl": float(day["countertrend_pl"]),
                "momentum_pl": float(day["momentum_pl"]),
                "proposed_mode": proposed_mode,
                "applied_mode": applied_mode,
                "regime_score": score,
                "score_span": parameters.score_span if parameters else pd.NA,
                "confidence_threshold": (
                    parameters.confidence_threshold if parameters else pd.NA
                ),
                "min_regime_days": (
                    parameters.min_regime_days if parameters else pd.NA
                ),
                "train_objective": selected.objective if selected else 0.0,
                "train_total_pl": selected.total_pl if selected else 0.0,
                "train_max_drawdown": (
                    selected.max_drawdown if selected else 0.0
                ),
                "train_switches": selected.switches if selected else 0,
                "strategy_pl": strategy_pl,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["cumulative_pl"] = result["strategy_pl"].cumsum()
    running_peak = result["cumulative_pl"].cummax().clip(lower=0.0)
    result["drawdown"] = running_peak - result["cumulative_pl"]
    return result


def _as_non_empty_sequence(
    raw: Mapping[str, object],
    name: str,
) -> Sequence[object]:
    """Возвращает непустую последовательность параметров или сообщает ошибку."""
    value = raw.get(name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise RegimeDataError(f"{name} должен быть непустым списком")
    return value


def load_walk_forward_config(
    settings: Mapping[str, object],
) -> WalkForwardConfig:
    """Читает и строго проверяет блок regime_walk_forward."""
    raw = settings.get("regime_walk_forward")
    if not isinstance(raw, Mapping):
        raise RegimeDataError("Отсутствует блок regime_walk_forward")
    try:
        config = WalkForwardConfig(
            train_window_days=int(raw["train_window_days"]),
            score_spans=tuple(
                int(value) for value in _as_non_empty_sequence(raw, "score_spans")
            ),
            confidence_thresholds=tuple(
                float(value)
                for value in _as_non_empty_sequence(raw, "confidence_thresholds")
            ),
            min_regime_days=tuple(
                int(value)
                for value in _as_non_empty_sequence(raw, "min_regime_days")
            ),
            drawdown_penalty=float(raw["drawdown_penalty"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RegimeDataError(f"Некорректные настройки walk-forward: {error}") from error
    if config.train_window_days < 2:
        raise RegimeDataError("train_window_days должен быть не меньше 2")
    if any(
        value < 1 or value >= config.train_window_days
        for value in config.score_spans
    ):
        raise RegimeDataError(
            "score_spans должны быть положительными и меньше train_window_days"
        )
    if any(
        not np.isfinite(value) or not 0.0 <= value <= 1.0
        for value in config.confidence_thresholds
    ):
        raise RegimeDataError(
            "confidence_thresholds должны находиться в диапазоне [0, 1]"
        )
    if any(value < 1 for value in config.min_regime_days):
        raise RegimeDataError("min_regime_days должны быть положительными")
    if (
        not np.isfinite(config.drawdown_penalty)
        or config.drawdown_penalty < 0.0
    ):
        raise RegimeDataError(
            "drawdown_penalty должен быть конечным и неотрицательным"
        )
    return config


def validate_source_results(
    source: pd.DataFrame,
    train_window_days: int,
) -> pd.DataFrame:
    """Проверяет контракт входного df_rez_output.xlsx."""
    required = {"TRADEDATE", "P/L", "max"}
    missing = required.difference(source.columns)
    if missing:
        raise RegimeDataError(
            f"Отсутствуют столбцы: {', '.join(sorted(missing))}"
        )
    result = source.loc[:, ["TRADEDATE", "P/L", "max"]].copy()
    try:
        result["TRADEDATE"] = pd.to_datetime(
            result["TRADEDATE"], errors="raise"
        )
        result["P/L"] = pd.to_numeric(
            result["P/L"], errors="raise"
        ).astype(float)
    except (TypeError, ValueError) as error:
        raise RegimeDataError(f"Некорректные входные данные: {error}") from error
    if result["TRADEDATE"].duplicated().any():
        raise RegimeDataError("Обнаружены повторяющиеся даты")
    if not np.isfinite(result["P/L"]).all():
        raise RegimeDataError("P/L содержит бесконечные или пустые значения")
    result = result.sort_values("TRADEDATE").reset_index(drop=True)
    if len(result) <= train_window_days:
        raise RegimeDataError(
            "В истории недостаточно данных для одного out-of-sample дня"
        )
    return result


def load_source_results(
    path: Path,
    train_window_days: int,
) -> pd.DataFrame:
    """Загружает и проверяет результат текущего симулятора."""
    if not path.is_file():
        raise RegimeDataError(f"Не найден входной файл: {path}")
    try:
        source = pd.read_excel(path)
    except Exception as error:
        raise RegimeDataError(f"Не удалось прочитать {path}: {error}") from error
    return validate_source_results(source, train_window_days)
