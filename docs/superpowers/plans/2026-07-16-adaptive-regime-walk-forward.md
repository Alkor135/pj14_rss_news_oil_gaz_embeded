# Adaptive Regime Walk-Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Реализовать изолированный причинный walk-forward бэктест, который для каждого из семи тикеров ежедневно выбирает импульсный, контртрендовый или нейтральный режим и создаёт отдельные Excel/HTML-отчёты.

**Architecture:** Корневой CLI запускает один или все тикеры. Модуль `research/adaptive_regime_walk_forward.py` отвечает только за конфигурацию, валидацию и расчёты, а `research/adaptive_regime_report.py` — за метрики и артефакты. Источник — существующий `{ticker}/df_rez_output.xlsx`; текущие симуляторы, прогнозы и живая торговля не импортируются и не изменяются.

**Tech Stack:** Python 3, стандартный `unittest`, `dataclasses`, `argparse`, pandas, NumPy, PyYAML, openpyxl, Plotly.

---

## Карта файлов

- Create: `research/__init__.py` — делает исследовательские расчёты импортируемым пакетом.
- Create: `research/adaptive_regime_walk_forward.py` — типы, загрузка, валидация, режимный индикатор, подбор кандидата и walk-forward.
- Create: `research/adaptive_regime_report.py` — метрики, Excel, HTML и общая сводка.
- Create: `run_adaptive_regime_walk_forward.py` — CLI и изоляция ошибок отдельных тикеров.
- Create: `tests/__init__.py` — пакет тестов.
- Create: `tests/test_adaptive_regime_walk_forward.py` — тесты ядра и отсутствия утечки.
- Create: `tests/test_adaptive_regime_report.py` — тесты Excel/HTML и метрик.
- Create: `tests/test_run_adaptive_regime_walk_forward.py` — тесты CLI и частично успешного запуска.
- Create: `docs/adaptive_momentum_countertrend_walk_forward_strategy.md` — подробное описание стратегии.
- Modify: `rts/settings.yaml`, `mix/settings.yaml`, `ng/settings.yaml`, `br/settings.yaml`, `gold/settings.yaml`, `si/settings.yaml`, `spyf/settings.yaml` — только новый блок `regime_walk_forward`.

### Task 1: Базовые типы, P/L и риск-метрики

**Files:**
- Create: `research/__init__.py`
- Create: `research/adaptive_regime_walk_forward.py`
- Create: `tests/__init__.py`
- Create: `tests/test_adaptive_regime_walk_forward.py`

- [ ] **Step 1: Создать падающие тесты базовых расчётов**

```python
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
        source = pd.DataFrame({"P/L": [10.0, -4.0]})
        result = build_payoff_columns(source)
        self.assertEqual(result["countertrend_pl"].tolist(), [10.0, -4.0])
        self.assertEqual(result["momentum_pl"].tolist(), [-10.0, 4.0])
        self.assertEqual(result["advantage"].tolist(), [-20.0, 8.0])

    def test_classify_score_has_neutral_zone(self):
        self.assertEqual(classify_score(0.21, 0.20), REGIME_MOMENTUM)
        self.assertEqual(classify_score(-0.21, 0.20), REGIME_COUNTERTREND)
        self.assertEqual(classify_score(0.20, 0.20), REGIME_FLAT)

    def test_max_drawdown_is_positive_magnitude(self):
        self.assertEqual(calculate_max_drawdown(pd.Series([5.0, -2.0, -4.0, 3.0])), 6.0)

    def test_objective_penalizes_drawdown(self):
        self.assertEqual(calculate_objective(12.0, 5.0, 1.5), 4.5)
```

- [ ] **Step 2: Запустить тесты и подтвердить ожидаемое падение импорта**

Run: `.venv\Scripts\python.exe -m unittest tests.test_adaptive_regime_walk_forward -v`

Expected: `ERROR` с `ModuleNotFoundError` или ошибкой отсутствующих функций.

- [ ] **Step 3: Реализовать минимальные типы и базовые функции**

В `research/adaptive_regime_walk_forward.py` добавить:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

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


def build_payoff_columns(source: pd.DataFrame) -> pd.DataFrame:
    """Добавляет P/L контртренда, импульса и их разность."""
    result = source.copy()
    result["countertrend_pl"] = pd.to_numeric(result["P/L"], errors="raise").astype(float)
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
    equity_with_origin = pd.concat([pd.Series([0.0]), equity.reset_index(drop=True)], ignore_index=True)
    drawdown = equity_with_origin.cummax() - equity_with_origin
    return float(drawdown.max())


def calculate_objective(total_pl: float, max_drawdown: float, penalty: float) -> float:
    """Считает прибыль за вычетом штрафа за максимальную просадку."""
    return float(total_pl - penalty * max_drawdown)
```

В новых Python-файлах добавить в начало русское описание назначения и примеры запуска согласно `AGENTS.md`; всем функциям оставить русские docstring.

- [ ] **Step 4: Запустить тесты базовых расчётов**

Run: `.venv\Scripts\python.exe -m unittest tests.test_adaptive_regime_walk_forward.BasicCalculationsTests -v`

Expected: `Ran 4 tests ... OK`.

- [ ] **Step 5: Зафиксировать базовое ядро**

```powershell
git add -- research/__init__.py research/adaptive_regime_walk_forward.py tests/__init__.py tests/test_adaptive_regime_walk_forward.py
git commit -m "Добавить базовые расчёты режимной стратегии"
```

### Task 2: Причинная симуляция одного кандидата

**Files:**
- Modify: `research/adaptive_regime_walk_forward.py`
- Modify: `tests/test_adaptive_regime_walk_forward.py`

- [ ] **Step 1: Добавить падающие тесты индикатора и минимального режима**

```python
from research.adaptive_regime_walk_forward import (
    RegimeParameters,
    compute_causal_scores,
    simulate_candidate,
)


class CandidateSimulationTests(unittest.TestCase):
    """Проверки причинной симуляции кандидата."""

    def test_score_does_not_use_current_day(self):
        advantage = pd.Series([2.0, 2.0, -100.0])
        original = compute_causal_scores(advantage, span=2)
        changed = compute_causal_scores(pd.Series([2.0, 2.0, 100.0]), span=2)
        self.assertEqual(original.iloc[2], changed.iloc[2])

    def test_candidate_holds_active_regime_for_minimum_days(self):
        frame = build_payoff_columns(pd.DataFrame({"P/L": [-1.0, -1.0, 10.0, 10.0, 10.0]}))
        params = RegimeParameters(1, 0.0, 3)
        result = simulate_candidate(frame, params, drawdown_penalty=1.0)
        self.assertEqual(result.daily.loc[1:3, "applied_mode"].tolist(), [
            REGIME_MOMENTUM,
            REGIME_MOMENTUM,
            REGIME_MOMENTUM,
        ])

    def test_candidate_daily_pl_matches_applied_mode(self):
        frame = build_payoff_columns(pd.DataFrame({"P/L": [-2.0, -2.0, -2.0]}))
        result = simulate_candidate(frame, RegimeParameters(1, 0.0, 1), 0.0)
        active = result.daily[result.daily["applied_mode"] == REGIME_MOMENTUM]
        self.assertTrue((active["strategy_pl"] == active["momentum_pl"]).all())
```

- [ ] **Step 2: Запустить новые тесты и увидеть ошибки отсутствующих функций**

Run: `.venv\Scripts\python.exe -m unittest tests.test_adaptive_regime_walk_forward.CandidateSimulationTests -v`

Expected: `ERROR` из-за отсутствующих `compute_causal_scores` и `simulate_candidate`.

- [ ] **Step 3: Реализовать причинный индикатор и блокировку режима**

Добавить в `research/adaptive_regime_walk_forward.py`:

```python
@dataclass(frozen=True)
class CandidateSimulation:
    """Результат внутренней симуляции одного набора параметров."""

    parameters: RegimeParameters
    daily: pd.DataFrame
    total_pl: float
    max_drawdown: float
    objective: float
    switches: int


def compute_causal_scores(advantage: pd.Series, span: int) -> pd.Series:
    """Считает нормированный EWM-индикатор только по предыдущим дням."""
    known = advantage.astype(float).shift(1)
    mean = known.ewm(span=span, adjust=False, min_periods=span).mean()
    scale = known.abs().ewm(span=span, adjust=False, min_periods=span).mean()
    return mean / scale.clip(lower=EPSILON)


def apply_minimum_regime_days(proposed: Sequence[str], minimum_days: int) -> list[str]:
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
    daily["regime_score"] = compute_causal_scores(daily["advantage"], parameters.score_span)
    proposed = [
        classify_score(value, parameters.confidence_threshold) if pd.notna(value) else REGIME_FLAT
        for value in daily["regime_score"]
    ]
    daily["proposed_mode"] = proposed
    daily["applied_mode"] = apply_minimum_regime_days(proposed, parameters.min_regime_days)
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
        objective=calculate_objective(total_pl, max_drawdown, drawdown_penalty),
        switches=switches,
    )
```

- [ ] **Step 4: Запустить все тесты ядра**

Run: `.venv\Scripts\python.exe -m unittest tests.test_adaptive_regime_walk_forward -v`

Expected: `Ran 7 tests ... OK`.

- [ ] **Step 5: Зафиксировать причинную симуляцию**

```powershell
git add -- research/adaptive_regime_walk_forward.py tests/test_adaptive_regime_walk_forward.py
git commit -m "Добавить причинную симуляцию режимного фильтра"
```

### Task 3: Подбор параметров и внешний walk-forward

**Files:**
- Modify: `research/adaptive_regime_walk_forward.py`
- Modify: `tests/test_adaptive_regime_walk_forward.py`

- [ ] **Step 1: Добавить падающие тесты выбора и отсутствия утечки**

```python
from research.adaptive_regime_walk_forward import (
    WalkForwardConfig,
    generate_candidates,
    run_walk_forward,
)


class WalkForwardTests(unittest.TestCase):
    """Проверки вложенного ежедневного walk-forward."""

    def setUp(self):
        self.config = WalkForwardConfig(
            train_window_days=6,
            score_spans=(1, 2),
            confidence_thresholds=(0.0, 0.2),
            min_regime_days=(1, 2),
            drawdown_penalty=1.0,
        )

    def test_generate_candidates_is_deterministic(self):
        first = generate_candidates(self.config)
        second = generate_candidates(self.config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)

    def test_current_outcome_does_not_change_current_decision(self):
        source = pd.DataFrame({
            "TRADEDATE": pd.date_range("2026-01-01", periods=12),
            "P/L": [-1.0] * 12,
            "max": [5] * 12,
        })
        original = run_walk_forward(source, self.config)
        changed_source = source.copy()
        changed_source.loc[11, "P/L"] = 1000.0
        changed = run_walk_forward(changed_source, self.config)
        self.assertEqual(original.iloc[-1]["applied_mode"], changed.iloc[-1]["applied_mode"])
        self.assertEqual(original.iloc[-1]["regime_score"], changed.iloc[-1]["regime_score"])

    def test_walk_forward_changes_from_countertrend_to_momentum(self):
        source = pd.DataFrame({
            "TRADEDATE": pd.date_range("2026-01-01", periods=30),
            "P/L": [1.0] * 15 + [-1.0] * 15,
            "max": [5] * 30,
        })
        result = run_walk_forward(source, self.config)
        self.assertIn(REGIME_COUNTERTREND, result["applied_mode"].iloc[:10].tolist())
        self.assertIn(REGIME_MOMENTUM, result["applied_mode"].iloc[-10:].tolist())

    def test_first_training_window_is_excluded(self):
        source = pd.DataFrame({
            "TRADEDATE": pd.date_range("2026-01-01", periods=10),
            "P/L": [1.0] * 10,
            "max": [5] * 10,
        })
        result = run_walk_forward(source, self.config)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.iloc[0]["TRADEDATE"], source.iloc[6]["TRADEDATE"])
```

- [ ] **Step 2: Запустить тесты и подтвердить ожидаемое падение**

Run: `.venv\Scripts\python.exe -m unittest tests.test_adaptive_regime_walk_forward.WalkForwardTests -v`

Expected: `ERROR` из-за отсутствующих `generate_candidates` и `run_walk_forward`.

- [ ] **Step 3: Реализовать подбор кандидата и внешний цикл**

Добавить функции с интерфейсами:

```python
def generate_candidates(config: WalkForwardConfig) -> list[RegimeParameters]:
    """Строит стабильную декартову сетку параметров."""
    return [
        RegimeParameters(span, threshold, minimum)
        for span in sorted(config.score_spans)
        for threshold in sorted(config.confidence_thresholds)
        for minimum in sorted(config.min_regime_days)
    ]


def select_candidate(training: pd.DataFrame, config: WalkForwardConfig) -> CandidateSimulation | None:
    """Выбирает лучший обучающий результат или нейтральный эталон."""
    simulations = [simulate_candidate(training, item, config.drawdown_penalty) for item in generate_candidates(config)]
    simulations.sort(key=lambda item: (-item.objective, item.max_drawdown, item.switches, item.parameters))
    best = simulations[0]
    return best if best.objective > 0.0 else None


def compute_next_score(training: pd.DataFrame, span: int) -> float:
    """Считает индикатор следующего дня по полностью известному обучающему окну."""
    mean = training["advantage"].ewm(span=span, adjust=False, min_periods=span).mean().iloc[-1]
    scale = training["advantage"].abs().ewm(span=span, adjust=False, min_periods=span).mean().iloc[-1]
    if pd.isna(mean) or pd.isna(scale):
        return float("nan")
    return float(mean / max(float(scale), EPSILON))


def run_walk_forward(source: pd.DataFrame, config: WalkForwardConfig) -> pd.DataFrame:
    """Выполняет ежедневный вложенный walk-forward без утечки будущего."""
    prepared = build_payoff_columns(source.reset_index(drop=True))
    current_mode = REGIME_FLAT
    locked_days_left = 0
    rows: list[dict[str, object]] = []
    for position in range(config.train_window_days, len(prepared)):
        training = prepared.iloc[position - config.train_window_days:position].copy()
        selected = select_candidate(training, config)
        if selected is None:
            score = float("nan")
            proposed_mode = REGIME_FLAT
            parameters = None
        else:
            parameters = selected.parameters
            score = compute_next_score(training, parameters.score_span)
            proposed_mode = classify_score(score, parameters.confidence_threshold)

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
            float(day["momentum_pl"]) if applied_mode == REGIME_MOMENTUM
            else float(day["countertrend_pl"]) if applied_mode == REGIME_COUNTERTREND
            else 0.0
        )
        rows.append({
            "TRADEDATE": day["TRADEDATE"],
            "max": day["max"],
            "countertrend_pl": float(day["countertrend_pl"]),
            "momentum_pl": float(day["momentum_pl"]),
            "proposed_mode": proposed_mode,
            "applied_mode": applied_mode,
            "regime_score": score,
            "score_span": parameters.score_span if parameters else pd.NA,
            "confidence_threshold": parameters.confidence_threshold if parameters else pd.NA,
            "min_regime_days": parameters.min_regime_days if parameters else pd.NA,
            "train_objective": selected.objective if selected else 0.0,
            "train_total_pl": selected.total_pl if selected else 0.0,
            "train_max_drawdown": selected.max_drawdown if selected else 0.0,
            "train_switches": selected.switches if selected else 0,
            "strategy_pl": strategy_pl,
        })
    result = pd.DataFrame(rows)
    result["cumulative_pl"] = result["strategy_pl"].cumsum()
    result["drawdown"] = result["cumulative_pl"].cummax().clip(lower=0.0) - result["cumulative_pl"]
    return result
```

При реализации отдельно исправить состояние `current_mode` после ветки блокировки так, чтобы оно всегда совпадало с `applied_mode`; тест смены режима должен это подтверждать.

- [ ] **Step 4: Запустить полный тест ядра**

Run: `.venv\Scripts\python.exe -m unittest tests.test_adaptive_regime_walk_forward -v`

Expected: все 11 тестов завершаются `OK`.

- [ ] **Step 5: Зафиксировать walk-forward**

```powershell
git add -- research/adaptive_regime_walk_forward.py tests/test_adaptive_regime_walk_forward.py
git commit -m "Добавить вложенный walk-forward выбор режима"
```

### Task 4: Настройки и валидация входных данных

**Files:**
- Modify: `research/adaptive_regime_walk_forward.py`
- Modify: `tests/test_adaptive_regime_walk_forward.py`

- [ ] **Step 1: Добавить тесты настроек, дат и короткой истории**

```python
import tempfile
from pathlib import Path

from research.adaptive_regime_walk_forward import (
    RegimeDataError,
    load_source_results,
    load_walk_forward_config,
    validate_source_results,
)


class InputValidationTests(unittest.TestCase):
    """Проверки настроек и входного Excel."""

    def test_config_rejects_span_not_smaller_than_training_window(self):
        settings = {"regime_walk_forward": {
            "train_window_days": 5,
            "score_spans": [5],
            "confidence_thresholds": [0.1],
            "min_regime_days": [1],
            "drawdown_penalty": 1.0,
        }}
        with self.assertRaisesRegex(RegimeDataError, "score_spans"):
            load_walk_forward_config(settings)

    def test_validation_sorts_dates_and_rejects_duplicates(self):
        source = pd.DataFrame({
            "TRADEDATE": ["2026-01-02", "2026-01-01"],
            "P/L": [1.0, 2.0],
            "max": [5, 6],
        })
        validated = validate_source_results(source, train_window_days=1)
        self.assertTrue(validated["TRADEDATE"].is_monotonic_increasing)
        with self.assertRaisesRegex(RegimeDataError, "повтор"):
            validate_source_results(pd.concat([source, source.iloc[[0]]]), 1)

    def test_validation_requires_one_oos_day(self):
        source = pd.DataFrame({"TRADEDATE": ["2026-01-01"], "P/L": [1.0], "max": [5]})
        with self.assertRaisesRegex(RegimeDataError, "недостаточно"):
            validate_source_results(source, train_window_days=1)
```

- [ ] **Step 2: Запустить тесты и увидеть ожидаемое падение импорта**

Run: `.venv\Scripts\python.exe -m unittest tests.test_adaptive_regime_walk_forward.InputValidationTests -v`

Expected: `ERROR` из-за отсутствующих функций загрузки и валидации.

- [ ] **Step 3: Реализовать строгую конфигурацию и загрузку Excel**

Реализовать интерфейсы:

```python
def load_walk_forward_config(settings: Mapping[str, object]) -> WalkForwardConfig:
    """Читает и проверяет блок regime_walk_forward."""
    raw = settings.get("regime_walk_forward")
    if not isinstance(raw, Mapping):
        raise RegimeDataError("Отсутствует блок regime_walk_forward")
    config = WalkForwardConfig(
        train_window_days=int(raw["train_window_days"]),
        score_spans=tuple(int(value) for value in raw["score_spans"]),
        confidence_thresholds=tuple(float(value) for value in raw["confidence_thresholds"]),
        min_regime_days=tuple(int(value) for value in raw["min_regime_days"]),
        drawdown_penalty=float(raw["drawdown_penalty"]),
    )
    if config.train_window_days < 2:
        raise RegimeDataError("train_window_days должен быть не меньше 2")
    if not config.score_spans or any(value < 1 or value >= config.train_window_days for value in config.score_spans):
        raise RegimeDataError("score_spans должны быть положительными и меньше train_window_days")
    if not config.confidence_thresholds or any(not 0.0 <= value <= 1.0 for value in config.confidence_thresholds):
        raise RegimeDataError("confidence_thresholds должны находиться в диапазоне [0, 1]")
    if not config.min_regime_days or any(value < 1 for value in config.min_regime_days):
        raise RegimeDataError("min_regime_days должны быть положительными")
    if not np.isfinite(config.drawdown_penalty) or config.drawdown_penalty < 0.0:
        raise RegimeDataError("drawdown_penalty должен быть конечным и неотрицательным")
    return config


def validate_source_results(source: pd.DataFrame, train_window_days: int) -> pd.DataFrame:
    """Проверяет контракт входного df_rez_output.xlsx."""
    required = {"TRADEDATE", "P/L", "max"}
    missing = required.difference(source.columns)
    if missing:
        raise RegimeDataError(f"Отсутствуют столбцы: {', '.join(sorted(missing))}")
    result = source.loc[:, ["TRADEDATE", "P/L", "max"]].copy()
    result["TRADEDATE"] = pd.to_datetime(result["TRADEDATE"], errors="raise")
    if result["TRADEDATE"].duplicated().any():
        raise RegimeDataError("Обнаружены повторяющиеся даты")
    result["P/L"] = pd.to_numeric(result["P/L"], errors="raise").astype(float)
    if not np.isfinite(result["P/L"]).all():
        raise RegimeDataError("P/L содержит бесконечные или пустые значения")
    result = result.sort_values("TRADEDATE").reset_index(drop=True)
    if len(result) <= train_window_days:
        raise RegimeDataError("В истории недостаточно данных для одного out-of-sample дня")
    return result


def load_source_results(path: Path, train_window_days: int) -> pd.DataFrame:
    """Загружает и проверяет результат текущего симулятора."""
    if not path.is_file():
        raise RegimeDataError(f"Не найден входной файл: {path}")
    return validate_source_results(pd.read_excel(path), train_window_days)
```

- [ ] **Step 4: Запустить все тесты ядра**

Run: `.venv\Scripts\python.exe -m unittest tests.test_adaptive_regime_walk_forward -v`

Expected: все тесты завершаются `OK`.

- [ ] **Step 5: Зафиксировать загрузку и валидацию**

```powershell
git add -- research/adaptive_regime_walk_forward.py tests/test_adaptive_regime_walk_forward.py
git commit -m "Добавить проверку настроек и входных результатов"
```

### Task 5: Метрики и исследовательские отчёты

**Files:**
- Create: `research/adaptive_regime_report.py`
- Create: `tests/test_adaptive_regime_report.py`

- [ ] **Step 1: Добавить тесты метрик и файлов отчёта**

```python
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from research.adaptive_regime_report import (
    build_summary,
    write_html_report,
    write_results_workbook,
)


class ReportTests(unittest.TestCase):
    """Проверки метрик и исследовательских артефактов."""

    def setUp(self):
        self.daily = pd.DataFrame({
            "TRADEDATE": pd.date_range("2026-01-01", periods=4),
            "max": [5, 5, 6, 6],
            "countertrend_pl": [2.0, -1.0, 3.0, -2.0],
            "momentum_pl": [-2.0, 1.0, -3.0, 2.0],
            "proposed_mode": ["countertrend", "flat", "countertrend", "momentum"],
            "applied_mode": ["countertrend", "flat", "countertrend", "momentum"],
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
        })

    def test_summary_contains_strategy_and_benchmarks(self):
        summary = build_summary(self.daily, "RTS")
        self.assertEqual(summary.loc["adaptive", "total_pl"], 7.0)
        self.assertIn("countertrend", summary.index)
        self.assertIn("momentum", summary.index)

    def test_writers_create_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = build_summary(self.daily, "RTS")
            workbook = write_results_workbook(self.daily, summary, {"train_window_days": 63}, output)
            html = write_html_report(self.daily, summary, "RTS", output)
            self.assertEqual(pd.ExcelFile(workbook).sheet_names, ["daily", "summary", "config"])
            self.assertIn("RTS", html.read_text(encoding="utf-8"))
```

- [ ] **Step 2: Запустить тесты и увидеть ожидаемое падение импорта**

Run: `.venv\Scripts\python.exe -m unittest tests.test_adaptive_regime_report -v`

Expected: `ERROR` из-за отсутствующего модуля отчёта.

- [ ] **Step 3: Реализовать метрики и Excel**

В `research/adaptive_regime_report.py` определить:

```python
def calculate_metrics(pl: pd.Series, active: pd.Series) -> dict[str, float | int]:
    """Считает сопоставимые метрики одной кривой P/L."""
    values = pl.astype(float)
    active_values = values[active.astype(bool)]
    gains = float(active_values[active_values > 0].sum())
    losses = abs(float(active_values[active_values < 0].sum()))
    return {
        "total_pl": float(values.sum()),
        "max_drawdown": calculate_max_drawdown(values),
        "profit_factor": gains / losses if losses > 0.0 else float("inf") if gains > 0.0 else 0.0,
        "win_rate": float((active_values > 0).mean()) if len(active_values) else 0.0,
        "active_days": int(active.sum()),
    }


def build_summary(daily: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Сравнивает адаптивную стратегию с двумя постоянными режимами."""
    rows = {
        "adaptive": calculate_metrics(daily["strategy_pl"], daily["applied_mode"].ne("flat")),
        "countertrend": calculate_metrics(daily["countertrend_pl"], pd.Series(True, index=daily.index)),
        "momentum": calculate_metrics(daily["momentum_pl"], pd.Series(True, index=daily.index)),
    }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.insert(0, "ticker", ticker)
    result["mode_share_momentum"] = float(daily["applied_mode"].eq("momentum").mean())
    result["mode_share_countertrend"] = float(daily["applied_mode"].eq("countertrend").mean())
    result["mode_share_flat"] = float(daily["applied_mode"].eq("flat").mean())
    result["switches"] = sum(
        left != right for left, right in zip(daily["applied_mode"], daily["applied_mode"].iloc[1:])
    )
    return result
```

`write_results_workbook()` должен создать каталог, записать листы `daily`, `summary`, `config` через `pd.ExcelWriter(..., engine="openpyxl")` и вернуть `Path` файла `results.xlsx`.

- [ ] **Step 4: Реализовать самодостаточный Plotly HTML**

`write_html_report()` должен собрать `make_subplots` с пятью блоками: cumulative P/L трёх вариантов, дневной adaptive P/L, drawdown, цветная шкала режимов, подобранные параметры. Использовать `fig.write_html(path, include_plotlyjs=True, full_html=True)`, чтобы локальный отчёт не зависел от интернета. Вернуть `Path` файла `report.html`.

- [ ] **Step 5: Запустить тесты отчёта**

Run: `.venv\Scripts\python.exe -m unittest tests.test_adaptive_regime_report -v`

Expected: `Ran 2 tests ... OK`.

- [ ] **Step 6: Зафиксировать отчёты**

```powershell
git add -- research/adaptive_regime_report.py tests/test_adaptive_regime_report.py
git commit -m "Добавить отчёты адаптивного walk-forward теста"
```

### Task 6: CLI для одного и всех тикеров

**Files:**
- Create: `run_adaptive_regime_walk_forward.py`
- Create: `tests/test_run_adaptive_regime_walk_forward.py`
- Modify: `research/adaptive_regime_report.py`

- [ ] **Step 1: Добавить падающие тесты частично успешного запуска**

Тесты должны создать временный проект с двумя тикерами: один с валидными настройками и Excel, второй без Excel. Проверить:

```python
statuses = run_many(("rts", "mix"), project_root)
self.assertEqual(statuses[0].status, "success")
self.assertEqual(statuses[1].status, "error")
self.assertEqual(exit_code_for(statuses), 0)
self.assertEqual(exit_code_for([statuses[1]]), 1)
```

Также проверить `parse_args(["--ticker", "RTS"])` и взаимную исключительность `--ticker`/`--all`.

- [ ] **Step 2: Запустить тесты и увидеть ожидаемое падение импорта**

Run: `.venv\Scripts\python.exe -m unittest tests.test_run_adaptive_regime_walk_forward -v`

Expected: `ERROR` из-за отсутствующего запускного модуля.

- [ ] **Step 3: Реализовать запускной скрипт**

В начале файла разместить русское описание и примеры:

```python
"""Запускает исследовательский адаптивный walk-forward тест.

Примеры запуска:
    python run_adaptive_regime_walk_forward.py --ticker rts
    python run_adaptive_regime_walk_forward.py --all
"""
```

Определить `TICKERS = ("rts", "mix", "ng", "br", "gold", "si", "spyf")`, dataclass `TickerStatus(ticker, status, message, total_pl, max_drawdown, output_dir)` и функции:

- `parse_args(argv)` — обязательная взаимоисключающая группа `--ticker`/`--all`;
- `run_ticker(ticker_lc, project_root)` — загрузка YAML, Excel, walk-forward и два отчёта;
- `run_many(tickers, project_root)` — `try/except` вокруг каждого тикера, продолжение после ошибки;
- `exit_code_for(statuses)` — `0`, если есть `success`, иначе `1`;
- `main(argv=None)` — печать краткой таблицы и запись общей сводки.

Преобразовывать значение `--ticker` к нижнему регистру и отклонять неизвестный тикер. Никаких импортов из тикерных `simulate_trade.py` или торговых скриптов.

- [ ] **Step 4: Реализовать общую сводку**

В `research/adaptive_regime_report.py` добавить `write_global_summary(statuses: Sequence[Mapping[str, object]], path: Path) -> Path`, создающую `research/adaptive_regime_walk_forward_summary.xlsx` с колонками `ticker`, `status`, `message`, `total_pl`, `max_drawdown`, `output_dir`.

- [ ] **Step 5: Запустить CLI-тесты и весь набор**

Run: `.venv\Scripts\python.exe -m unittest tests.test_run_adaptive_regime_walk_forward -v`

Expected: CLI-тесты завершаются `OK`.

Run: `.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: весь набор завершается `OK`.

- [ ] **Step 6: Зафиксировать CLI**

```powershell
git add -- run_adaptive_regime_walk_forward.py research/adaptive_regime_report.py tests/test_run_adaptive_regime_walk_forward.py
git commit -m "Добавить запуск walk-forward теста для всех тикеров"
```

### Task 7: Настройки семи тикеров и подробная документация

**Files:**
- Modify: `rts/settings.yaml`
- Modify: `mix/settings.yaml`
- Modify: `ng/settings.yaml`
- Modify: `br/settings.yaml`
- Modify: `gold/settings.yaml`
- Modify: `si/settings.yaml`
- Modify: `spyf/settings.yaml`
- Create: `docs/adaptive_momentum_countertrend_walk_forward_strategy.md`
- Modify: `tests/test_adaptive_regime_walk_forward.py`

- [ ] **Step 1: Добавить падающий тест наличия настроек во всех тикерах**

```python
def test_all_ticker_settings_have_research_block(self):
    root = Path(__file__).resolve().parents[1]
    for ticker in ("rts", "mix", "ng", "br", "gold", "si", "spyf"):
        settings = yaml.safe_load((root / ticker / "settings.yaml").read_text(encoding="utf-8"))
        config = load_walk_forward_config(settings)
        self.assertEqual(config.train_window_days, 63)
```

- [ ] **Step 2: Запустить тест и подтвердить отсутствие блока**

Run: `.venv\Scripts\python.exe -m unittest tests.test_adaptive_regime_walk_forward.InputValidationTests.test_all_ticker_settings_have_research_block -v`

Expected: `ERROR` с сообщением об отсутствующем `regime_walk_forward`.

- [ ] **Step 3: Добавить одинаковый блок в семь settings.yaml**

```yaml
# Отдельный исследовательский walk-forward; текущая торговля эти параметры не использует
regime_walk_forward:
  train_window_days: 63
  score_spans: [5, 8, 13, 21]
  confidence_thresholds: [0.05, 0.10, 0.20, 0.30]
  min_regime_days: [1, 3, 5]
  drawdown_penalty: 1.0
```

Не менять существующий `start_date_test: '2025-11-01'` и остальные пользовательские настройки.

- [ ] **Step 4: Написать подробный документ стратегии**

`docs/adaptive_momentum_countertrend_walk_forward_strategy.md` должен содержать:

1. причину исследования и разбор разворота RTS после 2026-05-03;
2. определения импульса, контртренда и нейтрального режима;
3. формулы `C_d`, `M_d`, `D_d`, нормированного EWM score и целевой функции;
4. ежедневную последовательность вложенного walk-forward;
5. объяснение отсутствия утечки будущего;
6. полный справочник параметров YAML;
7. команды запуска одного и всех тикеров;
8. описание листов Excel и панелей HTML;
9. интерпретацию метрик и сравнительных benchmark;
10. ограничения: короткая история, подбор сетки, отсутствие издержек и запрет автоматического переноса в live.

- [ ] **Step 5: Запустить тесты настроек**

Run: `.venv\Scripts\python.exe -m unittest tests.test_adaptive_regime_walk_forward.InputValidationTests -v`

Expected: все проверки завершаются `OK`.

- [ ] **Step 6: Зафиксировать настройки и документ**

```powershell
git add -- rts/settings.yaml mix/settings.yaml ng/settings.yaml br/settings.yaml gold/settings.yaml si/settings.yaml spyf/settings.yaml docs/adaptive_momentum_countertrend_walk_forward_strategy.md tests/test_adaptive_regime_walk_forward.py
git commit -m "Настроить адаптивный walk-forward для семи тикеров"
```

Перед коммитом проверить `git diff` и включить из `settings.yaml` только новый блок исследования вместе с уже запрошенным пользователем `start_date_test: '2025-11-01'`; не включать `AGENTS.md` и `.tmp/`.

### Task 8: Реальная проверка и защита текущего контура

**Files:**
- Generated: `rts/research/adaptive_regime_walk_forward/results.xlsx`
- Generated: `rts/research/adaptive_regime_walk_forward/report.html`
- Generated: аналогичные каталоги для доступных тикеров
- Generated: `research/adaptive_regime_walk_forward_summary.xlsx`

- [ ] **Step 1: Зафиксировать исходное состояние защищённых файлов**

Run: `git status --short`

Expected: только известные пользовательские файлы и реализованные изменения; сохранить список для сравнения.

- [ ] **Step 2: Запустить полный автоматический набор**

Run: `.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: `OK`, без skipped и errors.

- [ ] **Step 3: Запустить реальный walk-forward для семи тикеров**

Run: `.venv\Scripts\python.exe run_adaptive_regime_walk_forward.py --all`

Expected: обработка всех тикеров продолжается даже при ошибке одного; процесс возвращает `0`, если есть хотя бы один `success`; создаётся общая сводка.

- [ ] **Step 4: Проверить структуру созданных книг программно**

Run:

```powershell
.venv\Scripts\python.exe -c "from pathlib import Path; import pandas as pd; p=Path('research/adaptive_regime_walk_forward_summary.xlsx'); x=pd.read_excel(p); print(x[['ticker','status','total_pl','max_drawdown']].to_string(index=False)); assert len(x)==7; assert (x.status=='success').any()"
```

Expected: семь строк, минимум один `success`, конечные метрики успешных тикеров.

- [ ] **Step 5: Проверить арифметическую согласованность каждого успешного тикера**

Для каждого успешного `results.xlsx` проверить:

```python
daily = pd.read_excel(path, sheet_name="daily")
assert abs(daily["strategy_pl"].sum() - daily["cumulative_pl"].iloc[-1]) < 1e-9
assert (daily["drawdown"] >= 0.0).all()
assert set(daily["applied_mode"]) <= {"momentum", "countertrend", "flat"}
```

- [ ] **Step 6: Убедиться, что текущий контур не изменён запуском**

Run: `git status --short`

Expected: запуск добавил только исследовательские отчёты; `simulate_trade.py`, прогнозы, `run_enb_sim_an.py`, `run_all.py` и торговые скрипты отсутствуют в diff.

- [ ] **Step 7: Выполнить финальную проверку синтаксиса**

Run: `.venv\Scripts\python.exe -m compileall -q research run_adaptive_regime_walk_forward.py tests`

Expected: код возврата `0`, без вывода ошибок.

- [ ] **Step 8: Зафиксировать только необходимые финальные правки, если проверка их потребовала**

```powershell
git add -- research run_adaptive_regime_walk_forward.py tests docs rts/settings.yaml mix/settings.yaml ng/settings.yaml br/settings.yaml gold/settings.yaml si/settings.yaml spyf/settings.yaml
git commit -m "Завершить проверку адаптивного walk-forward теста"
```

Не создавать пустой коммит, если финальная проверка не потребовала изменений. Сгенерированные Excel/HTML добавлять в Git только если это уже принято для исследовательских артефактов проекта; иначе оставить их локальными результатами запуска.
