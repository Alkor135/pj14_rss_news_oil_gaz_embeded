"""Формирует Excel и HTML для адаптивного walk-forward исследования.

Примеры запуска через общий сценарий:
    python run_adaptive_regime_walk_forward.py --ticker rts
    python run_adaptive_regime_walk_forward.py --all
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research.adaptive_regime_walk_forward import calculate_max_drawdown


def calculate_metrics(
    daily_pl: pd.Series,
    active: pd.Series,
) -> dict[str, float | int]:
    """Считает сопоставимые метрики одной кривой P/L."""
    values = daily_pl.astype(float)
    active_mask = active.astype(bool)
    active_values = values[active_mask]
    gains = float(active_values[active_values > 0].sum())
    losses = abs(float(active_values[active_values < 0].sum()))
    if losses > 0.0:
        profit_factor = gains / losses
    elif gains > 0.0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0
    return {
        "total_pl": float(values.sum()),
        "max_drawdown": calculate_max_drawdown(values),
        "profit_factor": profit_factor,
        "win_rate": (
            float((active_values > 0).mean()) if len(active_values) else 0.0
        ),
        "active_days": int(active_mask.sum()),
    }


def build_summary(daily: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Сравнивает адаптивную стратегию с двумя постоянными режимами."""
    all_days = pd.Series(True, index=daily.index)
    rows = {
        "adaptive": calculate_metrics(
            daily["strategy_pl"], daily["applied_mode"].ne("flat")
        ),
        "countertrend": calculate_metrics(daily["countertrend_pl"], all_days),
        "momentum": calculate_metrics(daily["momentum_pl"], all_days),
    }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.insert(0, "ticker", ticker)
    result["mode_share_momentum"] = [
        float(daily["applied_mode"].eq("momentum").mean()),
        0.0,
        1.0,
    ]
    result["mode_share_countertrend"] = [
        float(daily["applied_mode"].eq("countertrend").mean()),
        1.0,
        0.0,
    ]
    result["mode_share_flat"] = [
        float(daily["applied_mode"].eq("flat").mean()),
        0.0,
        0.0,
    ]
    result["switches"] = [
        sum(
            left != right
            for left, right in zip(
                daily["applied_mode"], daily["applied_mode"].iloc[1:]
            )
        ),
        0,
        0,
    ]
    return result


def write_results_workbook(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    config: Mapping[str, object],
    output_dir: Path,
) -> Path:
    """Записывает дневные данные, метрики и снимок настроек в Excel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.xlsx"
    config_frame = pd.DataFrame(
        [
            {"parameter": key, "value": repr(value)}
            for key, value in config.items()
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        daily.to_excel(writer, sheet_name="daily", index=False)
        summary.rename_axis("strategy").reset_index().to_excel(
            writer, sheet_name="summary", index=False
        )
        config_frame.to_excel(writer, sheet_name="config", index=False)
    return path


def _summary_table(summary: pd.DataFrame) -> go.Table:
    """Создаёт Plotly-таблицу основных сравнительных метрик."""
    table = summary.rename_axis("strategy").reset_index()
    columns = [
        "strategy",
        "total_pl",
        "max_drawdown",
        "profit_factor",
        "win_rate",
        "active_days",
        "switches",
    ]
    display = table.loc[:, columns].copy()
    for column in ("total_pl", "max_drawdown", "profit_factor", "win_rate"):
        display[column] = display[column].map(
            lambda value: f"{value:.3f}" if pd.notna(value) else ""
        )
    return go.Table(
        header={"values": columns, "fill_color": "#dce6f1", "align": "left"},
        cells={"values": [display[column] for column in columns], "align": "left"},
    )


def write_html_report(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    ticker: str,
    output_dir: Path,
) -> Path:
    """Создаёт самодостаточный интерактивный Plotly-отчёт."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = daily.copy()
    data["countertrend_cumulative"] = data["countertrend_pl"].cumsum()
    data["momentum_cumulative"] = data["momentum_pl"].cumsum()
    mode_y = data["applied_mode"].map(
        {"flat": 0, "countertrend": 1, "momentum": 2}
    )
    mode_color = data["applied_mode"].map(
        {"flat": "#7f8c8d", "countertrend": "#c0392b", "momentum": "#27ae60"}
    )
    fig = make_subplots(
        rows=6,
        cols=1,
        vertical_spacing=0.04,
        specs=[
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "table"}],
        ],
        subplot_titles=(
            "Накопленный P/L",
            "Дневной P/L адаптивной стратегии",
            "Просадка",
            "Режим",
            "Подобранные параметры",
            "Итоговые метрики",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=data["TRADEDATE"], y=data["cumulative_pl"], name="Adaptive"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["TRADEDATE"],
            y=data["countertrend_cumulative"],
            name="Countertrend",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["TRADEDATE"], y=data["momentum_cumulative"], name="Momentum"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=data["TRADEDATE"], y=data["strategy_pl"], name="Adaptive daily"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["TRADEDATE"],
            y=-data["drawdown"],
            fill="tozeroy",
            name="Drawdown",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["TRADEDATE"],
            y=mode_y,
            mode="markers",
            marker={"color": mode_color, "size": 7},
            text=data["applied_mode"],
            name="Regime",
        ),
        row=4,
        col=1,
    )
    for column, name in (
        ("score_span", "Score span"),
        ("confidence_threshold", "Threshold"),
        ("min_regime_days", "Minimum days"),
    ):
        fig.add_trace(
            go.Scatter(
                x=data["TRADEDATE"],
                y=pd.to_numeric(data[column], errors="coerce"),
                name=name,
            ),
            row=5,
            col=1,
        )
    fig.add_trace(_summary_table(summary), row=6, col=1)
    adaptive = summary.loc["adaptive"]
    fig.update_layout(
        title=(
            f"{ticker} — Adaptive Regime Walk-Forward | "
            f"P/L {adaptive['total_pl']:.0f} | "
            f"MaxDD {adaptive['max_drawdown']:.0f}"
        ),
        template="plotly_white",
        height=1750,
        hovermode="x unified",
    )
    fig.update_yaxes(
        tickvals=[0, 1, 2],
        ticktext=["flat", "countertrend", "momentum"],
        row=4,
        col=1,
    )
    path = output_dir / "report.html"
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return path


def write_global_summary(
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> Path:
    """Записывает статусы и метрики всех запрошенных тикеров."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "ticker",
        "status",
        "message",
        "total_pl",
        "max_drawdown",
        "output_dir",
    ]
    pd.DataFrame(rows).reindex(columns=columns).to_excel(path, index=False)
    return path
