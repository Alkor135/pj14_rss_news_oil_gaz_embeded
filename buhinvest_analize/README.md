# Анализ доходности из Buhinvest (RUR)

Визуализация реальной доходности торговых операций из Excel-файла Buhinvest (в рублях).

Источник данных: `C:\Users\Alkor\gd\buhinvest_futures_RTS_MIX_full.xlsx` (лист `Data`).

## Скрипты

### `pl_buhinvest.py` — статические графики (Matplotlib)

- `pl_by_month.png` — столбчатый график ежемесячного P/L (синий — прибыль, красный — убыток)
- `cumulative_profit.png` — линейный график накопительной прибыли по дням

Используемые колонки Excel: `Дата`, `Profit/Loss к предыдущему`, `Общ. прибыль Руб.`.

### `pl_buhinvest_interactive.py` — интерактивный отчёт (Plotly)

Генерирует `pl_buhinvest_interactive.html` с 10 панелями:

- Дневной / недельный / месячный P/L
- Накопительная прибыль, баланс, доходность в процентах
- Распределение P/L (гистограмма)
- Drawdown
- Скользящие средние P/L
- Таблица ключевых метрик: Sharpe, Sortino, Calmar, Profit Factor, Recovery Factor, Expectancy

Открывается в `html_open.py` вместе со `strategy_analysis.html` каждого тикера.

## Запуск

```bash
python buhinvest_analize/pl_buhinvest.py
python buhinvest_analize/pl_buhinvest_interactive.py
```

Оба скрипта вызываются в конце `run_all.py` и `run_enb_sim_an.py`.

## Зависимости

- Python 3.10+
- `pandas`, `numpy`, `openpyxl`
- `matplotlib` (для `pl_buhinvest.py`)
- `plotly` (для `pl_buhinvest_interactive.py`)
