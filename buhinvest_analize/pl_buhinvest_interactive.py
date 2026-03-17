"""
Интерактивные графики доходности из файла Buhinvest (Plotly).
Генерирует HTML-файл с 10 панелями: дневной P/L, накопительная прибыль,
месячный/недельный P/L, баланс, доходность %, гистограмма P/L, drawdown,
скользящие средние P/L, таблица ключевых метрик стратегии.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

SAVE_PATH = Path(__file__).parent
FILE_PATH = r"C:\Users\Alkor\gd\buhinvest_futures_RTS_MIX_full.xlsx"

# ── Загрузка данных ──────────────────────────────────────────────────────
df = pd.read_excel(
    FILE_PATH, sheet_name="Data",
    usecols=[
        "Дата", "Вводы", "Всего на счетах",
        "Общ. прибыль Руб.", "Общ. прибыль %",
        "Profit/Loss к предыдущему", "Доходность змейкой %",
        "% годовых", "XIRR %", "За месяц",
    ],
)

df["Дата"] = pd.to_datetime(df["Дата"])
df = df.dropna(subset=["Дата"]).sort_values("Дата").reset_index(drop=True)

for col in ["Profit/Loss к предыдущему", "Общ. прибыль Руб.", "Общ. прибыль %",
            "Всего на счетах", "% годовых", "XIRR %", "Доходность змейкой %"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Profit/Loss к предыдущему"] = df["Profit/Loss к предыдущему"].fillna(0)
pl = df["Profit/Loss к предыдущему"]

# ── Производные данные ───────────────────────────────────────────────────
# Цвета дневных баров
day_colors = ["#d32f2f" if v < 0 else "#2e7d32" for v in pl]

# Месячная агрегация
df["Месяц"] = df["Дата"].dt.to_period("M")
monthly = df.groupby("Месяц", as_index=False)["Profit/Loss к предыдущему"].sum()
monthly["dt"] = monthly["Месяц"].dt.to_timestamp()
monthly.rename(columns={"Profit/Loss к предыдущему": "PL"}, inplace=True)
month_colors = ["#d32f2f" if v < 0 else "#1565c0" for v in monthly["PL"]]

# Недельная агрегация
df["Неделя"] = df["Дата"].dt.to_period("W")
weekly = df.groupby("Неделя", as_index=False)["Profit/Loss к предыдущему"].sum()
weekly["dt"] = weekly["Неделя"].apply(lambda p: p.start_time)
weekly.rename(columns={"Profit/Loss к предыдущему": "PL"}, inplace=True)
week_colors = ["#d32f2f" if v < 0 else "#00838f" for v in weekly["PL"]]

# Drawdown от максимума накопленной прибыли
cum = df["Общ. прибыль Руб."].fillna(0)
running_max = cum.cummax()
drawdown = cum - running_max  # всегда <= 0

# Скользящие средние дневного P/L
for w in (7, 14, 30):
    df[f"MA{w}"] = pl.rolling(w, min_periods=1).mean()

# ── Метрики стратегии ─────────────────────────────────────────────────────
total_profit = df["Общ. прибыль Руб."].iloc[-1]
total_days = len(df)
win_days = int((pl > 0).sum())
loss_days = int((pl < 0).sum())
zero_days = int((pl == 0).sum())
trade_days = win_days + loss_days
win_rate = win_days / max(trade_days, 1) * 100
max_dd = drawdown.min()
best_day = pl.max()
worst_day = pl.min()
avg_day = pl.mean()
median_day = pl.median()
std_day = pl.std()

# Gross profit / gross loss
gross_profit = pl[pl > 0].sum()
gross_loss = abs(pl[pl < 0].sum())
avg_win = pl[pl > 0].mean() if win_days else 0
avg_loss = abs(pl[pl < 0].mean()) if loss_days else 0

# Profit Factor = Gross Profit / Gross Loss
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

# Payoff Ratio = Avg Win / Avg Loss
payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

# Recovery Factor = Net Profit / |Max Drawdown|
recovery_factor = total_profit / abs(max_dd) if max_dd != 0 else float("inf")

# Expectancy (мат. ожидание на сделку)
expectancy = (win_rate / 100) * avg_win - (1 - win_rate / 100) * avg_loss

# Sharpe Ratio (annualized, Rf=0, ~252 торговых дня)
sharpe = (avg_day / std_day) * np.sqrt(252) if std_day > 0 else 0

# Sortino Ratio (только downside deviation)
downside = pl[pl < 0]
downside_std = downside.std() if len(downside) > 1 else 0
sortino = (avg_day / downside_std) * np.sqrt(252) if downside_std > 0 else 0

# Calmar Ratio = Annualized Return / |Max Drawdown|
date_range_days = (df["Дата"].max() - df["Дата"].min()).days or 1
annual_profit = total_profit * 365 / date_range_days
calmar = annual_profit / abs(max_dd) if max_dd != 0 else float("inf")

# Max consecutive wins / losses
def max_consecutive(series, condition):
    streaks = (series != condition).cumsum()
    filtered = series[series == condition]
    if filtered.empty:
        return 0
    return filtered.groupby(streaks[series == condition]).size().max()

signs = pl.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
max_consec_wins = max_consecutive(signs, 1)
max_consec_losses = max_consecutive(signs, -1)

# Длительность максимальной просадки (дней)
dd_start = None
max_dd_duration = 0
current_dd_start = None
for i in range(len(drawdown)):
    if drawdown.iloc[i] < 0:
        if current_dd_start is None:
            current_dd_start = i
    else:
        if current_dd_start is not None:
            duration = i - current_dd_start
            if duration > max_dd_duration:
                max_dd_duration = duration
            current_dd_start = None
if current_dd_start is not None:
    duration = len(drawdown) - current_dd_start
    if duration > max_dd_duration:
        max_dd_duration = duration

# Volatility (annualized)
volatility = std_day * np.sqrt(252)

# Win months / loss months
win_months = int((monthly["PL"] > 0).sum())
loss_months = int((monthly["PL"] < 0).sum())

# Краткая строка для заголовка
stats_text = (
    f"Итого: {total_profit:,.0f} ₽ | Дней: {total_days} | "
    f"Win: {win_days} ({win_rate:.0f}%) | Loss: {loss_days} | "
    f"PF: {profit_factor:.2f} | RF: {recovery_factor:.2f} | "
    f"Sharpe: {sharpe:.2f} | MaxDD: {max_dd:,.0f}"
)

# Таблица коэффициентов с расшифровками
coefficients = [
    {
        "name": "Recovery Factor",
        "formula": "Чистая прибыль / |Max Drawdown|",
        "value": f"{recovery_factor:.2f}",
        "description": (
            "Коэффициент восстановления — показывает, во сколько раз чистая прибыль "
            "превышает максимальную просадку. Отражает способность стратегии "
            "восстанавливаться после убытков. RF > 1 означает, что стратегия "
            "заработала больше, чем потеряла в худший период. Чем выше — тем лучше. "
            "RF < 1 — стратегия ещё не отбила свою максимальную просадку."
        ),
    },
    {
        "name": "Profit Factor",
        "formula": "Валовая прибыль / Валовый убыток",
        "value": f"{profit_factor:.2f}",
        "description": (
            "Фактор прибыли — отношение суммы всех прибыльных дней к сумме всех "
            "убыточных (по модулю). PF > 1 означает, что стратегия прибыльна в целом. "
            "PF = 1.5 значит: на каждый потерянный рубль стратегия зарабатывает 1.5 ₽. "
            "Значения 1.5–2.0 считаются хорошими, > 2.0 — отличными. "
            "PF < 1 — стратегия убыточна."
        ),
    },
    {
        "name": "Payoff Ratio",
        "formula": "Средний выигрыш / Средний проигрыш",
        "value": f"{payoff_ratio:.2f}",
        "description": (
            "Коэффициент выплат — отношение среднего размера прибыльной сделки к среднему "
            "размеру убыточной. Показывает, сколько в среднем зарабатывает стратегия "
            "на одной прибыльной сделке относительно одной убыточной. "
            "Payoff > 1 означает, что средний выигрыш больше среднего проигрыша. "
            "Даже при win rate < 50% стратегия может быть прибыльной при высоком Payoff."
        ),
    },
    {
        "name": "Sharpe Ratio",
        "formula": "(Средний дневной P/L / Std дневного P/L) × √252",
        "value": f"{sharpe:.2f}",
        "description": (
            "Коэффициент Шарпа — отношение доходности к риску (волатильности), "
            "приведённое к годовому масштабу (252 торговых дня). Безрисковая ставка = 0. "
            "Sharpe > 1.0 — хорошая доходность с учётом риска, > 2.0 — отличная, "
            "> 3.0 — исключительная. Sharpe < 0 — стратегия убыточна. "
            "Учитывает и положительные, и отрицательные отклонения."
        ),
    },
    {
        "name": "Sortino Ratio",
        "formula": "(Средний дневной P/L / Downside Std) × √252",
        "value": f"{sortino:.2f}",
        "description": (
            "Коэффициент Сортино — модификация Шарпа, учитывающая только нисходящую "
            "волатильность (убыточные дни). В отличие от Шарпа, не штрафует за "
            "положительные всплески доходности. Более справедливая оценка для стратегий "
            "с редкими крупными выигрышами. Интерпретация аналогична Шарпу: > 1 хорошо, "
            "> 2 отлично. Обычно Sortino > Sharpe."
        ),
    },
    {
        "name": "Calmar Ratio",
        "formula": "Годовая доходность / |Max Drawdown|",
        "value": f"{calmar:.2f}",
        "description": (
            "Коэффициент Калмара — отношение годовой прибыли к максимальной просадке. "
            "Похож на Recovery Factor, но нормализован к году. Показывает, сколько "
            "годовой прибыли приходится на единицу максимального риска. "
            "Calmar > 1 — годовая прибыль превышает худшую просадку. "
            "Calmar > 3 — отличное соотношение доходности и риска."
        ),
    },
    {
        "name": "Expectancy",
        "formula": "Win% × Ср.выигрыш − Loss% × Ср.проигрыш",
        "value": f"{expectancy:,.0f} ₽",
        "description": (
            "Математическое ожидание — средняя прибыль на одну сделку с учётом "
            "вероятности выигрыша и проигрыша. Положительное значение означает, что "
            "стратегия прибыльна в долгосрочной перспективе. Это ключевой показатель: "
            "если Expectancy > 0, стратегия имеет положительное преимущество (edge). "
            "Чем больше — тем сильнее преимущество над рынком."
        ),
    },
]

# ── Построение графиков ──────────────────────────────────────────────────
fig = make_subplots(
    rows=5, cols=2,
    subplot_titles=(
        "P/L по дням (руб.)",
        "Накопительная прибыль (руб.)",
        "P/L по месяцам (руб.)",
        "P/L по неделям (руб.)",
        "Баланс на счетах (руб.)",
        "Доходность: XIRR % и % годовых",
        "Распределение дневных P/L",
        "Drawdown от максимума (руб.)",
        "Скользящие средние P/L (7/14/30 дней)",
        "Recovery Factor (скользящий)",
    ),
    specs=[
        [{"type": "bar"}, {"type": "scatter"}],
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "histogram"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "scatter"}],
    ],
    vertical_spacing=0.06,
    horizontal_spacing=0.06,
)

# 1) Дневной P/L — бары
fig.add_trace(
    go.Bar(
        x=df["Дата"], y=pl, marker_color=day_colors,
        name="P/L день",
        hovertemplate="%{x|%Y-%m-%d}<br>P/L: %{y:,.0f} ₽<extra></extra>",
    ),
    row=1, col=1,
)

# 2) Накопительная прибыль
fig.add_trace(
    go.Scatter(
        x=df["Дата"], y=cum,
        mode="lines", fill="tozeroy",
        line=dict(color="#2e7d32", width=2),
        fillcolor="rgba(46,125,50,0.15)",
        name="Накопл. прибыль",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} ₽<extra></extra>",
    ),
    row=1, col=2,
)

# 3) Месячный P/L
fig.add_trace(
    go.Bar(
        x=monthly["dt"], y=monthly["PL"], marker_color=month_colors,
        name="P/L месяц",
        text=[f"{v:,.0f}" for v in monthly["PL"]],
        textposition="outside",
        hovertemplate="%{x|%Y-%m}<br>P/L: %{y:,.0f} ₽<extra></extra>",
    ),
    row=2, col=1,
)

# 4) Недельный P/L
fig.add_trace(
    go.Bar(
        x=weekly["dt"], y=weekly["PL"], marker_color=week_colors,
        name="P/L неделя",
        hovertemplate="Нед. %{x|%Y-%m-%d}<br>P/L: %{y:,.0f} ₽<extra></extra>",
    ),
    row=2, col=2,
)

# 5) Баланс на счетах
fig.add_trace(
    go.Scatter(
        x=df["Дата"], y=df["Всего на счетах"],
        mode="lines+markers",
        line=dict(color="#1565c0", width=2),
        marker=dict(size=3),
        name="Баланс",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} ₽<extra></extra>",
    ),
    row=3, col=1,
)
# Отметить вводы
deposits = df[df["Вводы"].notna()]
if not deposits.empty:
    fig.add_trace(
        go.Scatter(
            x=deposits["Дата"], y=deposits["Всего на счетах"],
            mode="markers+text",
            marker=dict(size=12, color="#ff6f00", symbol="triangle-up"),
            text=[f"Ввод {v:,.0f}" for v in deposits["Вводы"]],
            textposition="top center",
            name="Вводы",
            hovertemplate="%{x|%Y-%m-%d}<br>Ввод: %{text}<extra></extra>",
        ),
        row=3, col=1,
    )

# 6) Доходность %
fig.add_trace(
    go.Scatter(
        x=df["Дата"], y=df["XIRR %"] * 100,
        mode="lines", line=dict(color="#7b1fa2", width=1.5),
        name="XIRR %",
        hovertemplate="%{x|%Y-%m-%d}<br>XIRR: %{y:.1f}%<extra></extra>",
    ),
    row=3, col=2,
)
fig.add_trace(
    go.Scatter(
        x=df["Дата"], y=df["% годовых"] * 100,
        mode="lines", line=dict(color="#e65100", width=1.5),
        name="% годовых",
        hovertemplate="%{x|%Y-%m-%d}<br>% годовых: %{y:.1f}%<extra></extra>",
    ),
    row=3, col=2,
)

# 7) Гистограмма распределения дневного P/L
pl_pos = pl[pl > 0]
pl_neg = pl[pl < 0]
fig.add_trace(
    go.Histogram(
        x=pl_pos, marker_color="#2e7d32", opacity=0.7,
        name="Прибыль", nbinsx=30,
        hovertemplate="P/L: %{x:,.0f}<br>Кол-во: %{y}<extra></extra>",
    ),
    row=4, col=1,
)
fig.add_trace(
    go.Histogram(
        x=pl_neg, marker_color="#d32f2f", opacity=0.7,
        name="Убыток", nbinsx=30,
        hovertemplate="P/L: %{x:,.0f}<br>Кол-во: %{y}<extra></extra>",
    ),
    row=4, col=1,
)

# 8) Drawdown
fig.add_trace(
    go.Scatter(
        x=df["Дата"], y=drawdown,
        mode="lines", fill="tozeroy",
        line=dict(color="#d32f2f", width=1.5),
        fillcolor="rgba(211,47,47,0.2)",
        name="Drawdown",
        hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:,.0f} ₽<extra></extra>",
    ),
    row=4, col=2,
)

# 9) Скользящие средние P/L
for w, color in [(7, "#1565c0"), (14, "#ff6f00"), (30, "#7b1fa2")]:
    fig.add_trace(
        go.Scatter(
            x=df["Дата"], y=df[f"MA{w}"],
            mode="lines", line=dict(color=color, width=1.5),
            name=f"MA{w}",
            hovertemplate=f"MA{w}: " + "%{y:,.0f}<extra></extra>",
        ),
        row=5, col=1,
    )
# Нулевая линия для MA
fig.add_hline(y=0, line_dash="dash", line_color="gray", row=5, col=1)

# 10) Recovery Factor (скользящий) — накопл. прибыль / |текущий MaxDD|
rf_rolling = pd.Series(dtype=float, index=df.index)
for i in range(len(df)):
    dd_so_far = (cum.iloc[:i+1] - cum.iloc[:i+1].cummax()).min()
    rf_rolling.iloc[i] = cum.iloc[i] / abs(dd_so_far) if dd_so_far != 0 else 0
fig.add_trace(
    go.Scatter(
        x=df["Дата"], y=rf_rolling,
        mode="lines", line=dict(color="#00695c", width=2),
        name="Recovery Factor",
        hovertemplate="%{x|%Y-%m-%d}<br>RF: %{y:.2f}<extra></extra>",
    ),
    row=5, col=2,
)
fig.add_hline(y=1, line_dash="dash", line_color="gray", row=5, col=2,
              annotation_text="RF=1")

# ── Оформление ───────────────────────────────────────────────────────────
fig.update_layout(
    height=2000,
    width=1500,
    title_text=f"Buhinvest Futures RTS+MIX — Анализ доходности<br><sub>{stats_text}</sub>",
    title_x=0.5,
    showlegend=True,
    legend=dict(orientation="h", yanchor="top", y=-0.07, xanchor="center", x=0.5),
    template="plotly_white",
    hovermode="x unified",
)

# Формат осей Y в рублях
for row, col in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (4, 2), (5, 1)]:
    fig.update_yaxes(tickformat=",", row=row, col=col)
fig.update_yaxes(ticksuffix="%", row=3, col=2)
fig.update_xaxes(tickformat="%Y-%m-%d", row=4, col=1, title_text="P/L (руб.)")
fig.update_yaxes(title_text="RF", row=5, col=2)

# ── Таблица статистики (6 колонок: Доходность | Риск | Статистика сделок) ──
sec1 = [  # ДОХОДНОСТЬ
    ["<b>ДОХОДНОСТЬ</b>", ""],
    ["Чистая прибыль", f"{total_profit:,.0f} ₽"],
    ["Годовая прибыль (экстрапол.)", f"{annual_profit:,.0f} ₽"],
    ["Средний P/L в день", f"{avg_day:,.0f} ₽"],
    ["Медианный P/L в день", f"{median_day:,.0f} ₽"],
    ["Лучший день", f"{best_day:,.0f} ₽"],
    ["Худший день", f"{worst_day:,.0f} ₽"],
]

sec2 = [  # РИСК
    ["<b>РИСК</b>", ""],
    ["Max Drawdown", f"{max_dd:,.0f} ₽"],
    ["Длит. макс. просадки", f"{max_dd_duration} дней"],
    ["Волатильность (год.)", f"{volatility:,.0f} ₽"],
    ["Std дневного P/L", f"{std_day:,.0f} ₽"],
    ["VaR 95%", f"{np.percentile(pl, 5):,.0f} ₽"],
    ["CVaR 95%", f"{pl[pl <= np.percentile(pl, 5)].mean():,.0f} ₽"],
]

sec3 = [  # СТАТИСТИКА СДЕЛОК
    ["<b>СТАТИСТИКА СДЕЛОК</b>", ""],
    ["Торговых дней", f"{total_days}"],
    ["Win / Loss / Zero", f"{win_days} / {loss_days} / {zero_days}"],
    ["Win rate", f"{win_rate:.1f}%"],
    ["Ср. выигрыш / проигрыш", f"{avg_win:,.0f} / {avg_loss:,.0f} ₽"],
    ["Макс. серия побед", f"{max_consec_wins}"],
    ["Макс. серия убытков", f"{max_consec_losses}"],
]

num_rows = max(len(sec1), len(sec2), len(sec3))
for sec in (sec1, sec2, sec3):
    while len(sec) < num_rows:
        sec.append(["", ""])

cols = [[], [], [], [], [], []]  # name1, val1, name2, val2, name3, val3
colors = [[], [], []]
for i in range(num_rows):
    for j, sec in enumerate((sec1, sec2, sec3)):
        n, v = sec[i]
        is_hdr = v == "" and n.startswith("<b>")
        cols[j * 2].append(n)
        cols[j * 2 + 1].append(f"<b>{v}</b>" if v and not is_hdr else v)
        if is_hdr:
            colors[j].append("#e3f2fd")
        else:
            colors[j].append("#f5f5f5" if i % 2 == 0 else "white")

fig_stats = go.Figure(
    go.Table(
        columnwidth=[200, 130, 180, 120, 200, 140],
        header=dict(
            values=["<b>Показатель</b>", "<b>Значение</b>"] * 3,
            fill_color="#1565c0",
            font=dict(color="white", size=14),
            align="left",
            height=32,
        ),
        cells=dict(
            values=cols,
            fill_color=[colors[0], colors[0], colors[1], colors[1],
                        colors[2], colors[2]],
            font=dict(size=13, color="#212121"),
            align=["left", "right", "left", "right", "left", "right"],
            height=26,
        ),
    )
)
table_height = 32 + num_rows * 26 + 80  # header + rows + margins
fig_stats.update_layout(
    title_text="<b>Статистика стратегии</b>",
    title_x=0.5,
    title_font_size=18,
    height=table_height,
    width=1500,
    margin=dict(l=20, r=20, t=60, b=20),
)

# ── Таблица коэффициентов (с расшифровками на русском) ────────────────────
fig_table = go.Figure(
    go.Table(
        columnwidth=[150, 250, 80, 450],
        header=dict(
            values=[
                "<b>Коэффициент</b>",
                "<b>Формула</b>",
                "<b>Значение</b>",
                "<b>Расшифровка</b>",
            ],
            fill_color="#1565c0",
            font=dict(color="white", size=14),
            align="left",
            height=36,
        ),
        cells=dict(
            values=[
                [f"<b>{c['name']}</b>" for c in coefficients],
                [c["formula"] for c in coefficients],
                [f"<b>{c['value']}</b>" for c in coefficients],
                [c["description"] for c in coefficients],
            ],
            fill_color=[
                ["#f5f5f5" if i % 2 == 0 else "white" for i in range(len(coefficients))]
            ] * 4,
            font=dict(size=13, color="#212121"),
            align=["left", "left", "center", "left"],
            height=80,
        ),
    )
)
fig_table.update_layout(
    title_text="<b>Ключевые коэффициенты торговой стратегии</b>",
    title_x=0.5,
    title_font_size=18,
    height=700,
    width=1500,
    margin=dict(l=20, r=20, t=60, b=20),
)

# ── Сохранение (три figure в один HTML) ──────────────────────────────────
output = SAVE_PATH / "pl_buhinvest_interactive.html"

charts_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
coeff_html = fig_table.to_html(include_plotlyjs=False, full_html=False)

with open(output, "w", encoding="utf-8") as f:
    f.write("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n")
    f.write("<title>Buhinvest — Анализ доходности</title>\n</head><body>\n")
    f.write(charts_html)
    f.write("\n<hr style='margin:30px 0; border:1px solid #ccc'>\n")
    f.write(fig_stats.to_html(include_plotlyjs=False, full_html=False))
    f.write("\n<hr style='margin:30px 0; border:1px solid #ccc'>\n")
    f.write(coeff_html)
    f.write("\n</body></html>")

print(f"Интерактивный отчёт сохранён: {output}")
