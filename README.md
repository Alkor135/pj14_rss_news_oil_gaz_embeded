# pj14_rss_news_oil_gaz_embeded

TradeBot для торговли фьючерсами (RTS, MIX, NG, BR, GOLD, Si, SPYF) на основе эмбеддингов новостей.

## Принцип работы

Система фильтрует RSS-новости по ключевым словам "нефт"/"газ" (провайдеры: Investing, 1Prime, Interfax), генерирует векторные представления через Ollama, ищет похожие исторические дни по косинусной схожести и предсказывает направление следующего торгового дня. Интеграция с торговой платформой QUIK через `.tri`-файлы.

## Пайплайн

```
Beget (RSS сбор) → sync → download_minutes → convert_to_days → create_markdown
    → create_embedding (Ollama) → simulate_trade → strategy_analysis → trade (QUIK)
```

## Rolling walk-forward симуляционная торговля

Скрипты `{ticker}/simulate_trade.py` проводят бэктест без переобучения модели: эмбеддинги новостей уже рассчитаны, а walk-forward применяется к выбору окна похожих исторических дней `k`.

1. Скрипт читает настройки из `{ticker}/settings.yaml`: `ticker`, `model_name`, `provider`, `start_date_test`, `test_days`, пути к дневной БД котировок и `embeddings_ollama.pkl`.
2. Из дневной БД берется `OPEN` и считается будущий результат `NEXT_OPEN_TO_OPEN = OPEN.shift(-2) - OPEN.shift(-1)`.
3. Котировки объединяются с новостными чанками `CHUNKS` из `embeddings_ollama.pkl` по `TRADEDATE`.
4. Для каждого `k` от `3` до `30` текущие новости сравниваются с новостями предыдущих `k` дней. Сходство считается как cosine similarity по чанкам с усреднением top-5 совпадений.
5. Для текущей даты выбирается самый похожий прошлый день. Если знак его `NEXT_OPEN_TO_OPEN` совпал с фактическим знаком текущего дня, в `MAX_k` пишется `+abs(P/L)`, иначе `-abs(P/L)`.
6. На каждой новой дате выбирается лучшее `k` по скользящей сумме прошлых результатов: `PL_k = MAX_k.shift(1).rolling(window=test_days).sum()`. `shift(1)` исключает текущий день из выбора параметра, поэтому выбор `k` идет только по уже известной истории.
7. В итоговый `df_rez_output.xlsx` записываются `TRADEDATE`, дневной `P/L` для выбранного `k` и само выбранное окно `max`. После расчета P/L инвертируется (`P/L *= -1`), поэтому файл содержит уже зеркальную версию стратегии.

При `test_days: 23` в настройках код использует окно `settings["test_days"] + 1`, то есть фактически 24 строки истории.

Дополнительно `simulate_trade.py` сохраняет `{ticker}/explain_topk_all.pkl` с деталями похожих чанков, PNG-график cumulative P/L и выбранного `k`, а также файл прогноза следующей сессии в `predict_path`.

## Отчеты strategy_analysis.html

Скрипты `{ticker}/strategy_analysis.py` не пересчитывают стратегию. Они читают готовый `{ticker}/df_rez_output.xlsx`, строят интерактивный Plotly-отчет и сохраняют его в `{ticker}/plots/strategy_analysis.html`.

В HTML-отчет входят:

- дневной P/L и накопленная прибыль;
- недельная, месячная и годовая агрегация P/L;
- распределение дневных результатов;
- drawdown от максимума накопленной прибыли;
- скользящие средние дневного P/L за 7, 14 и 30 дней;
- выбранное walk-forward окно `max` (`Best Window (k)`) по дням;
- скользящий Recovery Factor;
- таблица статистики стратегии и таблица ключевых коэффициентов.

Основные метрики отчета: чистая прибыль, Max Drawdown, Profit Factor, Recovery Factor, Payoff Ratio, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Expectancy, win rate, серии прибыльных/убыточных дней и число прибыльных месяцев.

## Запуск

```bash
# Весь пайплайн (через Планировщик задач)
python run_all.py

# Только аналитика (embedding → simulate → analysis)
python run_enb_sim_an.py

# Открыть HTML-отчёты в Chrome
python html_open.py
```

## Структура

| Папка | Назначение |
| --- | --- |
| `rts/`, `mix/`, `ng/`, `br/`, `gold/`, `si/`, `spyf/` | Пайплайн для каждого тикера (settings.yaml + скрипты) |
| `trade/` | Исполнение сделок в QUIK (RTS, MIX, NG) |
| `beget/` | Сбор RSS-новостей на сервере + синхронизация |
| `buhinvest_analize/` | Анализ реальной доходности торговли |

## Зависимости

- **Python 3.10+** с библиотеками из `requirements.txt`
- **Ollama** на `localhost:11434` с моделью эмбеддингов
- **QUIK** (для живой торговли)

Подробная документация: [CLAUDE.md](CLAUDE.md)
