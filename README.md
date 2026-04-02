# pj14_rss_news_oil_gaz_embeded

TradeBot для торговли фьючерсами (RTS, MIX, NG, BR, GOLD, Si, SPYF) на основе эмбеддингов новостей.

## Принцип работы

Система фильтрует RSS-новости по ключевым словам "нефт"/"газ" (провайдеры: Investing, 1Prime, Interfax), генерирует векторные представления через Ollama, ищет похожие исторические дни по косинусной схожести и предсказывает направление следующего торгового дня. Интеграция с торговой платформой QUIK через `.tri`-файлы.

## Пайплайн

```
Beget (RSS сбор) → sync → download_minutes → convert_to_days → create_markdown
    → create_embedding (Ollama) → simulate_trade → strategy_analysis → trade (QUIK)
```

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
