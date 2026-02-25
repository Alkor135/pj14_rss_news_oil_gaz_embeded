# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TradeBot для торговли фьючерсами (RTS, природный газ) на основе эмбеддингов новостей. Система фильтрует новости по ключевым словам "нефт"/"газ" (провайдеры: Investing, Prime, Interfax), генерирует векторные представления через Ollama, ищет похожие исторические дни и предсказывает направление следующего дня. Интеграция с торговой платформой QUIK через `.tri`-файлы.

## Running Scripts

```bash
# 1. Создать markdown-файлы из БД новостей
python rts/create_markdown_files.py

# 2. Создать/обновить кэш эмбеддингов (требует Ollama на localhost:11434)
python rts/create_embedding.py

# 3. Бэктест торговой стратегии
python rts/simulate_trade.py

# 4. Анализ заголовков (что даёт прибыль/убыток)
python rts/analyze_explain.py

# 5. Просмотр содержимого кэша эмбеддингов
python rts/check_pkl.py

# 6. Исполнение сделок в QUIK (живая торговля)
python trade/trade_rts_tri.py   # RTS фьючерсы
python trade/trade_ng_tri.py    # Природный газ
python trade/trade_mix_tri.py   # Смешанный
```

## Architecture & Data Flow

```
News DBs (SQLite)  →  create_markdown_files.py  →  Daily .md files
                                                       ↓
Quote DBs (SQLite)           create_embedding.py  →  embeddings_ollama.pkl
                                     ↓ (Ollama API)
                          simulate_trade.py  →  P/L charts + explain_topk_all.pkl
                                ↓                          ↓
                    trade_*_tri.py              analyze_explain.py
                          ↓
                   QUIK (input.tri)  →  Live Trading
```

### Pipeline Details

**Stage 1 — Markdown generation** (`create_markdown_files.py`):
- Читает `rss_news_*.db` из `db_news_dir`, берёт последние `num_dbs=8` файлов
- Фильтрует по провайдерам и ключевым словам "нефт"/"газ"
- Торговой день: 21:00 предыдущего дня — 20:59:59 текущего
- Сохраняет в `md_path` как `YYYY-MM-DD.md`

**Stage 2 — Embedding cache** (`create_embedding.py`):
- Проверяет MD5 каждого .md файла, пропускает неизменённые
- Разбивает на чанки по параграфам (`\n\n`) с учётом лимита токенов
- Запрашивает эмбеддинги у Ollama, нормализует L2
- Хранит в pickle: `{TRADEDATE, MD5_hash, CHUNKS=[{chunk_id, tokens, text, embedding}]}`

**Stage 3 — Simulation** (`simulate_trade.py`):
- Для каждой даты перебирает окна `k=3..30` дней
- Находит наиболее похожий день (топ-5 чанков, средняя косинусная схожесть)
- Сигнал = ±|NEXT_BODY| в зависимости от совпадения направления
- Выбирает лучшее `k` по скользящей P/L за `test_days=23` дня

**Stage 4 — Execution** (`trade/trade_rts_tri.py` и др.):
- Читает файл предсказания текущего дня (`YYYY-MM-DD.txt`)
- Сравнивает с предыдущим предсказанием, формирует QUIK-транзакции
- Дописывает блок в `C:\QUIK_VTB_2025_ЕБС\algotrade\input.tri`

## Configuration (`rts/settings.yaml`)

Все пути и параметры централизованы в `settings.yaml`. Ключевые настройки:

| Параметр | Назначение |
|---|---|
| `ticker` | Инструмент (RTS / NG) |
| `model_name` | Модель эмбеддингов: `embeddinggemma` (default), `bge-m3`, `qwen3-embedding:0.6b` |
| `provider` | Источники новостей (`investing_prime_interfax`) |
| `url_ai` | Ollama API: `http://localhost:11434/api/embeddings` |
| `md_path` | Папка с markdown-файлами новостей |
| `cache_file` | Путь к pickle-кэшу эмбеддингов |
| `start_date_test` | Начало периода бэктеста (`2025-10-01`) |
| `test_days` | Скользящее окно оценки P/L (23 дня) |

## External Dependencies

**Сервисы:**
- **Ollama** на `localhost:11434` — должен быть запущен с нужной моделью эмбеддингов

**Внешние пути (Windows):**
- Новости: `C:/Users/Alkor/gd/db_rss/rss_news_*.db`
- Котировки: `C:/Users/Alkor/gd/data_quote_db/{ticker}_futures_*.db`
- Markdown: `C:/Users/Alkor/gd/md_rss_all_oil_gaz/`
- Предсказания: `C:/Users/Alkor/gd/predict_ai/{ticker_lc}_all_oil_gaz_gemma/`
- QUIK: `C:\QUIK_VTB_2025_ЕБС\algotrade\input.tri`

**Python-библиотеки:** `pandas`, `numpy`, `sqlite3`, `pickle`, `yaml`, `chromadb`, `tiktoken`, `matplotlib`

## Logs & Output

- Логи: `rts/log/` — автоматически, хранятся 3 последних на каждый скрипт
- Графики: `rts/plots/{model_name}_{provider}_{timestamp}.png`
- Данные анализа: `explain_topk_all.pkl` (детали совпадений для `analyze_explain.py`)
- Транзакции QUIK: `.tri`-файлы в `trade/`
