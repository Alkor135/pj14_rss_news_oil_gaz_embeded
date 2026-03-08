# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TradeBot для торговли фьючерсами (RTS, природный газ) на основе эмбеддингов новостей. Система фильтрует новости по ключевым словам "нефт"/"газ" (провайдеры: Investing, Prime, Interfax), генерирует векторные представления через Ollama, ищет похожие исторические дни и предсказывает направление следующего дня. Интеграция с торговой платформой QUIK через `.tri`-файлы.

## Running Scripts

```bash
# Запуск всего пайплайна последовательно (через Планировщик задач)
python run_all.py

# Или по отдельности:

# 0. Синхронизация БД новостей с сервера Beget
python beget/sync_files.py

# 0a. Скачать минутные свечи с MOEX ISS API в SQLite
python rts/download_minutes_to_db.py

# 0b. Конвертировать минутные бары в дневные (21:00–20:59:59)
python rts/convert_minutes_to_days.py

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
python trade/trade_ng_tri.py    # Тикер фьючерса на природный газ
python trade/trade_mix_tri.py   # Тикер фьючерса на индекс Мосбиржи
```

## Architecture & Data Flow

```text
MOEX ISS API  →  download_minutes_to_db.py  →  {ticker}_futures_minute_2025.db
                                                       ↓
                 convert_minutes_to_days.py  →  {ticker}_futures_day_2025_21-00.db
                                                       ↓
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

**Stage 0a — Download minute candles** (`download_minutes_to_db.py`):

- Скачивает минутные свечи с MOEX ISS API с пагинацией (по 500 записей)
- Определяет активный фьючерс по `assetcode` и `LSTTRADE` (ближайшая экспирация)
- Инкрементальная докачка: проверяет полноту данных за каждую дату, досчитывает недостающие
- Минутные данные за текущую сессию доступны на MOEX после 19:05

**Stage 0b — Convert to daily** (`convert_minutes_to_days.py`):

- Формирует дневные OHLC-свечи из минутных данных с границами 21:00–20:59:59
- При ролловере (два контракта в одном дне) корректирует цены старого контракта на gap
- Удаляет последнюю запись в БД при каждом запуске (пересчитывает крайний день)
- Добавляет только отсутствующие по дате записи

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
- Применяет инверсию P/L (`*= -1`) — стратегия торгует против предсказания
- Сохраняет explain-данные в `explain_topk_all.pkl` и результаты в `df_rez_output.xlsx`
- Делает предсказание на следующую сессию и сохраняет в `predict_path/{YYYY-MM-DD}.txt`

**Stage 4 — Execution** (`trade/trade_rts_tri.py` и др.):

- Читает файл предсказания текущего дня (`YYYY-MM-DD.txt`)
- Открывает позицию в противоположную сторону от предсказания (инверсия стратегии)
- Сравнивает с предыдущим предсказанием, формирует QUIK-транзакции
- Дописывает блок в `C:\QUIK_VTB_2025_ЕБС\algotrade\input.tri`
- Поддержка ролловера: когда `ticker_close != ticker_open`, выполняется переоткрытие позиции на новом контракте
- Тикеры и количества захардкожены в каждом скрипте и обновляются вручную при смене контракта

## Configuration (`rts/settings.yaml`)

Все пути и параметры централизованы в `settings.yaml`. Ключевые настройки:

| Параметр | Назначение |
| --- | --- |
| `ticker` | Инструмент (RTS / NG) |
| `model_name` | Модель эмбеддингов: `embeddinggemma` (default), `bge-m3`, `qwen3-embedding:0.6b` |
| `provider` | Источники новостей (`investing_prime_interfax`) |
| `url_ai` | Ollama API: `http://localhost:11434/api/embeddings` |
| `md_path` | Папка с markdown-файлами новостей |
| `cache_file` | Путь к pickle-кэшу эмбеддингов |
| `start_date_test` | Начало периода бэктеста (`2025-10-01`) |
| `test_days` | Скользящее окно оценки P/L (23 дня) |
| `start_date_download_minutes` | Дата начала загрузки минутных свечей с MOEX |
| `path_db_minute` | Путь к SQLite с минутными барами |
| `path_db_day` | Путь к SQLite с дневными барами (21:00–20:59:59) |

Размер чанков (`max_chunk_tokens`) зависит от модели и определяется в `create_embedding.py`.

## External Dependencies

**Сервисы:**

- **Ollama** на `localhost:11434` — должен быть запущен с нужной моделью эмбеддингов

**Внешние пути (Windows):**

- Новости: `C:/Users/Alkor/gd/db_rss/rss_news_*.db`
- Котировки: `C:/Users/Alkor/gd/data_quote_db/{ticker}_futures_*.db`
- Markdown: `C:/Users/Alkor/gd/md_rss_all_oil_gaz/`
- Предсказания: `C:/Users/Alkor/gd/predict_ai/{ticker_lc}_all_oil_gaz_gemma/`
- QUIK: `C:\QUIK_VTB_2025_ЕБС\algotrade\input.tri`

**Python-библиотеки:** см. `requirements.txt` (основные: `pandas`, `numpy`, `chromadb`, `tiktoken`, `matplotlib`, `requests`, `PyYAML`)

## Logs & Output

- Логи: `rts/log/` — автоматически, хранятся 3 последних на каждый скрипт
- Графики: `rts/plots/{model_name}_{provider}_{timestamp}.png`
- Данные анализа: `explain_topk_all.pkl` (детали совпадений для `analyze_explain.py`)
- Транзакции QUIK: `.tri`-файлы в `trade/`

## Beget Server — RSS Collection Infrastructure

Папка `beget/` содержит инфраструктуру сбора RSS-новостей на удалённом Ubuntu-сервере (Beget) и синхронизацию результатов на локальную машину.

```text
beget/
├── sync_files.py                      # rsync через WSL: сервер → Windows
├── collect_rss_links_to_yaml.py       # Утилита сбора RSS-ссылок Investing
├── settings.yaml                      # 65 RSS-ссылок Investing (для investing-скрапера)
├── README.md
└── server/                            # Скрипты для Ubuntu-сервера Beget
    ├── rss_scraper_all_providers_to_db_month_msk.py   # Основной: 3 провайдера → единая БД
    ├── rss_scraper_interfax_to_db_month_msk.py        # Legacy: только Interfax
    ├── rss_scraper_investing_to_db_month_msk.py       # Legacy: только Investing
    ├── rss_scraper_prime_to_db_month_msk.py           # Legacy: только 1Prime
    ├── settings.yaml                  # RSS-ленты для all_providers скрапера
    ├── requirements.txt
    └── README.md
```

**Основной скрипт** — `rss_scraper_all_providers_to_db_month_msk.py`:

- Async (aiohttp), Semaphore для контроля параллелизма, HTTP-заголовки
- RSS-ленты из `server/settings.yaml` (ключи: `rss.interfax`, `rss.prime`, `rss.investing`)
- Сохраняет в `rss_news_YYYY_MM.db` с полями `loaded_at, date, title, provider`
- Дедупликация по `(date, title, provider)` перед вставкой
- Запускается по cron на сервере (`/home/user/rss_scraper/`)

**Синхронизация** — `sync_files.py`:

- rsync через WSL с сервера `root@109.172.46.10`
- Синхронизирует `db_data/` → `C:\Users\Alkor\gd\db_rss\` и `db_rss_investing/` → `C:\Users\Alkor\gd\db_rss_investing\`
- Параметры `--inplace --partial --size-only` для SQLite + Google Drive совместимости

**Два settings.yaml:**

- `beget/settings.yaml` — ссылки для investing-скрапера (ключ `rss_links`)
- `beget/server/settings.yaml` — ленты для all_providers (ключи `rss.interfax/prime/investing`, `base_dir`)

## Code Patterns

- Все скрипты в `rts/` читают конфигурацию из `rts/settings.yaml` через `yaml.safe_load` на уровне модуля
- Скрипты в `trade/` не используют `settings.yaml` — конфигурация (тикеры, пути) захардкожена
- Три скрипта `trade_*_tri.py` почти идентичны, различаются только тикерами, количеством и путями
- Логирование: каждый скрипт создаёт файл `{script_name}_{timestamp}.txt`, ротация до 3 файлов (`rts/log/` для rts-скриптов, `trade/log/` для trade-скриптов)
- `.tri`-файлы пишутся в кодировке `cp1251` (требование QUIK)
- Эмбеддинги нормализуются L2 при создании, поэтому `cosine = dot product`
- Серверные скраперы используют `TimedRotatingFileHandler` с `backupCount=3`
