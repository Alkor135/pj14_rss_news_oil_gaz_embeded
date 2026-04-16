# beget — инфраструктура RSS-сбора

Папка содержит инструменты для сбора RSS-новостей на удалённом Ubuntu-сервере Beget и синхронизации результатов на локальную Windows-машину.

## Содержимое

| Файл | Назначение |
| --- | --- |
| `sync_files.py` | Синхронизация `.db`-файлов с сервера через WSL + rsync |
| `collect_rss_links_to_yaml.py` | Утилита: парсит страницу Investing webmaster-tools и собирает RSS-ссылки в `links.yaml` |
| `settings.yaml` | Список RSS-ссылок Investing (ключ `rss_links`) — используется сервером |
| `server/` | Скрипты, работающие на Ubuntu-сервере Beget (см. `server/README.md`) |

## sync_files.py

Запускает `rsync` через WSL с сервера `root@109.172.46.10` на две пары директорий:

| Источник (сервер) | Назначение (Windows) |
| --- | --- |
| `/home/user/rss_scraper/db_rss_investing/` | `C:\Users\Alkor\gd\db_rss_investing\` |
| `/home/user/rss_scraper/db_data/` | `C:\Users\Alkor\gd\db_rss\` |

Флаги `--inplace --partial --size-only` — совместимость с SQLite и Google Drive. Логи операций пишутся в `sync.log` в целевых папках.

### Требования

- Windows с WSL (Ubuntu)
- `rsync` в WSL: `sudo apt install rsync`
- SSH-ключи для beget-сервера (вход без пароля)

### Запуск

```bash
python beget/sync_files.py
```

Автоматически вызывается первым шагом из `run_all.py`.

## collect_rss_links_to_yaml.py

Разовая утилита. Парсит HTML страницы `https://ru.investing.com/webmaster-tools/rss`, собирает все ссылки на `.rss`-файлы и сохраняет в `links.yaml`. Запускать на IP, где страница открывается без 403.

```bash
python beget/collect_rss_links_to_yaml.py
```

## Два settings.yaml

- `beget/settings.yaml` — RSS-ссылки Investing (ключ `rss_links`) для investing-скрапера
- `beget/server/settings.yaml` — ленты для all_providers (ключи `rss.interfax` / `rss.prime` / `rss.investing` + `base_dir`)
