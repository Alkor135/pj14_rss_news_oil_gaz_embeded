"""
Мастер-скрипт для последовательного запуска пайплайна по всем тикерам.
Порядок: sync → (download → convert → embedding → simulate → trade) × N тикеров.
Останавливается при первой ошибке (exit code != 0).
Запускается Планировщиком задач через одно задание.
"""

import subprocess
import sys
import os

BASE = r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded"
PYTHON = os.path.join(BASE, ".venv", "Scripts", "python.exe")

# список скриптов по порядку
SCRIPTS = [
    r"beget\sync_files.py",

    r"rts\download_minutes_to_db.py",
    r"rts\convert_minutes_to_days.py",
    r"rts\create_markdown_files.py",
    r"rts\create_embedding.py",
    r"rts\simulate_trade.py",
    r"trade\trade_rts_tri.py",

    r"mix\download_minutes_to_db.py",
    r"mix\convert_minutes_to_days.py",
    r"mix\create_embedding.py",
    r"mix\simulate_trade.py",
    r"trade\trade_mix_tri.py",

    r"ng\download_minutes_to_db.py",
    r"ng\convert_minutes_to_days.py",
    r"ng\create_embedding.py",
    r"ng\simulate_trade.py",

    r"br\download_minutes_to_db.py",
    r"br\convert_minutes_to_days.py",
    r"br\create_embedding.py",
    r"br\simulate_trade.py",

    r"gold\download_minutes_to_db.py",
    r"gold\convert_minutes_to_days.py",
    r"gold\create_embedding.py",
    r"gold\simulate_trade.py",

    r"si\download_minutes_to_db.py",
    r"si\convert_minutes_to_days.py",
    r"si\create_embedding.py",
    r"si\simulate_trade.py",

    r"spyf\download_minutes_to_db.py",
    r"spyf\convert_minutes_to_days.py",
    r"spyf\create_embedding.py",
    r"spyf\simulate_trade.py",
]

def run_script(script: str) -> int:
    script_path = os.path.join(BASE, script)
    cwd = os.path.dirname(script_path)
    print(f"\n=== Запуск: {script} ===")
    result = subprocess.run([PYTHON, script_path], cwd=cwd)
    return result.returncode

def main():
    for script in SCRIPTS:
        code = run_script(script)
        if code != 0:
            print(f"❌ Ошибка выполнения {script}, код {code}")
            os.system("pause")
            sys.exit(code)
    print("\n✅ Все скрипты выполнены успешно")
    input("\nНажмите Enter для выхода...")  # вместо sys.exit вручную

if __name__ == "__main__":
    main()
