"""
Мастер-скрипт для последовательного запуска пайплайна по всем тикерам.
Порядок: embedding → simulate → analysis) × N тикеров.
Останавливается при первой ошибке (exit code != 0).
Запускается вручную для обновления анализируемых данных на другом компьютере.
"""

import subprocess
import sys
import os

BASE = r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded"
PYTHON = os.path.join(BASE, ".venv", "Scripts", "python.exe")

# список скриптов по порядку
SCRIPTS = [
    r"rts\create_embedding.py",
    r"rts\simulate_trade.py",

    r"mix\create_embedding.py",
    r"mix\simulate_trade.py",

    r"ng\create_embedding.py",
    r"ng\simulate_trade.py",

    r"br\create_embedding.py",
    r"br\simulate_trade.py",

    r"gold\create_embedding.py",
    r"gold\simulate_trade.py",

    r"si\create_embedding.py",
    r"si\simulate_trade.py",

    r"spyf\create_embedding.py",
    r"spyf\simulate_trade.py",

    r"rts\strategy_analysis.py",
    r"mix\strategy_analysis.py",
    r"ng\strategy_analysis.py",
    r"br\strategy_analysis.py",
    r"gold\strategy_analysis.py",
    r"si\strategy_analysis.py",
    r"spyf\strategy_analysis.py",

    r"buhinvest_analize\pl_buhinvest.py",
    r"buhinvest_analize\pl_buhinvest_interactive.py",
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
