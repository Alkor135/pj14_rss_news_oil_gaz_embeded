# import webbrowser
# import os
# import time

# # Список файлов
# files = [
#     r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded\buhinvest_analize\pl_buhinvest_interactive.html",
#     r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded\rts\plots\strategy_analysis.html",
#     r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded\mix\plots\strategy_analysis.html",
#     r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded\br\plots\strategy_analysis.html",
#     r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded\gold\plots\strategy_analysis.html",
#     r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded\ng\plots\strategy_analysis.html",
#     r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded\si\plots\strategy_analysis.html",
#     r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded\spyf\plots\strategy_analysis.html",
# ]

# for file in files:
#     path = os.path.abspath(file)
#     url = f"file:///{path.replace(os.sep, '/')}"
    
#     print(f"[OPEN] {url}")
#     webbrowser.open_new_tab(url)
    
#     time.sleep(0.3)  # небольшая пауза, чтобы вкладки не слипались


"""
Открывает HTML-отчёты всех стратегий в одном новом окне Google Chrome.
Включает strategy_analysis.html для каждого тикера (rts, mix, br, gold, ng, si, spyf)
и отчёт реальной доходности buhinvest (pl_buhinvest_interactive.html).
Путь к chrome.exe и список файлов захардкожены в скрипте.
"""

import subprocess

# Список файлов
files = [
    r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded\buhinvest_analize\pl_buhinvest_interactive.html",
    r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded\rts\plots\strategy_analysis.html",
    r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded\mix\plots\strategy_analysis.html",
    r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded\br\plots\strategy_analysis.html",
    r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded\gold\plots\strategy_analysis.html",
    r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded\ng\plots\strategy_analysis.html",
    r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded\si\plots\strategy_analysis.html",
    r"C:\Users\Alkor\VSCode\pj14_rss_news_oil_gaz_embeded\spyf\plots\strategy_analysis.html",
]

chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

subprocess.Popen([chrome, "--new-window"] + files)
