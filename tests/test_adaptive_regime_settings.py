"""Тесты наличия исследовательских настроек у всех тикеров."""

import unittest
from pathlib import Path

import yaml

from research.adaptive_regime_walk_forward import load_walk_forward_config


class TickerSettingsTests(unittest.TestCase):
    """Проверки блока regime_walk_forward в семи settings.yaml."""

    def test_all_ticker_settings_have_research_block(self):
        """Каждый тикер должен иметь независимое окно обучения 63 дня."""
        root = Path(__file__).resolve().parents[1]
        for ticker in ("rts", "mix", "ng", "br", "gold", "si", "spyf"):
            with self.subTest(ticker=ticker):
                settings = yaml.safe_load(
                    (root / ticker / "settings.yaml").read_text(encoding="utf-8")
                )
                config = load_walk_forward_config(settings)
                self.assertEqual(config.train_window_days, 63)


if __name__ == "__main__":
    unittest.main()
