"""Тесты файлового контракта входных результатов."""

import tempfile
import unittest
from pathlib import Path

from research.adaptive_regime_walk_forward import (
    RegimeDataError,
    load_source_results,
)


class InputFileTests(unittest.TestCase):
    """Проверки загрузки Excel текущего симулятора."""

    def test_missing_input_file_has_clear_error(self):
        """Отсутствующий Excel должен давать понятную диагностическую ошибку."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "df_rez_output.xlsx"

            with self.assertRaisesRegex(RegimeDataError, "Не найден"):
                load_source_results(path, train_window_days=2)


if __name__ == "__main__":
    unittest.main()
