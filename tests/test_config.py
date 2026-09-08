import tempfile
import unittest
from pathlib import Path

from robust_tabular.config import load_config


class ConfigTests(unittest.TestCase):
    def test_loads_valid_config(self):
        text = """
seed: 7
models: [hist_gradient_boosting]
datasets:
  - {name: demo, openml_id: 1, task: classification}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(text, encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.datasets[0].name, "demo")

    def test_rejects_unknown_scenario(self):
        text = """
scenarios: [unknown]
datasets:
  - {name: demo, openml_id: 1, task: classification}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown scenarios"):
                load_config(path)

    def test_core_benchmark_defines_180_fits(self):
        path = Path(__file__).parents[1] / "configs" / "core_benchmark.yaml"
        config = load_config(path)
        cases_per_dataset = (
            int("standard" in config.scenarios)
            + len(config.low_data_fractions)
            + len(config.missingness_rates)
            + int("covariate_shift" in config.scenarios)
        )
        total = len(config.datasets) * len(config.models) * cases_per_dataset * config.repeats
        self.assertEqual(total, 180)


if __name__ == "__main__":
    unittest.main()
