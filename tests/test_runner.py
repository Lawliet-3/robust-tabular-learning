import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from robust_tabular.config import BenchmarkConfig, DatasetConfig
from robust_tabular.data import DatasetBundle
from robust_tabular.runner import run_benchmark


class RunnerTests(unittest.TestCase):
    def test_end_to_end_with_builtin_model(self):
        rng = np.random.default_rng(3)
        X = pd.DataFrame(
            {
                "x1": rng.normal(size=160),
                "x2": rng.normal(size=160),
                "group": pd.Series(np.where(rng.random(160) > 0.5, "a", "b"), dtype="category"),
            }
        )
        y = pd.Series((X["x1"] + 0.3 * X["x2"] > 0).astype(int))
        bundle = DatasetBundle("synthetic", "classification", X, y)
        with tempfile.TemporaryDirectory() as directory:
            config = BenchmarkConfig(
                seed=5,
                repeats=1,
                models=["hist_gradient_boosting"],
                datasets=[DatasetConfig("synthetic", 1, "classification")],
                scenarios=["standard", "low_data", "missingness", "covariate_shift"],
                low_data_fractions=[0.5],
                missingness_rates=[0.2],
                output_dir=directory,
                generate_report=False,
            )
            with patch("robust_tabular.runner.load_openml_dataset", return_value=bundle):
                results = run_benchmark(config)
            self.assertEqual(len(results), 4)
            self.assertEqual(set(results["status"]), {"ok"})
            self.assertTrue((Path(directory) / "raw_results.csv").exists())
            self.assertTrue((Path(directory) / "summary.csv").exists())
            self.assertTrue((Path(directory) / "environment.json").exists())
            self.assertIn("performance_degradation", results)
            self.assertIn("relative_performance_degradation", results)

    def test_resume_does_not_duplicate_completed_runs(self):
        X = pd.DataFrame({"x": np.arange(40, dtype=float)})
        y = pd.Series([0, 1] * 20)
        bundle = DatasetBundle("synthetic", "classification", X, y)
        with tempfile.TemporaryDirectory() as directory:
            config = BenchmarkConfig(
                models=["hist_gradient_boosting"],
                datasets=[DatasetConfig("synthetic", 1, "classification")],
                scenarios=["standard"],
                repeats=1,
                output_dir=directory,
                generate_report=False,
            )
            with patch("robust_tabular.runner.load_openml_dataset", return_value=bundle):
                first = run_benchmark(config)
                resumed = run_benchmark(config)
            self.assertEqual(len(first), 1)
            self.assertEqual(len(resumed), 1)

    def test_resume_rejects_a_different_configuration(self):
        X = pd.DataFrame({"x": np.arange(40, dtype=float)})
        y = pd.Series([0, 1] * 20)
        bundle = DatasetBundle("synthetic", "classification", X, y)
        with tempfile.TemporaryDirectory() as directory:
            original = BenchmarkConfig(
                seed=42,
                models=["hist_gradient_boosting"],
                datasets=[DatasetConfig("synthetic", 1, "classification")],
                scenarios=["standard"],
                output_dir=directory,
                generate_report=False,
            )
            changed = BenchmarkConfig(
                seed=43,
                models=["hist_gradient_boosting"],
                datasets=[DatasetConfig("synthetic", 1, "classification")],
                scenarios=["standard"],
                output_dir=directory,
                generate_report=False,
            )
            with patch("robust_tabular.runner.load_openml_dataset", return_value=bundle):
                run_benchmark(original)
                with self.assertRaisesRegex(ValueError, "different configuration"):
                    run_benchmark(changed)


if __name__ == "__main__":
    unittest.main()
