import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from robust_tabular.report import create_markdown_report, create_plots


class ReportTests(unittest.TestCase):
    def test_report_and_plots_are_generated(self):
        rows = []
        for score in [0.70, 0.74, 0.72]:
            rows.extend(
                [
                    {
                        "dataset": "demo",
                        "task": "classification",
                        "model": "model_a",
                        "scenario": "standard",
                        "status": "ok",
                        "f1_macro": score,
                        "fit_seconds": 0.2,
                        "relative_performance_degradation": 0.0,
                    },
                    {
                        "dataset": "demo",
                        "task": "classification",
                        "model": "model_a",
                        "scenario": "missingness",
                        "status": "ok",
                        "f1_macro": score - 0.05,
                        "fit_seconds": 0.2,
                        "relative_performance_degradation": 0.05 / score,
                    },
                ]
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "raw_results.csv"
            environment = root / "environment.json"
            report = root / "REPORT.md"
            pd.DataFrame(rows).to_csv(results, index=False)
            environment.write_text(
                json.dumps({"git_commit": "abc123", "platform": "test", "python": "3.12"}),
                encoding="utf-8",
            )
            create_markdown_report(results, report, environment)
            plots = create_plots(results, root / "figures")
            text = report.read_text(encoding="utf-8")
            self.assertIn("0.7200 ± 0.0200", text)
            self.assertIn("abc123", text)
            if plots:
                self.assertEqual(len(plots), 3)
                self.assertTrue(all(path.exists() for path in plots))


if __name__ == "__main__":
    unittest.main()
