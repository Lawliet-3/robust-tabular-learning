import unittest

import numpy as np

from robust_tabular.metrics import evaluate


class MetricTests(unittest.TestCase):
    def test_classification_metrics(self):
        metrics = evaluate(
            "classification",
            np.array([0, 0, 1, 1]),
            np.array([0, 0, 1, 1]),
            np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]),
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1_macro"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_regression_metrics(self):
        metrics = evaluate("regression", np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["r2"], 1.0)


if __name__ == "__main__":
    unittest.main()

