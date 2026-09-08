import unittest

import numpy as np
import pandas as pd

from robust_tabular.data import covariate_shift_split, inject_missingness, low_data_subset


class DataStressTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(
            {
                "number": np.arange(100, dtype=float),
                "category": pd.Series(["a", "b"] * 50, dtype="category"),
            }
        )
        self.y = pd.Series([0, 1] * 50)

    def test_missingness_is_reproducible_and_non_mutating(self):
        first = inject_missingness(self.X, 0.3, seed=4)
        second = inject_missingness(self.X, 0.3, seed=4)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(int(self.X.isna().sum().sum()), 0)
        self.assertGreater(int(first.isna().sum().sum()), 0)

    def test_low_data_preserves_both_classes(self):
        X_small, y_small = low_data_subset(self.X, self.y, 0.2, "classification", seed=1)
        self.assertEqual(len(X_small), 20)
        self.assertEqual(set(y_small), {0, 1})

    def test_shift_split_is_disjoint_and_has_requested_size(self):
        X_train, X_test, y_train, y_test = covariate_shift_split(
            self.X, self.y, test_size=0.2, seed=2
        )
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(X_train), 80)
        self.assertFalse(set(X_train.index) & set(X_test.index))
        self.assertEqual(len(y_train) + len(y_test), 100)


if __name__ == "__main__":
    unittest.main()

