from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from .config import DatasetConfig


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    task: str
    X: pd.DataFrame
    y: pd.Series


def load_openml_dataset(config: DatasetConfig, seed: int = 42) -> DatasetBundle:
    bunch = fetch_openml(data_id=config.openml_id, as_frame=True, parser="auto")
    X = bunch.data.copy()
    y = bunch.target.copy()
    if config.target and config.target in X.columns:
        y = X.pop(config.target)
    if config.max_samples and len(X) > config.max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(X), size=config.max_samples, replace=False))
        X = X.iloc[indices]
        y = y.iloc[indices]
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    if config.task == "classification":
        y = y.astype("category").cat.codes
    else:
        y = pd.to_numeric(y, errors="raise")
    return DatasetBundle(config.name, config.task, X, y)


def random_split(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)


def covariate_shift_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a deterministic covariate shift using a random feature projection.

    Rows with the highest projection scores form the test set. The target is never
    used to construct the split, so this simulates a changed input population.
    """
    if len(X) < 5:
        raise ValueError("At least five rows are required for a covariate-shift split")
    encoded = pd.DataFrame(index=X.index)
    for column in X.columns:
        series = X[column]
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce")
            median = values.median()
            values = values.fillna(0.0 if pd.isna(median) else median)
            scale = values.std()
            encoded[column] = (values - values.mean()) / (scale if scale > 0 else 1.0)
        else:
            encoded[column] = pd.factorize(series.astype("string").fillna("__missing__"))[0]
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=encoded.shape[1])
    scores = encoded.to_numpy(dtype=float) @ weights
    n_test = max(1, int(round(len(X) * test_size)))
    order = np.argsort(scores, kind="stable")
    test_idx, train_idx = order[-n_test:], order[:-n_test]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def inject_missingness(X: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    if not 0 <= rate < 1:
        raise ValueError("rate must be in [0, 1)")
    rng = np.random.default_rng(seed)
    mask = rng.random(X.shape) < rate
    corrupted = X.copy()
    return corrupted.mask(mask)


def low_data_subset(
    X: pd.DataFrame,
    y: pd.Series,
    fraction: float,
    task: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if fraction >= 1:
        return X.copy(), y.copy()
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_small, _, y_small, _ = train_test_split(
        X, y, train_size=fraction, random_state=seed, stratify=stratify
    )
    return X_small, y_small

