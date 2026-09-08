from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    openml_id: int
    task: str
    target: str | None = None
    max_samples: int | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 42
    test_size: float = 0.2
    models: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm", "catboost"])
    datasets: list[DatasetConfig] = field(default_factory=list)
    scenarios: list[str] = field(
        default_factory=lambda: ["standard", "low_data", "missingness", "covariate_shift"]
    )
    low_data_fractions: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5])
    missingness_rates: list[float] = field(default_factory=lambda: [0.1, 0.3, 0.5])
    repeats: int = 3
    output_dir: str = "results"
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_config(path: str | Path) -> BenchmarkConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    datasets = [DatasetConfig(**item) for item in raw.pop("datasets", [])]
    config = BenchmarkConfig(datasets=datasets, **raw)
    _validate(config)
    return config


def _validate(config: BenchmarkConfig) -> None:
    valid_tasks = {"classification", "regression"}
    valid_scenarios = {"standard", "low_data", "missingness", "covariate_shift"}
    if not config.datasets:
        raise ValueError("At least one dataset is required")
    if not 0 < config.test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if config.repeats < 1:
        raise ValueError("repeats must be at least 1")
    for dataset in config.datasets:
        if dataset.task not in valid_tasks:
            raise ValueError(f"Unsupported task for {dataset.name}: {dataset.task}")
    unknown = set(config.scenarios) - valid_scenarios
    if unknown:
        raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
    for value in [*config.low_data_fractions, *config.missingness_rates]:
        if not 0 < value < 1:
            raise ValueError("Scenario fractions and rates must be between 0 and 1")

