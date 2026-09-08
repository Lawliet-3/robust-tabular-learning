from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import BenchmarkConfig
from .data import (
    DatasetBundle,
    covariate_shift_split,
    inject_missingness,
    load_openml_dataset,
    low_data_subset,
    random_split,
)
from .metrics import evaluate
from .models import ModelUnavailableError, build_model


def _fit_evaluate(
    bundle: DatasetBundle,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    model = build_model(model_name, bundle.task, X_train, seed, params)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = model.predict(X_test)
    predict_seconds = time.perf_counter() - started
    score = None
    if bundle.task == "classification" and hasattr(model, "predict_proba"):
        score = model.predict_proba(X_test)
    metrics = evaluate(bundle.task, y_test.to_numpy(), np.asarray(prediction), score)
    try:
        model_size_mb = len(pickle.dumps(model)) / (1024 * 1024)
    except (pickle.PickleError, TypeError, AttributeError):
        model_size_mb = float("nan")
    return {
        **metrics,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "latency_ms_per_1000": predict_seconds * 1_000_000 / max(1, len(X_test)),
        "model_size_mb": model_size_mb,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
    }


def _scenario_cases(
    bundle: DatasetBundle,
    config: BenchmarkConfig,
    repeat_seed: int,
):
    X_train, X_test, y_train, y_test = random_split(
        bundle.X, bundle.y, bundle.task, config.test_size, repeat_seed
    )
    if "standard" in config.scenarios:
        yield "standard", 0.0, X_train, X_test, y_train, y_test
    if "low_data" in config.scenarios:
        for fraction in config.low_data_fractions:
            small_X, small_y = low_data_subset(
                X_train, y_train, fraction, bundle.task, repeat_seed
            )
            yield "low_data", fraction, small_X, X_test, small_y, y_test
    if "missingness" in config.scenarios:
        for rate in config.missingness_rates:
            corrupted = inject_missingness(X_test, rate, repeat_seed + int(rate * 1000))
            yield "missingness", rate, X_train, corrupted, y_train, y_test
    if "covariate_shift" in config.scenarios:
        shift = covariate_shift_split(
            bundle.X, bundle.y, config.test_size, repeat_seed
        )
        yield "covariate_shift", config.test_size, *shift


def _add_degradation(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "status" not in results:
        return results
    successful = results[results["status"] == "ok"]
    if successful.empty:
        results["performance_degradation"] = np.nan
        return results
    primary = np.where(results["task"] == "classification", "f1_macro", "rmse")
    results["primary_metric"] = primary
    results["performance_degradation"] = np.nan
    keys = ["dataset", "model", "repeat"]
    standards = successful[successful["scenario"] == "standard"].set_index(keys)
    for index, row in results[results["status"] == "ok"].iterrows():
        key = (row["dataset"], row["model"], row["repeat"])
        if key not in standards.index:
            continue
        metric = row["primary_metric"]
        baseline = standards.loc[key, metric]
        value = row.get(metric, np.nan)
        if pd.isna(value) or pd.isna(baseline):
            continue
        results.loc[index, "performance_degradation"] = (
            baseline - value if metric == "f1_macro" else value - baseline
        )
    return results


def run_benchmark(config: BenchmarkConfig) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset_config in config.datasets:
        bundle = load_openml_dataset(dataset_config, config.seed)
        for repeat in range(config.repeats):
            repeat_seed = config.seed + repeat
            for scenario, severity, X_train, X_test, y_train, y_test in _scenario_cases(
                bundle, config, repeat_seed
            ):
                for model_name in config.models:
                    base = {
                        "dataset": bundle.name,
                        "openml_id": dataset_config.openml_id,
                        "task": bundle.task,
                        "model": model_name,
                        "scenario": scenario,
                        "severity": severity,
                        "repeat": repeat,
                        "seed": repeat_seed,
                    }
                    try:
                        measured = _fit_evaluate(
                            bundle,
                            model_name,
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            repeat_seed,
                            config.model_params.get(model_name, {}),
                        )
                        records.append({**base, "status": "ok", "error": "", **measured})
                    except ModelUnavailableError as exc:
                        records.append({**base, "status": "skipped", "error": str(exc)})
                    except Exception as exc:  # isolate failed model/dataset combinations
                        records.append(
                            {**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                        )
    results = _add_degradation(pd.DataFrame.from_records(records))
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "raw_results.csv", index=False)
    with (output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
    if not results.empty:
        numeric = results.select_dtypes(include="number").columns.tolist()
        summary = (
            results[results["status"] == "ok"]
            .groupby(["dataset", "task", "model", "scenario", "severity"], dropna=False)[numeric]
            .agg(["mean", "std"])
        )
        summary.columns = ["_".join(column).rstrip("_") for column in summary.columns]
        summary.reset_index().to_csv(output / "summary.csv", index=False)
    return results

