from __future__ import annotations

import json
import os
import pickle
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
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
from .report import create_markdown_report, create_plots

RUN_KEY = [
    "dataset",
    "openml_id",
    "task",
    "model",
    "scenario",
    "severity",
    "repeat",
    "seed",
]
TRACKED_PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "torch",
    "tabpfn",
]


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
    results = results.copy()
    if results.empty or "status" not in results:
        return results
    successful = results[results["status"] == "ok"]
    if successful.empty:
        results["performance_degradation"] = np.nan
        results["relative_performance_degradation"] = np.nan
        return results
    primary = np.where(results["task"] == "classification", "f1_macro", "rmse")
    results["primary_metric"] = primary
    results["performance_degradation"] = np.nan
    results["relative_performance_degradation"] = np.nan
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
        degradation = baseline - value if metric == "f1_macro" else value - baseline
        results.loc[index, "performance_degradation"] = degradation
        if baseline != 0:
            results.loc[index, "relative_performance_degradation"] = degradation / abs(baseline)
    return results


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_outputs(results: pd.DataFrame, output: Path) -> pd.DataFrame:
    enriched = _add_degradation(results)
    _atomic_csv(enriched, output / "raw_results.csv")
    if not enriched.empty:
        numeric = enriched.select_dtypes(include="number").columns.tolist()
        summary = (
            enriched[enriched["status"] == "ok"]
            .groupby(["dataset", "task", "model", "scenario", "severity"], dropna=False)[
                numeric
            ]
            .agg(["mean", "std"])
        )
        summary.columns = ["_".join(column).rstrip("_") for column in summary.columns]
        _atomic_csv(summary.reset_index(), output / "summary.csv")
    return enriched


def _package_versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = None
    return found


def _git_commit() -> str | None:
    if commit := os.environ.get("GITHUB_SHA"):
        return commit
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _write_provenance(config: BenchmarkConfig, output: Path) -> None:
    (output / "run_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    environment = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "packages": _package_versions(),
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )


def _experiment_config(config: BenchmarkConfig) -> dict[str, Any]:
    comparable = asdict(config)
    comparable.pop("resume", None)
    comparable.pop("generate_report", None)
    return comparable


def _validate_resume_config(config: BenchmarkConfig, output: Path) -> None:
    path = output / "run_config.json"
    if not path.exists():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    previous.pop("resume", None)
    previous.pop("generate_report", None)
    if previous != _experiment_config(config):
        raise ValueError(
            "The checkpoint was created with a different configuration; "
            "use --restart or choose another output_dir"
        )


def _completed_keys(results: pd.DataFrame) -> set[tuple[Any, ...]]:
    if results.empty or not set(RUN_KEY).issubset(results.columns):
        return set()
    return set(results[RUN_KEY].itertuples(index=False, name=None))


def run_benchmark(config: BenchmarkConfig) -> pd.DataFrame:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "raw_results.csv"
    if config.resume and checkpoint.exists():
        _validate_resume_config(config, output)
        existing = pd.read_csv(checkpoint)
        records: list[dict[str, Any]] = existing.to_dict(orient="records")
    else:
        if checkpoint.exists():
            checkpoint.unlink()
        records = []
    completed = _completed_keys(pd.DataFrame.from_records(records))
    _write_provenance(config, output)

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
                    key = tuple(base[column] for column in RUN_KEY)
                    if key in completed:
                        continue
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
                    completed.add(key)
                    _write_outputs(pd.DataFrame.from_records(records), output)
                    print(
                        f"checkpointed dataset={bundle.name} model={model_name} "
                        f"scenario={scenario} severity={severity} repeat={repeat}"
                    )

    results = _write_outputs(pd.DataFrame.from_records(records), output)
    if config.generate_report:
        create_markdown_report(checkpoint, output / "REPORT.md", output / "environment.json")
        create_plots(checkpoint, output / "figures")
    return results
