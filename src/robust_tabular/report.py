from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _mean_std_table(frame: pd.DataFrame, metric: str, ascending: bool) -> pd.DataFrame:
    table = frame.groupby(["dataset", "model"])[metric].agg(["mean", "std"]).reset_index()
    table["result"] = table.apply(
        lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}"
        if pd.notna(row["std"])
        else f"{row['mean']:.4f}",
        axis=1,
    )
    return table.sort_values(["dataset", "mean"], ascending=[True, ascending])[
        ["dataset", "model", "result"]
    ]


def create_markdown_report(
    results_path: str | Path,
    output_path: str | Path,
    environment_path: str | Path | None = None,
) -> None:
    results = pd.read_csv(results_path)
    ok = results[results["status"] == "ok"].copy()
    lines = ["# Robust Tabular Learning — Benchmark Report", ""]
    lines.extend(
        [
            f"Successful runs: **{len(ok)}** of **{len(results)}**.",
            "Values aggregate all completed seeds and are shown as mean ± standard deviation.",
            "",
        ]
    )
    for task, metric, ascending in [
        ("classification", "f1_macro", False),
        ("regression", "rmse", True),
    ]:
        subset = ok[(ok["task"] == task) & (ok["scenario"] == "standard")]
        if subset.empty or metric not in subset:
            continue
        table = _mean_std_table(subset, metric, ascending)
        lines.extend([f"## Standard {task}", "", _markdown_table(table), ""])

    robust = ok[ok["scenario"] != "standard"]
    if not robust.empty and "relative_performance_degradation" in robust:
        table = (
            robust.groupby(["model", "scenario"])["relative_performance_degradation"]
            .agg(["mean", "std"])
            .reset_index()
        )
        table["mean"] *= 100
        table["std"] *= 100
        table = table.rename(
            columns={"mean": "mean degradation (%)", "std": "std (%)"}
        ).sort_values(["scenario", "mean degradation (%)"])
        lines.extend(["## Relative robustness degradation", "", _markdown_table(table), ""])

    failures = results[results["status"] != "ok"]
    if not failures.empty:
        lines.extend(
            [
                "## Skipped or failed runs",
                "",
                _markdown_table(
                    failures[["model", "dataset", "status", "error"]].drop_duplicates()
                ),
                "",
            ]
        )

    if environment_path and Path(environment_path).exists():
        environment: dict[str, Any] = json.loads(
            Path(environment_path).read_text(encoding="utf-8")
        )
        lines.extend(
            [
                "## Environment",
                "",
                f"- Git commit: `{environment.get('git_commit') or 'unavailable'}`",
                f"- Platform: `{environment.get('platform') or 'unavailable'}`",
                f"- Python: `{str(environment.get('python', 'unavailable')).splitlines()[0]}`",
                "",
            ]
        )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def _task_axes(plt: Any, tasks: list[str], title: str) -> tuple[Any, list[Any]]:
    figure, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 5), squeeze=False)
    figure.suptitle(title)
    return figure, list(axes[0])


def create_plots(results_path: str | Path, output_dir: str | Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    results = pd.read_csv(results_path)
    ok = results[results["status"] == "ok"].copy()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    tasks = [task for task in ["classification", "regression"] if task in set(ok["task"])]
    if not tasks:
        return created

    standard = ok[ok["scenario"] == "standard"]
    figure, axes = _task_axes(plt, tasks, "Clean performance")
    for axis, task in zip(axes, tasks, strict=True):
        metric = "f1_macro" if task == "classification" else "rmse"
        pivot = standard[standard["task"] == task].pivot_table(
            index="dataset", columns="model", values=metric, aggfunc="mean"
        )
        pivot.plot(kind="bar", ax=axis)
        label = (
            "Macro-F1 (higher is better)"
            if task == "classification"
            else "RMSE (lower is better)"
        )
        axis.set_ylabel(label)
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    path = output / "clean_performance.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    created.append(path)

    robust = ok[ok["scenario"] != "standard"]
    if not robust.empty and robust["relative_performance_degradation"].notna().any():
        pivot = robust.pivot_table(
            index="scenario",
            columns="model",
            values="relative_performance_degradation",
            aggfunc="mean",
        ) * 100
        axis = pivot.plot(kind="bar", figsize=(10, 5), title="Relative robustness degradation")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("Primary-metric degradation (%)")
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=20)
        figure = axis.get_figure()
        figure.tight_layout()
        path = output / "robustness_degradation.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        created.append(path)

    figure, axes = _task_axes(plt, tasks, "Predictive quality versus fit cost")
    for axis, task in zip(axes, tasks, strict=True):
        metric = "f1_macro" if task == "classification" else "rmse"
        grouped = (
            standard[standard["task"] == task]
            .groupby("model", as_index=False)[["fit_seconds", metric]]
            .mean()
        )
        axis.scatter(grouped["fit_seconds"], grouped[metric])
        for row in grouped.itertuples(index=False):
            axis.annotate(row.model, (row.fit_seconds, getattr(row, metric)), fontsize=8)
        axis.set_xscale("log")
        axis.set_xlabel("Mean fit time (seconds, log scale)")
        axis.set_ylabel("Macro-F1" if task == "classification" else "RMSE")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    path = output / "quality_vs_cost.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    created.append(path)
    return created
