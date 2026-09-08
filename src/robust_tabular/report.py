from __future__ import annotations

from pathlib import Path

import pandas as pd


def create_markdown_report(results_path: str | Path, output_path: str | Path) -> None:
    results = pd.read_csv(results_path)
    ok = results[results["status"] == "ok"].copy()
    lines = ["# Robust Tabular Learning — Benchmark Report", ""]
    lines.append(f"Successful runs: **{len(ok)}** of **{len(results)}**.")
    lines.append("")
    for task, metric, ascending in [
        ("classification", "f1_macro", False),
        ("regression", "rmse", True),
    ]:
        subset = ok[(ok["task"] == task) & (ok["scenario"] == "standard")]
        if subset.empty or metric not in subset:
            continue
        table = (
            subset.groupby("model", as_index=False)[metric]
            .mean()
            .sort_values(metric, ascending=ascending)
        )
        lines.extend([f"## Standard {task}", "", table.to_markdown(index=False), ""])
    robust = ok[ok["scenario"] != "standard"]
    if not robust.empty:
        table = (
            robust.groupby(["model", "scenario"], as_index=False)["performance_degradation"]
            .mean()
            .sort_values(["scenario", "performance_degradation"])
        )
        lines.extend(["## Robustness degradation", "", table.to_markdown(index=False), ""])
    failures = results[results["status"] != "ok"]
    if not failures.empty:
        lines.extend(
            [
                "## Skipped or failed runs",
                "",
                failures[["model", "dataset", "status", "error"]]
                .drop_duplicates()
                .to_markdown(index=False),
                "",
            ]
        )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")

