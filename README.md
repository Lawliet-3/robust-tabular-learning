# Robust Tabular Learning

A reproducible research benchmark comparing **XGBoost, LightGBM, CatBoost,
FT-Transformer, and TabPFN** under conditions that resemble real tabular ML
deployments—not only clean random train/test splits.

This project supports the research direction in my KMUTT Computer Engineering
master's application: understanding when pretrained and neural tabular models
improve on strong tree ensembles, and how those choices change under limited
data, missing values, and distribution shift.

## Research questions

1. Which model family performs best on clean classification and regression tasks?
2. How quickly does each model degrade as training data becomes scarce?
3. Which models are most robust to missing values introduced at inference time?
4. How well do the models generalize under covariate shift?
5. What accuracy–cost trade-off emerges from training time, inference latency,
   and serialized model size?

## Experiment design

| Axis | Values |
|---|---|
| Models | XGBoost, LightGBM, CatBoost, FT-Transformer, TabPFN |
| Data | Seven pinned OpenML datasets; classification and regression |
| Clean baseline | Stratified random split for classification |
| Low data | 10%, 25%, and 50% of the original training partition |
| Missing data | 10%, 30%, and 50% MCAR masking at inference time |
| Shift | Target-independent test split from the upper tail of a random feature projection |
| Quality | Accuracy, macro-F1, ROC-AUC, RMSE, MAE, R² |
| Efficiency | Fit time, prediction time, latency per 1,000 rows, model size |
| Reliability | Three seeded repetitions; raw and aggregated outputs |

The shift split deliberately uses features only—not the target—to avoid leaking
outcomes into the scenario definition. Every dataset is pinned by OpenML data ID.

## Quick start

Python 3.10–3.13 is supported. A GPU is useful for TabPFN and FT-Transformer but
is not required for the tree-only smoke benchmark.

```bash
git clone <your-repository-url>
cd robust-tabular-learning
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e '.[all,dev]'
robust-tabular configs/benchmark.yaml
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1` instead of the `source`
command. TabPFN's first run can download pretrained weights, so run the smoke
configuration first and reserve the full benchmark for a longer session.

For a fast pipeline check using only scikit-learn:

```bash
pip install -e '.[dev]'
robust-tabular configs/smoke.yaml
```

Outputs are written beneath the configured `output_dir`:

- `raw_results.csv`: one row per dataset/model/scenario/severity/repeat
- `summary.csv`: means and standard deviations for numeric measures
- `run_config.json`: exact configuration used for the run

Generate a short Markdown report after a completed run:

```python
from robust_tabular.report import create_markdown_report

create_markdown_report("results/full/raw_results.csv", "results/full/report.md")
```

## Configuration

Edit [`configs/benchmark.yaml`](configs/benchmark.yaml) to change datasets,
repetitions, corruption levels, or model hyperparameters. Missing optional
dependencies are recorded as `skipped`; an error in one model/dataset pair is
recorded as `failed` while the rest of the matrix continues.

For a defensible final study, keep the main benchmark configuration frozen,
report all repetitions, and treat any later hyperparameter exploration as a
separate ablation rather than silently tuning on the test results.

## Reproducibility and tests

```bash
ruff check .
pytest --cov=robust_tabular
```

CI runs linting and unit tests on every push and pull request. Unit tests use a
small synthetic mixed-type dataset and do not download OpenML data.

## Repository layout

```text
configs/                  Benchmark and smoke-test configurations
src/robust_tabular/       Data, perturbations, models, runner, metrics, report
tests/                    Offline unit and integration tests
.github/workflows/ci.yml  Continuous integration
```

## Scope and limitations

- MCAR masking is a controlled first robustness study; MAR/MNAR mechanisms are
  natural follow-up ablations.
- The covariate-shift split is synthetic and reproducible. Temporal or
  domain-defined shifts should be added where dataset metadata supports them.
- One-hot preprocessing provides a consistent comparison interface, but a later
  native-categorical ablation would better expose CatBoost's inductive bias.
- TabPFN may download model weights on first use and can require substantial
  compute depending on package/model version.

## Citation

If you use this benchmark, cite the repository and record the commit hash,
configuration file, Python environment, and hardware used for each experiment.
