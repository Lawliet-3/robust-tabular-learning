# Robust Tabular Learning

A reproducible research benchmark comparing **XGBoost, LightGBM, CatBoost,
FT-Transformer, and TabPFN** under conditions that resemble real tabular ML
deployments—not only clean random train/test splits.

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

## Pipeline validation

The table below is the output of the lightweight smoke benchmark using
`HistGradientBoostingClassifier` on the pinned OpenML diabetes dataset (data ID
37). It verifies the complete data-loading, perturbation, evaluation, and CSV
reporting pipeline before running the more computationally expensive five-model
study.

| Scenario | Severity | Train rows | Accuracy | Macro-F1 | ROC-AUC | F1 degradation | Fit time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard | — | 614 | 0.7468 | 0.7231 | 0.8207 | 0.0000 | 0.245 |
| Low data | 25% training data | 153 | 0.7143 | 0.6707 | 0.8028 | 0.0524 | 0.075 |
| Missingness | 30% MCAR | 614 | 0.7727 | 0.7395 | 0.7852 | -0.0164 | 0.245 |
| Covariate shift | 20% shifted test set | 614 | 0.8701 | 0.6513 | 0.7965 | 0.0718 | 0.268 |

These are single-seed pipeline-validation results, not final comparative
evidence. In particular, a negative degradation means that macro-F1 happened to
increase relative to the clean split in this run. Timing varies by hardware.

### Reproduce these results

```bash
git clone https://github.com/Lawliet-3/robust-tabular-learning.git
cd robust-tabular-learning
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
pip install -e '.[dev]'
robust-tabular configs/smoke.yaml
```

The command creates:

- `results/smoke/raw_results.csv`: every individual scenario run
- `results/smoke/summary.csv`: aggregated metrics
- `results/smoke/run_config.json`: the resolved configuration and seed
- `results/smoke/environment.json`: software, platform, hardware, and commit metadata
- `results/smoke/REPORT.md`: generated mean and degradation tables
- `results/smoke/figures/`: clean performance, robustness, and cost plots

## Core comparative benchmark

The recommended main study is defined in
[`configs/core_benchmark.yaml`](configs/core_benchmark.yaml). It compares all
five models on `credit-g`, `spambase`, and `kin8nm`, using three seeds and one
representative level for each stressor. This produces 180 fits:

```text
3 datasets × 5 models × 4 conditions × 3 seeds = 180 fits
```

Run it in a GPU environment:

```bash
pip install -e '.[all,dev]'
robust-tabular configs/core_benchmark.yaml
```

Every completed fit is saved atomically to `results/core/raw_results.csv`. If
the process stops, run the same command again and it resumes from the checkpoint.
Use `--restart` only when you intentionally want to discard the checkpoint and
rerun the entire matrix. Reports, figures, configuration, and environment
metadata are regenerated automatically.

Python 3.10–3.13 is supported. TabPFN may download pretrained weights on its
first run. For the larger seven-dataset, multi-severity extension, use:

```bash
robust-tabular configs/benchmark.yaml
```

## Configuration

Edit [`configs/core_benchmark.yaml`](configs/core_benchmark.yaml) to change datasets,
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
