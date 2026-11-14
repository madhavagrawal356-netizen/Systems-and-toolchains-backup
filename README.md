# SmartOps-Health

**Forecasting next week's hospital bed occupancy, and staffing for the uncertainty.**

06-763 Systems and Toolchains for AI Engineers · final project

Can a probabilistic forecast of a hospital's occupied inpatient beds one week ahead beat
"next week looks like this week" on weeks and hospitals it never saw? And does staffing to its
upper quantile reduce shortfalls? The users are hospital capacity managers and nurse staffing
coordinators, who set next week's schedule before demand is known.

We implemented the Stage 1/2 model ladder,
a shared evaluation harness, leakage checks, MLflow tracking, and final test
evaluation. **Ladder LightGBM achieved the best official test MAE: 4.414 beds, a 9.15%
improvement over persistence.** The quantile MLP provides a calibrated 80% interval,
but does not pass the proposed staffing-shortfall check. Our remaining uncertainty work focuses on validation-only calibration and staffing analysis.

## Data

US HHS, *COVID-19 Reported Patient Impact and Hospital Capacity by Facility*
([healthdata.gov `anag-cw7u`](https://healthdata.gov/d/anag-cw7u)), public domain as a U.S.
Government work. Data is never committed; download scripts rebuild `data/` from source.

The data pipeline writes `data/clean/beds_weekly.parquet` (one row per hospital per week),
`data/clean/counts.json` (every count we report), and `data/processed/features.parquet`
(the shared model inputs and next-week target). Decisions and numbers:
[docs/steps/02-data.md](docs/steps/02-data.md).

The source snapshot contains 1,045,406 reports from 5,172 hospitals. The cleaner inserts
9,221 missing hospital-weeks so that a lag always means a calendar week, rather than
the previous available report. The resulting forecast table contains 968,688 rows.
Counts come from `data/clean/counts.json` and the
`data/processed/audit/leakage_audit.json`.

Suppressed averages are recovered from the published sum divided by reporting coverage
when possible: 80,608 source rows are recovered this way. Truly unknown targets remain
missing. Official models use measured or derived `y`, never the midpoint-filled
`y_filled` target. Over-capacity and low-coverage reports are flagged and retained;
predictions are not capped at reported capacity. This preserves observed demand and
avoids treating a suppression guess as ground truth. A suppression/cleaning ablation
is not part of the results reported here.

## Modeling method and decision rationale

### Forecast target and evaluation design

Each row represents a forecast made at the end of `origin_week` for `target_week`,
exactly seven days later. The target is the **weekly average number of occupied
inpatient beds**. It can be fractional; it is not a count of new admissions.
`beds_lag1` is occupancy observed in the origin week.

| Target-week period | Dates | Role in the official ladder |
|---|---|---|
| Training | 2020-08-02 to 2022-12-25 | Fit weights, imputers, scalers and encoders |
| Validation | 2023-01-01 to 2023-06-25 | Select tree iterations and the MLP checkpoint |
| Test | 2023-07-02 to 2024-04-21 | Evaluate the frozen, validation-selected models |

An additional hospital split holds out 1,034 hospitals (20%) and leaves 4,138 in the
training group. Membership is deterministic: hospitals are ordered by a hash of their
identifier and seed 42. Holdout hospitals never contribute to fitting or early stopping,
including during their training-period weeks. Their own past observations remain
available as forecast inputs. The roster is retrospective, so this evaluates unseen
hospital identities with observed history, not hospitals with no history at all.

We use chronological splits to test performance on later weeks and avoid random mixing
of past and future. The hospital split separately tests generalization across facilities.
The sample's period is determined by its **target date**, including at split boundaries.
See [config.py](src/smartops/config.py) and [features.py](src/smartops/features.py).

### Why predict changes around persistence?

Persistence predicts `next_week = beds_lag1`. It is a strong baseline because occupancy
usually changes less from one week to the next than it differs between hospitals.
Seasonal naive, which predicts `beds_lag52`, was also checked and was substantially worse
on the rows shared by the two baselines; see `data/processed/baselines.json`.

We therefore train the ladder on `change = y - beds_lag1`, then reconstruct
`prediction = max(0, beds_lag1 + predicted_change)`. This asks the model to learn a
correction to a strong baseline, reducing the need to explain hospital size through the
absolute target. It is a modeling choice, not a proven causal explanation of the gains.
The same reconstruction is applied to the MLP's three change quantiles.

### Features actually used

The shared table contains occupancy lags 1, 2, 3, 4 and 52, lagged capacity, lagged
COVID/flu measures, state and target-week calendar information. The official linear,
LightGBM and MLP models use the **seven engineered numeric features below plus state**;
they do not use every raw column in the table. Momentum uses only the recent change.

| Feature | Construction | Reason for including it |
|---|---|---|
| Momentum | `lag1 - lag2` | Capture recent movement or reversal |
| Four-week slope | `(3*lag1 + lag2 - lag3 - 3*lag4) / 10` | Describe the recent trend across several weeks |
| Eight-week mean-reversion distance | `lag1 - trailing_8_week_mean` | Measure departure from the hospital's recent level |
| Origin occupancy rate | `lag1 / capacity_lag1` | Represent demand relative to reported capacity |
| Holiday-week flag | Known holiday calendar for the target week | Allow recurring calendar effects |
| Week sine and cosine | Cyclic encoding of target ISO week using a 52-week period | Represent seasonality without a large year-end discontinuity |
| State | Origin-time state label | Allow shared geographic differences |

The eight-week mean ends at the forecast origin and requires eight observed weekly
values. Missing history and zero-capacity ratios remain missing until preprocessing.
Calendar information about the target week is known in advance; target-week occupancy,
capacity, suppression labels and quality flags are not model inputs. Engineered features
are computed in the shared [feature loader](src/smartops/features.py).

### The official model ladder

We increase model flexibility while keeping the target, split, input population
and scoring rules fixed. This lets us test whether additional complexity improves
on simple corrections to persistence.

| Model | Implementation | Decision rationale |
|---|---|---|
| Momentum | One no-intercept least-squares coefficient: `predicted_change = phi * momentum` | A simple, interpretable check of whether recent movement predicts next week's change |
| Linear regression | Ordinary least squares on the seven numeric features and one-hot state | Test whether several signals help without nonlinear interactions; this is not ridge regression |
| Ladder LightGBM | Boosted trees with quantile objective at alpha 0.5 | Test nonlinear interactions with an objective aligned with median/absolute-error prediction |
| Quantile MLP | PyTorch network with state embeddings and p10/p50/p90 outputs | Learn nonlinear relationships and conditional uncertainty for capacity planning |

We fill missing numeric values with medians learned from training rows. Linear and
MLP numeric inputs are also standardized using training statistics. Linear and LightGBM
use one-hot state encoding with unseen categories ignored; the MLP uses an eight-dimensional
state embedding with a reserved unknown-state code. Embeddings let the network learn a
compact state representation; no state similarity is assumed or claimed in advance.
Momentum falls back to a zero change when its input is missing. Its fitted coefficient
is approximately **-0.152**, indicating mild reversal in this fit.

LightGBM uses a learning rate of 0.03, 31 leaves, at least 100 samples per child, a
3,000-tree ceiling, and validation early stopping after 100 non-improving iterations.
It predicts a median only; the official tree model does not provide uncertainty bands.

The MLP has hidden layers of 64 and 32 units with ReLU activations, dropout 0.1 after
the first layer, and three outputs. The lower quantile is constructed as the median
minus a positive softplus offset, and the upper quantile as the median plus one.
This prevents quantile crossing by construction.

We train the MLP to minimize mean **pinball loss** across quantiles 0.1, 0.5 and 0.9. For residual
`e = actual - prediction`, the quantile loss is `max(q*e, (q-1)*e)`: underprediction is
penalized more at high quantiles, so p90 can represent an upper demand estimate.
AdamW uses learning rate 0.001, weight decay 0.00001, batch size 1,024 and gradient
clipping at 5. Training runs for at most 60 epochs, stops after eight epochs without
validation improvement, and restores the best validation-pinball checkpoint.
Early stopping, dropout and weight decay limit overfitting; these settings are a fixed
experiment configuration, not the result of an exhaustive hyperparameter search.

All official fits use seed 42 and training hospitals' training weeks. The final-test
command repeats that fitting/selection procedure and then freezes preprocessing and
weights. There is **no train-plus-validation refit** in the official ladder. Test
forecasts may use observations from earlier test weeks, but never update fitted weights
from test targets. This is rolling one-week forecasting, not a single multi-week forecast.
Implementation: [ladder_models.py](src/smartops/ladder_models.py).

### Scoring and success criteria

The [shared scorer](src/smartops/evaluate.py) aligns predictions by hospital and target
week, validates saved targets, rejects duplicate keys, and uses identical finite rows
for every model in a comparison. Official models must predict every row with a known
target and persistence input; missing model predictions fail evaluation rather than
silently making the test easier. Row counts, exclusions and SHA256 fingerprints are saved.

- **MAE in beds:** mean absolute error across eligible hospital-weeks; lower is better.
- **Skill:** `100 * (1 - model_MAE / persistence_MAE)` on the same rows.
- **Hospital win share:** fraction of hospitals with lower mean MAE than persistence.
  This supplements pooled MAE, whose error magnitude can be dominated by large facilities.
- **Uncertainty:** quantile pinball loss, p10-to-p90 interval coverage and mean width.
- **Staffing proxy:** plan capacity at p90; report shortfall frequency, mean unmet beds,
  and mean idle beds. The comparator is persistence plus the training residual's p90
  (7.3 beds), with the threshold fixed before validation/test scoring. These are bed-demand
  proxies, not a nurse-roster optimization or evidence of operational savings.

The implemented proposal checks require at least 3% point skill, positive skill on unseen
hospitals, and beating persistence for more than half of training hospitals. The 80%
interval should cover 77-83% of targets. Staffing usefulness requires **both** a lower
shortfall frequency and no increase in mean idle beds. These are descriptive thresholds;
no confidence intervals or significance tests are reported. p10/p50/p90 cannot establish
coverage of a central 90% interval, which would require p05 and p95.

## Official validation results

The 2026-11-14 post-holiday-fix rerun uses the shared scoring harness in
[evaluate.py](src/smartops/evaluate.py). Every official model and persistence use
**100,151 identical validation rows** from training hospitals and **25,278 identical
validation rows** from held-out hospitals. Fits use only training weeks and training
hospitals; validation selects iterations/epochs. The test results below are from the
earlier pre-fix evaluation and have not been rerun.

| Model | Val rows | Val MAE (beds) | Skill vs. persistence | Unseen-hospital skill |
|---|---:|---:|---:|---:|
| Persistence | 100,151 | 4.748 | 0.0% | 0.0% |
| Momentum | 100,151 | 4.632 | 2.4% | 2.5% |
| Linear | 100,151 | 4.555 | 4.1% | 4.4% |
| LightGBM check | 100,151 | 4.381 | 7.7% | 8.0% |
| Quantile MLP median | 100,151 | 4.445 | 6.4% | 6.7% |

The MLP's 80% interval covers 80.42% of training-hospital validation targets and
80.26% for held-out hospitals, both within the proposed 77-83% band. Its p90 staffing
shortfall rate is 10.41% and 10.50%, respectively, versus 9.20% and 8.96% for
persistence plus the training-error p90. Mean shortfall and idle beds are lower for
the MLP in both groups, but the staffing check still fails because shortfall frequency
is higher. A 90% interval is unavailable from p10/p50/p90.

### Validation artifacts and tracking

The ladder saves `data/processed/ladder_predictions.parquet` and
`data/processed/ladder_metrics.json`. It starts an MLflow parent run before
loading inputs, starts a child run before each model fit, and links each validation
score run to its fit. Runs include parameters, seed, source snapshots, package
versions, feature-file SHA256, row fingerprints, group metrics and output artifacts.
Exceptions produce an actual `FAILED` run and a traceback artifact. The same lifecycle
wrapper also covers the baseline entry point.

Engineered features are shared in [features.py](src/smartops/features.py). Imputers,
scalers, state encoders and category vocabularies are fitted using training rows.
By default the official loader filters out test rows and limits clean history to validation
origins. The explicit `final-test` command opens test rows only after fitting and selection.

### Saving and reusing fitted models

We save the selected models before scoring, under
`models/<parent-run-id>/` at the project root, outside the ignored data directory.
Each fit gets its own directory, and MLflow
stores the same bundle under the parent run's `models` artifacts. The metrics JSON
records `models_path`, so we can connect predictions to the exact fitted objects.

The bundle contains:

- `mlp_weights.pt`: the MLP checkpoint selected by validation pinball loss.
- `preprocessors.pkl`: the fitted linear pipeline and the fitted LightGBM and MLP
  preprocessors, including training medians, scaling and category encodings.
- `lightgbm.txt`: the native tree model at its selected iteration.
- `manifest.json`: the MLP state-to-index map and architecture, momentum coefficient,
  feature order, quantiles, training residual p90, fit run IDs, source/input hashes,
  package versions and checksums for the saved files.

Loading does not train models, start an MLflow run, or open test data. Use the locked
environment and load only our trusted bundles: the sklearn objects use Python pickle.
Checksums detect accidental file damage; they do not establish who created a bundle.
The feature table and cleaned history must match the fit, because saved preprocessors
do not replace the shared feature-generation code.

After correcting the cross-year holiday feature, we reran the ladder on 2026-11-14.
The fitted bundle is `models/1ba4b8f0b32c4c0389375a41859d4b2b/`, from MLflow parent
run `1ba4b8f0b32c4c0389375a41859d4b2b`. The run refreshed
`data/processed/ladder_metrics.json` and `data/processed/ladder_predictions.parquet`.
Reloading the bundle reproduced all **125,429 validation rows exactly**, including
every MLP quantile. We did not score test data in this rerun or reload check.

We can use that saved bundle directly, without another fit:

```bash
uv run python - <<'PY'
from pathlib import Path

from smartops.features import load_model_rows
from smartops.ladder_models import load_ladder_models

processed = Path("data/processed")
models = load_ladder_models(Path("models/1ba4b8f0b32c4c0389375a41859d4b2b"))
rows = load_model_rows(processed / "features.parquet")  # excludes test by default
validation = rows.loc[
    rows["period"].eq("val") & rows["y"].notna() & rows["beds_lag1"].notna()
]
predictions = models.predict(validation)
predictions.to_parquet(processed / "reloaded_validation_predictions.parquet", index=False)
PY
```

`predict()` preserves input order and hospital groups and returns all four point
forecasts plus p10/p50/p90. It only needs engineered inputs and finite `beds_lag1`;
labels are optional. Missing numeric features use saved training medians, and unseen
states keep the reserved MLP code. After moving a bundle to another machine, pass its
new directory to `load_ladder_models()` instead of the old absolute `models_path`.
For future `ladder` runs, read `models_path` from `data/processed/ladder_metrics.json`
to select that run's bundle. We keep the fitted bundles in [models/](models/) so they
can be committed to GitHub and loaded by teammates without refitting. Generated data
and local MLflow logs remain excluded from Git.

Older runs saved predictions and metrics but not fitted weights. We cannot recover
those weights from predictions. A new fit creates a new bundle and run ID; it does not
retroactively add models to an earlier evaluation.

The 2026-11-14 bundle and validation scores use the corrected cross-year holiday flag.
The loader checks each day's calendar year, so a week starting 2023-12-31 includes
New Year's Day 2024. The test results below still come from the earlier pre-fix fit;
we have not rescored test after this correction.

## Final test results

The earlier official test evaluation covered **2023-07-02 through 2024-04-21**.
Its models were fitted before the cross-year holiday correction and its test metrics
remain historical; they have not been refreshed by the 2026-11-14 validation rerun.
Models were fitted on training weeks from training hospitals only, with iteration/checkpoint
selection on validation.
Weights, preprocessing and the persistence error quantile stayed fixed for test; there was
no train-plus-validation refit or test-based weight updates in this procedure. Each
forecast is one week ahead and may use observations from earlier test weeks as
origin-time history.

Every official model uses the same **165,195 test rows from training hospitals** and
**41,844 test rows from unseen hospitals**. Missing targets or persistence inputs exclude
2,247 and 479 rows respectively; no additional eligible rows are dropped by a model.

| Model | Training-hospital test MAE | Skill vs persistence | Unseen-hospital test MAE | Skill vs persistence |
|---|---:|---:|---:|---:|
| Persistence | 4.859 | 0.00% | 4.734 | 0.00% |
| Momentum | 4.747 | 2.29% | 4.629 | 2.23% |
| Linear | 4.654 | 4.20% | 4.534 | 4.24% |
| LightGBM check | 4.414 | 9.15% | 4.303 | 9.12% |
| Quantile MLP median | 4.473 | 7.94% | 4.365 | 7.80% |

LightGBM beats persistence for 83.58% of training hospitals and 83.89% of unseen hospitals.
Linear, LightGBM and the MLP pass the proposed point-accuracy and generalization checks;
momentum falls below the proposed 3% skill threshold.

| MLP uncertainty/staffing metric | Training hospitals | Unseen hospitals |
|---|---:|---:|
| 80% interval coverage | 80.13% | 80.15% |
| Mean 80% interval width (beds) | 13.691 | 13.399 |
| p10 pinball loss | 1.147 | 1.114 |
| p50 pinball loss | 2.237 | 2.183 |
| p90 pinball loss | 1.068 | 1.067 |
| MLP p90 shortfall rate | 10.26% | 10.15% |
| Persistence-plus-error shortfall rate | 9.79% | 9.19% |
| MLP mean shortfall (beds) | 0.399 | 0.408 |
| Persistence-plus-error mean shortfall (beds) | 0.873 | 0.826 |
| MLP mean idle beds | 7.094 | 6.991 |
| Persistence-plus-error mean idle beds | 8.096 | 8.056 |

The MLP passes the proposed 80% calibration check but still fails the staffing check:
shortfall frequency is higher despite lower shortfall magnitude and fewer idle beds.
A 90% interval remains unavailable because these models output p10/p50/p90 only.

Evaluation compares the four ladder models with persistence on the full eligible
population for each hospital group. It requires only the shared feature table and
ladder predictions; there is no separate exploratory-model comparison.

Full outputs:

- `data/processed/ladder_test_metrics.json`
- `data/processed/ladder_test_predictions.parquet`
- `data/processed/final_test/evaluation.json`
- `data/processed/final_test/metrics.csv`
- `data/processed/final_test/evaluation_rows.parquet`

MLflow parent run: `c3682f61463946cc9476d30f3dfa1e11` (`ladder-final-test`), with
`test-persistence`, `test-momentum`, `test-linear`, `test-lightgbm` and `test-quantile_mlp`
score runs. New `final-test-comparison` runs contain the independently rescored ladder
comparison. Older runs preserve the experiment roster used when they were recorded.

The `uv run smartops final-test` command reruns training, validation selection and test
scoring. **Do not run it again until the team agrees.** The results above are from the
saved evaluation; adding model persistence does not update those scores.

## Completed audits

`smartops audit` produces these machine-readable reports in `data/processed/audit/`:

- `data/processed/audit/evaluation.json`: common-row scores, row counts,
  exclusions, row SHA256s and the proposal's pass/fail checks.
- `data/processed/audit/evaluation_rows.parquet`: exact rows and
  predictions behind each comparison. Missing official predictions fail the audit.
- `data/processed/audit/leakage_audit.json`: 17 checks over all 968,688
  feature rows; zero mismatches in lags, next-week labels, dates, split membership,
  calendar values or origin-time state/subtype attributes.
- `data/processed/audit/mlflow_audit.json`: run inventory, artifacts,
  provenance, lifecycle status and historical repairs/imports.

Our leakage review corrected a latent code path: the cleaner now carries
attributes forward from reports available at each week instead of copying the last
report backward. Ladder preprocessing and state vocabularies use training rows only.
The supplied state and subtype values already match their as-of values, so the
feature table did not need to be rebuilt. Tests perturb future attributes, targets
and history and verify that earlier inputs remain unchanged. Publication/revision
timestamps are absent, so the audit cannot certify real-time source availability.
The hospital holdout roster is fixed retrospectively for this evaluation.

Earlier experiments had already scored test data before this audit, so the project
cannot claim an untouched test set. Current validation and test reports compare only
the ladder models and persistence, retaining the full eligible population.

Our historical MLflow audit corrected one run explicitly tagged `FAILED` but
incorrectly marked `FINISHED`. Saved experiment bundles were imported as
`legacy_artifact_import` runs, with unknown original run times and no claim of
reproduced training. Original artifacts are preserved in those runs. Unlogged
historical attempts without surviving artifacts cannot be reconstructed.

To repeat the idempotent historical repair/import:

```bash
uv run smartops audit --repair-tracking --backfill
```

Backfill now considers the saved ladder bundle only.
Existing MLflow records remain historical evidence of past runs, including retired
experiments; they are not part of the active model roster or current comparisons.
Newly changed artifact bundles receive new imports; existing signatures are skipped.
MLflow uses `data/processed/mlflow.db` by default. Test runs
use temporary tracking databases and do not enter the project experiment inventory.

## Reproducing the pipeline and preparing a submission

### Rebuilding generated data

**We can rebuild the data and rerun validation with the commands below, provided
the environment and source download are available.**
Running only [ladder_models.py](src/smartops/ladder_models.py) is not an end-to-end entry
point: it defines model functions and has no script entry point. Use the `smartops` CLI
from the project root.

We have **not** verified a full delete-and-rebuild with a fresh dependency install
and source download. The completed final evaluation
used the existing data. The automated tests
and matching independent scores validate those paths, but are not a guarantee that an
external download or a new machine's installation will succeed.

After deleting generated files, run `data` and `ladder` before `audit`: the audit
requires cleaned history, features and ladder validation predictions. `final-test`
fits the ladder, generates its own test predictions and rescores those predictions.
Neither command requires prediction files from any separate model experiment.

### Data and validation command order

Use Python 3.12, as specified by [.python-version](.python-version), and
[uv](https://docs.astral.sh/uv/). Preserve [pyproject.toml](pyproject.toml) and
[uv.lock](uv.lock); the lock file records the resolved dependency versions.
The download requires internet access and approximately 481 MB for the raw CSV alone;
allow additional space for the environment, derived tables, models and MLflow artifacts.

Run each command from the project root and proceed only after it succeeds:

```bash
uv sync --locked
uv run smartops info
uv run smartops data
uv run smartops ladder
uv run smartops audit
```

`data` performs download/verification, ingestion, cleaning and feature generation in
order and creates missing output directories. `ladder` creates the validation artifacts
required by `audit`, including the retained ladder LightGBM, and saves the fitted models.
The separate `baselines` command also scores test data, so we keep it out of this
validation-only sequence while test evaluation is on hold. `final-test` repeats the fixed
training/validation procedure, scores test, writes the comparison and refreshes the
MLflow inventory. Fresh fits create new run IDs and may differ across package/hardware
environments; no fresh-run duration is guaranteed.

For verification after setup:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Model-persistence verification passed all 35 automated tests, lint and formatting
checks. The tests include saving/loading the full ladder with synthetic data, unseen
states and missing numeric inputs, plus rejection of damaged bundles. They use
temporary tracking databases; the separate real-data check above uses validation only.

Historical `audit --repair-tracking --backfill` is for preserving/importing old artifacts;
it is not needed to log a newly executed full pipeline.

### What to preserve before removing generated data

For a submission rebuild, use a separate copy rather than erase the only evaluated copy.
Keep the code, tests, documentation, dependency files and **[data/raw/SHA256SUMS](data/raw/SHA256SUMS)**.
The reference raw-data SHA256 is:

```text
b3ef37e7e8d9888ff241caab83ec43be7e26be3c592a5a4e120acbf541edea7f
```

Deleting the whole data directory also deletes that reference checksum. The downloader
will then record the newly downloaded file's checksum, which verifies that download
against itself rather than proving it matches the evaluated snapshot. Keep the original
checksum when omitting the large CSV from the submission. If the CSV already exists
but its checksum file is missing, verification fails.

Back up `data/processed/` (including `mlflow.db`), [models/](models/) and `mlruns/`
to retain the submitted run evidence. Deleting the database loses the old run
metadata; retraining creates new runs and does not reconstruct historical failures.
Artifact locations can contain absolute paths, so copying the database alone is not a
portable MLflow archive. Close a running local MLflow server before moving its database.

### Viewing MLflow experiments

From the project root:

```bash
uv run mlflow ui --backend-store-uri sqlite:///data/processed/mlflow.db --port 5000
```

Open `http://127.0.0.1:5000`, select **Model training**, and open
**smartops-model-ladder**. The GenAI trace overview is not the model-run table.
Parameters, metrics and run status are stored in `data/processed/mlflow.db`;
local artifacts for the recorded experiment are under `mlruns/`.
This is local tracking; an online tracking service has not been configured.

## Repository guide

| Location | Purpose |
|---|---|
| [src/smartops/data](src/smartops/data/) | Download, ingestion and cleaning |
| [features.py](src/smartops/features.py) | Shared target, lag table and causal engineered features |
| [ladder_models.py](src/smartops/ladder_models.py) | Official momentum, linear, LightGBM and quantile MLP fitting/scoring |
| [evaluate.py](src/smartops/evaluate.py) / [audit.py](src/smartops/audit.py) | Shared metrics, prediction alignment, leakage and tracking audits |
| [tracking.py](src/smartops/tracking.py) | MLflow run lifecycle and provenance |
| [cli.py](src/smartops/cli.py) | Commands for reproducing the pipeline |
| [tests](tests/) | Automated tests, including leakage, scoring and frozen test evaluation |
| `data/processed/` | Generated features, predictions, metrics and tracking database |
| [models/](models/) | Saved model bundles, including fitted preprocessors and state maps; ready to commit |
| [docs/steps/02-data.md](docs/steps/02-data.md) | Earlier data-stage decisions and handoff; this README describes the final modeling protocol |

## Credits and AI assistance

The data source is the US HHS facility-level hospital-capacity dataset linked above.
We used AI coding assistance for implementation, audits and documentation. Reported
metrics come from executed experiments and saved evaluation artifacts; methodological
limitations and unsuccessful checks are included alongside the improvements.
