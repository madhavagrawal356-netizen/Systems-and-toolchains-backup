# Step 2: data pipeline and shared feature table

Lead: Jiyao · Branch: `step-02-data` · Status: ready for review · Next: Model (Madhav)

## Purpose

1. Turn the raw HHS facility file into one clean, validated table, and make every count we report
   reproducible with one command. This is "data done" for the 11/11 status check and the
   25-point Data part of the rubric.
2. Build the **one shared feature table** that every baseline and model trains and is scored on,
   so that the model-vs-baseline comparison is fair by construction.
3. Give the first baseline numbers (persistence, seasonal naive) for the status check.

## What to run

```bash
uv sync
uv run smartops data        # download (or verify) → ingest → clean → features, ~10 s after download
uv run smartops baselines   # persistence and seasonal naive MAE on val and test
```

The download is 481 MB and takes about 8 minutes. If you already have the CSV, put it in
`data/raw/`. `smartops verify` checks it against the committed `data/raw/SHA256SUMS`.

## Outline of the pipeline

| Command | Writes | Does |
|---|---|---|
| `download` / `verify` | `data/raw/<csv>` | Streams the file and verifies SHA256 |
| `ingest` | `data/interim/raw.parquet` (80 MB) | Types all 128 columns, keeps IDs as text, removes nothing, checks every CSV record became a row |
| `clean` | `data/clean/beds_weekly.parquet` (12 MB), `hospitals.parquet`, `counts.json` | Resolves sentinels, flags bad rows, builds the weekly grid, assigns splits, writes counts |
| `features` | `data/processed/features.parquet` (14 MB) | One row per forecast: next week's target and the proposal's inputs |
| `baselines` | `data/processed/baselines.json` | Persistence and seasonal naive MAE |

Code: `src/smartops/data/{download,ingest,clean}.py`, `src/smartops/features.py`,
`src/smartops/baselines.py`. Tests: `tests/test_data.py` and `tests/test_features.py` run on
small hand-built CSVs, one row per case, so CI never downloads the real file. 21 tests.

## What we found, and what we decided

### 1. Most "-999999" values are not lost: we recover them exactly

HHS hides a 7-day **average** when it is small, but usually still publishes the 7-day **sum** and
the number of days reported (**coverage**). In every row where all three are published, the
average equals sum ÷ coverage up to rounding to one decimal (0 rows differ by more than 0.06). So:

| Target status | Rows (all weeks) | Meaning | `beds_used` | `beds_used_filled` |
|---|---|---|---|---|
| `measured` | 938,908 | Reported average | value | value |
| `derived` | 80,608 | Average hidden, sum shown → sum ÷ coverage | value | value |
| `suppressed` | 5,278 | Sum hidden too (a sum of 1 to 3) → truly unknown | NULL | 2 ÷ coverage (midpoint) |
| `missing` | 20,612 | Not reported (NULL, coverage 0) | NULL | NULL |

So the proposal's "85,886 sentinel rows (8.2%)" is really **5,278 unknown rows (0.5%)**. This gives
the ablation a clear story: drop all 85,886 vs recover 80,608 and handle only 5,278.

Measured and filled values never share a column. Every measure (capacity, COVID, flu, ICU) gets the
same three columns: `<m>`, `<m>_filled`, `<m>_status`.

### 2. Nothing is deleted; suspect rows are flagged

| Flag | Rows | Note |
|---|---|---|
| `flag_over_capacity` | 4,475 | `beds_used > capacity`. Median excess 3 beds, 99th pct 62. **Open decision** |
| `flag_low_coverage` | 60,752 | Reported on only 1 to 6 of 7 days |
| `flag_zero_capacity` | 75 | Beds in use but capacity 0 or missing |
| `is_corrected` | 39,492 | HHS corrected the row after first release (from the source) |

The proposal said 4,359 over-capacity rows. We now find 4,475 because recovering sentinels makes
more rows comparable (4,368 measured + 107 derived).

### 3. A complete weekly grid, so lags are right

1,501 hospitals skip at least one week (9,221 missing weeks in total). If lags were "the previous
row", lag 1 would sometimes be two weeks back. `beds_weekly` has one row per hospital per week from
its first to its last report. The 9,221 inserted rows have `reported = false` and status
`not_reported`. `smartops features` refuses to run if any step is not exactly 7 days.

### 4. Splits are assigned once, for everyone

- `hospital_split`: 1,034 of 5,172 hospitals (20%) are `holdout`. They are chosen by sorting
  md5(hospital_pk + seed 42), so every machine holds out the same hospitals.
- `period`: `train` 2020-08-02 to 2022-12-25, `val` 2023-01-01 to 2023-06-25, `test` 2023-07-02
  to 2024-04-21; `pre` is earlier and only used as lag history. Dates are in `config.py`.
- **In the feature table, a sample's period is the period of its target week.** No val or test
  target is ever trained on. (This settles the boundary question from the first draft of this doc.)

### 5. The shared feature table follows the proposal exactly

The proposal fixes the inputs ("lagged occupancy (1 to 4 and 52 weeks), capacity, week of year,
state, and lagged COVID and flu counts") and the baselines (persistence = lag 1, seasonal naive =
lag 52, linear model "on the same lags"). So the table is shared, not per-model:

- One row = one forecast: standing at the end of `origin_week`, predict `y` = beds in
  `target_week` = origin + 7 days.
- Lags count back from the target week: `beds_lag1` = beds in the origin week, `beds_lag52` =
  beds 52 weeks before the target week (same week last year).
- Unknown inputs stay NULL. How to handle them (drop, fill, add an indicator) is each model's
  choice and part of the ablation.

| | Rows |
|---|---|
| Feature rows (target week from 2020-08-02) | 968,688 |
| with a known target `y` | 955,517 |
| with `y` and `beds_lag1` (persistence can predict) | 950,464 |
| with `y` and `beds_lag52` (seasonal naive can predict) | 751,658 |
| with `y` and all five occupancy lags | 743,098 |

`beds_lag52` is missing mostly in the first year of training, because the data only starts in
mid-2020.

## First baseline numbers (MAE in beds)

From `data/processed/baselines.json`. The average hospital has 106 occupied beds in val (median 35).

| Period / hospitals | n (persistence) | Persistence | n (common) | Persistence | Seasonal naive |
|---|---|---|---|---|---|
| val / train hospitals | 100,151 | **4.75** | 98,794 | 4.79 | 12.40 |
| val / holdout hospitals | 25,278 | **4.68** | 25,022 | 4.70 | 12.04 |
| test / train hospitals | 165,195 | 4.86 | 162,872 | 4.89 | 11.32 |
| test / holdout hospitals | 41,844 | 4.73 | 41,189 | 4.77 | 11.18 |

"Common" rows are those where both baselines can predict, so those two columns compare directly.
Persistence is far ahead of seasonal naive. This confirms the proposal's risk: **persistence is the
number to beat, at about 4.7 beds**. These are a sanity check; the official baseline and model
numbers are the Model lead's.

## Handoff to Madhav (Model)

**Done for you (shared, please do not redefine):** the target, the split, the proposal's inputs,
and the two naive baselines. If all models read `features.parquet`, the comparison is fair.

**Yours:** the linear baseline, the PyTorch quantile MLP, MLflow, how to encode `state` and
`week_of_year`, how to handle NULL inputs, and any extra features. To add a feature, add a column
in `src/smartops/features.py` (so everyone gets it) or build it in your own code from
`beds_weekly`. Either way, keep the rows and the target unchanged.

**Rules for the rows:**

| To... | Use |
|---|---|
| train | `period = 'train' AND hospital_split = 'train' AND y IS NOT NULL` |
| tune / early-stop | `period = 'val' AND hospital_split = 'train'` |
| report "unseen weeks" | `period = 'test' AND hospital_split = 'train'` |
| report "unseen weeks and hospitals" | `period = 'test' AND hospital_split = 'holdout'` |
| target | `y` by default; `y_filled` only for the suppression ablation |

Never train on `holdout` hospitals in any period. Rows with NULL `y` cannot be trained or scored,
but they are already used where they matter: as lag history inside the table.

**Starter:**

```python
import duckdb

train = duckdb.sql("""
    SELECT * FROM 'data/processed/features.parquet'
    WHERE period = 'train' AND hospital_split = 'train' AND y IS NOT NULL
""").df()
```

## Column dictionary

### `data/processed/features.parquet` (one row per forecast)

| Column | Meaning |
|---|---|
| `hospital_pk`, `origin_week`, `target_week` | Keys. `target_week = origin_week + 7 days` |
| `period`, `hospital_split` | Split of the target week / hospital (see rules above) |
| `y` | Target: beds used in the target week (measured or derived), NULL if unknown |
| `y_filled` | `y`, plus the midpoint fill for suppressed weeks (ablation only) |
| `y_status` | `measured`, `derived`, `suppressed`, `missing` or `not_reported` |
| `y_capacity`, `y_flag_over_capacity` | Target-week capacity and over-capacity flag (for the staffing analysis and ablation, **not inputs**) |
| `beds_lag1` … `beds_lag4`, `beds_lag52` | Beds used 1 to 4 and 52 weeks before the target week |
| `beds_lag1_status` | Status of the origin week (why `beds_lag1` may be NULL) |
| `capacity_lag1`, `covid_lag1`, `flu_lag1` | Capacity, COVID beds and flu patients in the origin week |
| `week_of_year` | ISO week of the target week (1 to 53) |
| `state` | Hospital state |
| `hospital_subtype` | Metadata for error analysis (not a proposal input) |

### `data/clean/beds_weekly.parquet` (one row per hospital per week)

| Column | Meaning |
|---|---|
| `hospital_pk`, `week` | Keys (week = HHS collection week, a Sunday) |
| `period`, `hospital_split` | Split of this week / hospital |
| `state`, `hospital_name`, `city`, `hospital_subtype`, `is_metro_micro` | From the hospital's latest report |
| `reported` | False for the 9,221 weeks inserted into gaps |
| `beds_used`, `capacity`, `total_beds`, `covid_beds`, `flu_patients`, `icu_used`, `icu_capacity` | Measured or derived 7-day averages; NULL if unknown |
| `<measure>_filled` | The same, plus the midpoint fill for suppressed values |
| `<measure>_status` | `measured`, `derived`, `suppressed`, `missing`, `not_reported` |
| `beds_used_coverage` | Days of the week the target was reported (0 to 7) |
| `is_corrected`, `flag_*` | See "Nothing is deleted" above |

### `data/clean/hospitals.parquet` (one row per hospital)

`hospital_pk`, attributes as above, `first_week`, `last_week`, `n_reports`, `hospital_split`.
This is also the lookup table for the LLM's hospital-name matching.

## Other things the model team should know

- **The column count is 128, not 129.** The healthdata.gov web page counts a hidden system column
  (`:@computed_region_pqdx_y6mm`) that the CSV does not contain. Fix this in the slides and doc.
- **COVID and flu inputs are sparse early.** COVID beds: 76,530 suppressed and 134,535 missing
  rows. Flu is missing for 72% of 2020 rows.
- Subtypes: Short Term 672,981 rows, Critical Access 282,494, Long Term 70,429, Children's 19,502.
  Long Term and Children's hospitals are candidates for the failure case.
- MAE in beds is dominated by large hospitals (mean 106, median 35 beds). Consider also reporting
  error relative to capacity.

## Open decisions for the team

1. **Over-capacity rows (4,475):** keep, cap at capacity, or drop? Proposal: keep for training and
   report the effect in the ablation.
2. **Suppressed target (4,358 rows in window):** drop, or train on `y_filled`? This is the main
   ablation (Suhani).

## For the zip (Suhani)

The raw CSV is 481 MB: under the 500 MB limit on its own, but not with everything else. Ship
`download` + `SHA256SUMS` as planned. `data/clean/` is about 12 MB and can go in the zip.

## AI use

The pipeline code, tests and this document were drafted with Claude (Claude Code), then checked by
running them on the full dataset and against hand-built test cases.
