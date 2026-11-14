"""The shared feature table every model and baseline trains and is scored on.

One row is one forecast: standing at the end of `origin_week`, predict occupied beds in
`target_week` (= origin_week + 7 days). The inputs are exactly the ones the proposal fixes:
occupancy lags 1-4 and 52, capacity, week of year, state, and lagged COVID and flu counts.

Lags are counted back from the target week, so
    beds_lag1  = beds in origin_week          (persistence predicts y = beds_lag1)
    beds_lag52 = beds 52 weeks before target  (seasonal naive predicts y = beds_lag52)

The table is built from data/clean/beds_weekly.parquet, whose weekly grid has no gaps, so
"k rows back" is always "k weeks back". A sample's `period` and `hospital_split` are those of
its target week, so no val or test target is ever trained on.

Values a model may not see are kept as NULL (not filled): each model decides how to handle
them, and the ablation measures what that choice does.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from smartops.config import TRAIN_START
from smartops.data.clean import DataValidationError, period_sql

OCCUPANCY_LAGS = (1, 2, 3, 4, 52)

# Columns a model may use as inputs. Everything else in the table is a key, a target, or metadata.
FEATURES = (
    *(f"beds_lag{k}" for k in OCCUPANCY_LAGS),
    "capacity_lag1",
    "week_of_year",
    "state",
    "covid_lag1",
    "flu_lag1",
)


def _features_sql(clean_path: Path) -> str:
    # beds_lag{k} sits k weeks before the target, i.e. k-1 rows before the origin row.
    lags = ",\n".join(
        f"lag(beds_used, {k - 1}) OVER w AS beds_lag{k}" if k > 1 else "beds_used AS beds_lag1"
        for k in OCCUPANCY_LAGS
    )
    return f"""
        CREATE TABLE features AS
        WITH f AS (
            SELECT hospital_pk,
                   week AS origin_week,
                   lead(week) OVER w AS target_week,
                   -- targets (next week)
                   lead(beds_used) OVER w AS y,
                   lead(beds_used_filled) OVER w AS y_filled,
                   lead(beds_used_status) OVER w AS y_status,
                   lead(capacity) OVER w AS y_capacity,
                   lead(flag_over_capacity) OVER w AS y_flag_over_capacity,
                   -- inputs (known at the end of the origin week)
                   {lags},
                   beds_used_status AS beds_lag1_status,
                   capacity AS capacity_lag1,
                   covid_beds AS covid_lag1,
                   flu_patients AS flu_lag1,
                   state,
                   -- metadata
                   hospital_subtype,
                   hospital_split
            FROM '{clean_path}'
            WINDOW w AS (PARTITION BY hospital_pk ORDER BY week)
        )
        SELECT hospital_pk, origin_week, target_week,
               {period_sql("target_week")} AS period,
               hospital_split,
               y, y_filled, y_status, y_capacity, y_flag_over_capacity,
               {", ".join(f"beds_lag{k}" for k in OCCUPANCY_LAGS)},
               beds_lag1_status, capacity_lag1,
               weekofyear(target_week) AS week_of_year,
               state, covid_lag1, flu_lag1,
               hospital_subtype
        FROM f
        WHERE target_week IS NOT NULL AND target_week >= DATE '{TRAIN_START}'
        ORDER BY hospital_pk, target_week
    """


def build_features(clean_path: Path, out_path: Path) -> dict:
    """Write the feature table. Returns (and prints) how many rows have each piece."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    gaps = con.execute(
        f"""SELECT count(*) FROM (
                SELECT week - lag(week) OVER (PARTITION BY hospital_pk ORDER BY week) AS d
                FROM '{clean_path}') WHERE d <> 7"""
    ).fetchone()[0]
    if gaps:
        raise DataValidationError(f"beds_weekly has {gaps} non-weekly steps; rerun clean")

    con.execute(_features_sql(clean_path))
    con.execute(f"COPY features TO '{out_path}' (FORMAT parquet, COMPRESSION zstd)")

    def one(sql: str) -> int:
        return int(con.execute(f"SELECT count(*) FROM features {sql}").fetchone()[0])

    summary = {
        "rows": one(""),
        "rows_with_target_y": one("WHERE y IS NOT NULL"),
        "rows_with_target_and_lag1": one("WHERE y IS NOT NULL AND beds_lag1 IS NOT NULL"),
        "rows_with_target_and_lag52": one("WHERE y IS NOT NULL AND beds_lag52 IS NOT NULL"),
        "rows_with_target_and_all_occupancy_lags": one(
            "WHERE y IS NOT NULL AND "
            + " AND ".join(f"beds_lag{k} IS NOT NULL" for k in OCCUPANCY_LAGS)
        ),
    }
    print(f"wrote {out_path} ({summary['rows']:,} rows)")
    for k, v in summary.items():
        print(f"  {v:>12,}  {k}")
    return summary


NUMERIC_FEATURES = tuple(name for name in FEATURES if name != "state")
INPUT_FEATURES = (*NUMERIC_FEATURES, "state")
STAGE_NUMERIC_FEATURES = (
    "momentum",
    "slope_4w",
    "mean_reversion_distance_8w",
    "occupancy_rate_lag1",
    "holiday_week",
    "week_sin",
    "week_cos",
)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = (
        date(year, month + 1, 1) - timedelta(days=1)
        if month < 12
        else date(year + 1, 1, 1) - timedelta(days=1)
    )
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _holiday_dates(year: int) -> set[date]:
    # Federal holidays; observed dates are included when the holiday lands on a weekend.
    fixed = [date(year, month, day) for month, day in ((1, 1), (6, 19), (7, 4), (11, 11), (12, 25))]
    holidays = set(fixed)
    holidays.update(
        {
            _nth_weekday(year, 1, 0, 3),
            _nth_weekday(year, 2, 0, 3),
            _last_weekday(year, 5, 0),
            _nth_weekday(year, 9, 0, 1),
            _nth_weekday(year, 10, 0, 2),
            _nth_weekday(year, 11, 3, 4),
        }
    )
    for day in fixed:
        if day.weekday() == 5:
            holidays.add(day - timedelta(days=1))
        elif day.weekday() == 6:
            holidays.add(day + timedelta(days=1))
    return holidays


def load_model_rows(
    features_path: Path, history_path: Path | None = None, *, include_test: bool = False
) -> pd.DataFrame:
    """Build causal inputs; test rows require an explicit final-evaluation opt-in."""
    periods = ["train", "val", "test"] if include_test else ["train", "val"]
    df = pd.read_parquet(features_path, filters=[("period", "in", periods)])
    required = {
        "hospital_pk",
        "origin_week",
        "target_week",
        "period",
        "hospital_split",
        "y",
        "beds_lag1",
        "beds_lag2",
        *INPUT_FEATURES,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"features.parquet is missing columns: {missing}")

    df = df.loc[df["period"].isin(periods) & df["hospital_split"].isin(["train", "holdout"])].copy()
    for col in ("y", "beds_lag1", "beds_lag2", *NUMERIC_FEATURES):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("origin_week", "target_week"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["change"] = df["y"] - df["beds_lag1"]
    df["momentum"] = df["beds_lag1"] - df["beds_lag2"]
    df["slope_4w"] = (
        3 * df["beds_lag1"] + df["beds_lag2"] - df["beds_lag3"] - 3 * df["beds_lag4"]
    ) / 10.0
    df["occupancy_rate_lag1"] = df["beds_lag1"] / pd.to_numeric(
        df["capacity_lag1"], errors="coerce"
    ).replace(0, np.nan)
    history_path = Path(
        history_path or (Path(features_path).parent.parent / "clean" / "beds_weekly.parquet")
    )
    if history_path.exists():
        history = pd.read_parquet(
            history_path,
            columns=["hospital_pk", "week", "beds_used"],
            filters=[("week", "<=", df["origin_week"].max())],
        )
        history["week"] = pd.to_datetime(history["week"], errors="coerce")
        history["beds_used"] = pd.to_numeric(history["beds_used"], errors="coerce")
        history = history.sort_values(["hospital_pk", "week"])
        gaps = history.groupby("hospital_pk")["week"].diff().dropna()
        if not gaps.eq(pd.Timedelta(days=7)).all():
            raise ValueError("Non-weekly clean history; cannot compute an 8-week mean")
        history["mean_8w"] = history.groupby("hospital_pk", sort=False)["beds_used"].transform(
            lambda values: values.rolling(8, min_periods=8).mean()
        )
        df = df.merge(
            history[["hospital_pk", "week", "mean_8w"]],
            left_on=["hospital_pk", "origin_week"],
            right_on=["hospital_pk", "week"],
            how="left",
            validate="many_to_one",
        ).drop(columns="week")
        df["mean_reversion_distance_8w"] = df["beds_lag1"] - df["mean_8w"]
        df = df.drop(columns="mean_8w")
    else:
        # The processed table itself contains origin-time observations on the weekly grid.
        history = df[["hospital_pk", "origin_week", "beds_lag1"]].sort_values(
            ["hospital_pk", "origin_week"]
        )
        gaps = history.groupby("hospital_pk")["origin_week"].diff().dropna()
        if not gaps.eq(pd.Timedelta(days=7)).all():
            raise ValueError("Non-weekly feature history; cannot compute an 8-week mean")
        means = history.groupby("hospital_pk")["beds_lag1"].transform(
            lambda values: values.rolling(8, min_periods=8).mean()
        )
        df["mean_reversion_distance_8w"] = df["beds_lag1"] - means.reindex(df.index)
    dates_by_year = {year: _holiday_dates(year) for year in range(2019, 2032)}
    # A target week can cross New Year: check each day's calendar, not the start year's.
    df["holiday_week"] = [
        int(
            any(
                day in dates_by_year.get(day.year, set())
                for day in (week.date() + timedelta(days=offset) for offset in range(7))
            )
        )
        if pd.notna(week)
        else 0
        for week in df["target_week"]
    ]
    df["week_sin"] = np.sin(2 * math.pi * (df["week_of_year"] - 1) / 52.0)
    df["week_cos"] = np.cos(2 * math.pi * (df["week_of_year"] - 1) / 52.0)
    return df
