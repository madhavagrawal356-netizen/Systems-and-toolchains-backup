"""Clean and validate the HHS facility data, and write the tables the model reads.

Decisions made here (see docs/steps/02-data.md for the numbers behind them):

1. The privacy sentinel -999999 is not one thing. HHS hides a 7-day *average* when it is small,
   but often still publishes the 7-day *sum* and the number of days reported (coverage). In those
   rows the average is exactly sum / coverage, so we recover it ("derived"). Only when the sum is
   hidden too (a sum of 1 to 3) is the value truly unknown ("suppressed").
2. Measured and filled-in values never share a column. `<m>` holds measured or derived values
   only; `<m>_filled` additionally fills suppressed rows with the midpoint (a sum of 2 over the
   days reported). `<m>_status` says which case each row is.
3. Nothing is deleted. Suspect rows are flagged (`flag_*`); which flags to drop is a modeling
   decision, measured in the ablation.
4. Each hospital gets one row per week from its first to its last report, with explicit rows for
   weeks it did not report, so "lag 1" always means the previous calendar week.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from smartops.config import (
    DATA_END,
    HOLDOUT_FRACTION,
    RANDOM_SEED,
    TEST_START,
    TRAIN_START,
    VAL_START,
)
from smartops.data.download import FILENAME as RAW_FILENAME

SENTINEL = -999999

# Output name -> HHS column stem. Each stem has _7_day_avg, _7_day_sum and _7_day_coverage.
MEASURES = {
    "beds_used": "inpatient_beds_used",  # the forecast target: all-cause occupied inpatient beds
    "capacity": "inpatient_beds",  # staffed inpatient beds available
    "total_beds": "total_beds",
    "covid_beds": "inpatient_beds_used_covid",
    "flu_patients": "total_patients_hospitalized_confirmed_influenza",
    "icu_used": "icu_beds_used",
    "icu_capacity": "total_icu_beds",
}
TARGET = "beds_used"
STATUSES = ("measured", "derived", "suppressed", "missing")

HOSPITAL_ATTRS = ("state", "hospital_name", "city", "hospital_subtype", "is_metro_micro")


class DataValidationError(ValueError):
    """A check that must hold for the cleaned data to be trusted failed."""


def period_sql(week: str) -> str:
    """SQL expression giving the evaluation period ('pre', 'train', 'val', 'test') of a week."""
    return f"""CASE WHEN {week} < DATE '{TRAIN_START}' THEN 'pre'
                    WHEN {week} < DATE '{VAL_START}' THEN 'train'
                    WHEN {week} < DATE '{TEST_START}' THEN 'val'
                    WHEN {week} <= DATE '{DATA_END}' THEN 'test' END"""


def _measure_sql(name: str, stem: str) -> str:
    a, s, c = f"{stem}_7_day_avg", f"{stem}_7_day_sum", f"{stem}_7_day_coverage"
    recoverable = f"{a} = {SENTINEL} AND {s} >= 0 AND {c} > 0"
    return f"""
        CASE WHEN {a} >= 0 THEN {a}
             WHEN {recoverable} THEN {s}::DOUBLE / {c} END AS {name},
        CASE WHEN {a} >= 0 THEN {a}
             WHEN {recoverable} THEN {s}::DOUBLE / {c}
             WHEN {a} = {SENTINEL} AND {c} > 0 THEN 2.0 / {c} END AS {name}_filled,
        CASE WHEN {a} IS NULL THEN 'missing'
             WHEN {a} >= 0 THEN 'measured'
             WHEN {recoverable} THEN 'derived'
             WHEN {a} = {SENTINEL} THEN 'suppressed'
             ELSE 'invalid' END AS {name}_status"""


def _reports_sql(interim_path: Path) -> str:
    measures = ",".join(_measure_sql(n, s) for n, s in MEASURES.items())
    attrs = ", ".join(HOSPITAL_ATTRS)
    return f"""
        CREATE TABLE reports AS
        SELECT hospital_pk,
               collection_week AS week,
               {attrs},
               is_corrected,
               inpatient_beds_used_7_day_coverage AS beds_used_coverage,
               {measures}
        FROM '{interim_path}'
    """


def _validate_reports(con: duckdb.DuckDBPyConnection) -> None:
    def one(sql: str) -> int:
        return con.execute(sql).fetchone()[0]

    dups = one(
        "SELECT count(*) FROM "
        "(SELECT 1 FROM reports GROUP BY hospital_pk, week HAVING count(*) > 1)"
    )
    if dups:
        raise DataValidationError(f"{dups} (hospital, week) pairs appear more than once")
    not_sunday = one("SELECT count(*) FROM reports WHERE dayofweek(week) <> 0")
    if not_sunday:
        raise DataValidationError(f"{not_sunday} rows have a collection_week that is not a Sunday")
    null_keys = one("SELECT count(*) FROM reports WHERE hospital_pk IS NULL OR week IS NULL")
    if null_keys:
        raise DataValidationError(f"{null_keys} rows have no hospital_pk or collection_week")
    for name in MEASURES:
        bad = one(f"SELECT count(*) FROM reports WHERE {name}_status = 'invalid'")
        if bad:
            raise DataValidationError(f"{bad} rows of {name} are negative but not the sentinel")


def _hospitals_sql() -> str:
    latest = ", ".join(f"arg_max({c}, week) AS {c}" for c in HOSPITAL_ATTRS)
    return f"""
        CREATE TABLE hospitals AS
        WITH h AS (
            SELECT hospital_pk, {latest},
                   min(week) AS first_week, max(week) AS last_week, count(*) AS n_reports,
                   count(DISTINCT state) AS n_states
            FROM reports GROUP BY hospital_pk
        ), ranked AS (
            SELECT *, row_number() OVER (ORDER BY md5(hospital_pk || ':{RANDOM_SEED}')) AS r,
                      count(*) OVER () AS n
            FROM h
        )
        SELECT * EXCLUDE (r, n),
               CASE WHEN r <= round(n * {HOLDOUT_FRACTION}) THEN 'holdout' ELSE 'train' END
                   AS hospital_split
        FROM ranked
        ORDER BY hospital_pk
    """


def _grid_sql() -> str:
    statuses = ", ".join(f"coalesce(r.{n}_status, 'not_reported') AS {n}_status" for n in MEASURES)
    values = ", ".join(f"r.{n}, r.{n}_filled" for n in MEASURES)
    # Attributes must be known at this week, including on inserted gap rows.
    # The hospital summary uses latest values for display only, never as features.
    attrs = ", ".join(
        f"last_value(r.{c} IGNORE NULLS) OVER "
        f"(PARTITION BY w.hospital_pk ORDER BY w.week "
        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS {c}"
        for c in HOSPITAL_ATTRS
    )
    return f"""
        CREATE TABLE beds_weekly AS
        WITH weeks AS (
            SELECT hospital_pk,
                   unnest(generate_series(first_week::TIMESTAMP, last_week::TIMESTAMP,
                                          INTERVAL 7 DAY))::DATE AS week
            FROM hospitals
        )
        SELECT w.hospital_pk, w.week,
               {period_sql("w.week")} AS period,
               h.hospital_split,
               {attrs},
               r.hospital_pk IS NOT NULL AS reported,
               {values},
               {statuses},
               r.beds_used_coverage,
               coalesce(r.is_corrected, false) AS is_corrected,
               coalesce(r.beds_used > r.capacity, false) AS flag_over_capacity,
               coalesce(r.beds_used_coverage BETWEEN 1 AND 6, false) AS flag_low_coverage,
               coalesce(r.beds_used > 0 AND coalesce(r.capacity, 0) <= 0, false)
                   AS flag_zero_capacity
        FROM weeks w
        JOIN hospitals h USING (hospital_pk)
        LEFT JOIN reports r ON r.hospital_pk = w.hospital_pk AND r.week = w.week
        ORDER BY w.hospital_pk, w.week
    """


def _counts(con: duckdb.DuckDBPyConnection, n_arrived: int, sha256: str | None) -> dict:
    def one(sql: str) -> int:
        return int(con.execute(sql).fetchone()[0])

    def by(sql: str) -> dict[str, int]:
        return {str(k): int(v) for k, v in con.execute(sql).fetchall()}

    status = {
        name: by(f"SELECT {name}_status, count(*) FROM reports GROUP BY 1 ORDER BY 2 DESC")
        for name in MEASURES
    }
    t = status[TARGET]
    in_window = f"week >= DATE '{TRAIN_START}'"
    pre = one(f"SELECT count(*) FROM reports WHERE NOT ({in_window})")
    win = f"FROM reports WHERE {in_window}"
    win_missing = one(f"SELECT count(*) {win} AND {TARGET}_status = 'missing'")
    win_suppressed = one(f"SELECT count(*) {win} AND {TARGET}_status = 'suppressed'")
    usable = one(f"SELECT count(*) {win} AND {TARGET} IS NOT NULL")

    grid_rows = one("SELECT count(*) FROM beds_weekly")
    reported = one("SELECT count(*) FROM beds_weekly WHERE reported")
    counts = {
        "source": {
            "file": f"data/raw/{RAW_FILENAME}",
            "sha256": sha256,
            "rows": n_arrived,
            "hospitals": one("SELECT count(*) FROM hospitals"),
            "first_week": str(con.execute("SELECT min(week) FROM reports").fetchone()[0]),
            "last_week": str(con.execute("SELECT max(week) FROM reports").fetchone()[0]),
        },
        "waterfall": [
            {"step": "rows arrived", "rows": n_arrived},
            {"step": f"minus weeks before {TRAIN_START} (kept only as lag history)", "rows": -pre},
            {"step": "minus target not reported (NULL)", "rows": -win_missing},
            {
                "step": "minus target suppressed (sum of 1-3, truly unknown)",
                "rows": -win_suppressed,
            },
            {"step": "= rows with a usable target (measured or derived)", "rows": usable},
        ],
        "target_status_all_weeks": t,
        "target_usable_in_window": {
            "measured": one(f"SELECT count(*) {win} AND {TARGET}_status = 'measured'"),
            "derived": one(f"SELECT count(*) {win} AND {TARGET}_status = 'derived'"),
            "filled_only_in_beds_used_filled": win_suppressed,
        },
        "feature_status_all_weeks": {k: v for k, v in status.items() if k != TARGET},
        "flags_on_reported_rows": {
            f: one(f"SELECT count(*) FROM beds_weekly WHERE {f}")
            for f in (
                "flag_over_capacity",
                "flag_low_coverage",
                "flag_zero_capacity",
                "is_corrected",
            )
        },
        "grid": {
            "rows": grid_rows,
            "reported_rows": reported,
            "unreported_weeks_inserted": grid_rows - reported,
            "hospitals_with_gaps": one(
                "SELECT count(DISTINCT hospital_pk) FROM beds_weekly WHERE NOT reported"
            ),
            "hospitals_reporting_in_more_than_one_state": one(
                "SELECT count(*) FROM hospitals WHERE n_states > 1"
            ),
        },
        "splits": {
            "hospitals": by("SELECT hospital_split, count(*) FROM hospitals GROUP BY 1 ORDER BY 1"),
            "usable_target_rows": by(
                f"""SELECT period || '/' || hospital_split, count(*) FROM beds_weekly
                    WHERE {TARGET} IS NOT NULL GROUP BY 1 ORDER BY 1"""
            ),
        },
    }
    if grid_rows - reported != one("SELECT count(*) FROM beds_weekly WHERE NOT reported"):
        raise DataValidationError("grid row accounting does not add up")
    if sum(s["rows"] for s in counts["waterfall"][:-1]) != usable:
        raise DataValidationError("waterfall does not add up")
    return counts


def clean(interim_path: Path, out_dir: Path, sha256: str | None = None) -> dict:
    """Build data/clean/ from data/interim/raw.parquet. Returns the counts it also writes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    n_arrived = con.execute(f"SELECT count(*) FROM '{interim_path}'").fetchone()[0]

    con.execute(_reports_sql(interim_path))
    _validate_reports(con)
    con.execute(_hospitals_sql())
    con.execute(_grid_sql())
    counts = _counts(con, n_arrived, sha256)

    for table in ("beds_weekly", "hospitals"):
        path = out_dir / f"{table}.parquet"
        con.execute(f"COPY {table} TO '{path}' (FORMAT parquet, COMPRESSION zstd)")
    (out_dir / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(
        f"wrote {out_dir}/beds_weekly.parquet ({counts['grid']['rows']:,} rows), "
        f"hospitals.parquet, counts.json"
    )
    return counts
