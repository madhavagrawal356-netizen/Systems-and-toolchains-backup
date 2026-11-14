"""Feature table and baseline tests.

Hospital A reports beds = i in week i for 60 weeks, so every lag has an obvious right answer.
Hospital B skips one week, so we can check a lag never jumps across a gap.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest
from test_data import row, target, week, write_csv

from smartops.baselines import score_baselines
from smartops.data.clean import clean
from smartops.data.ingest import ingest
from smartops.features import NUMERIC_FEATURES, build_features, load_model_rows


@pytest.fixture
def feats(tmp_path):
    rows = [row("A", week(i), **target(float(i), 7 * i, 7)) for i in range(60)]
    rows += [row("B", week(i), **target(100.0 + i, 700 + 7 * i, 7)) for i in (0, 1, 3, 4)]
    ingest(write_csv(tmp_path / "raw.csv", rows), tmp_path / "raw.parquet")
    clean(tmp_path / "raw.parquet", tmp_path / "clean")
    summary = build_features(tmp_path / "clean" / "beds_weekly.parquet", tmp_path / "f.parquet")
    rel = duckdb.sql(f"SELECT * FROM '{tmp_path}/f.parquet'")
    recs = [dict(zip(rel.columns, t, strict=True)) for t in rel.fetchall()]
    by_key = {(r["hospital_pk"], str(r["target_week"])): r for r in recs}
    return by_key, summary, tmp_path


def test_target_is_next_week_and_lags_count_back_from_it(feats):
    by_key, _, _ = feats
    r = by_key[("A", week(55))]
    assert str(r["origin_week"]) == week(54)
    assert r["y"] == 55
    assert [r[f"beds_lag{k}"] for k in (1, 2, 3, 4)] == [54, 53, 52, 51]
    assert r["beds_lag52"] == 55 - 52


def test_lag52_is_null_without_a_year_of_history(feats):
    by_key, _, _ = feats
    assert by_key[("A", week(51))]["beds_lag52"] is None
    assert by_key[("A", week(52))]["beds_lag52"] == 0


def test_lags_do_not_jump_across_a_missing_week(feats):
    by_key, _, _ = feats
    after_gap = by_key[("B", week(3))]  # origin week 2 was not reported
    assert after_gap["beds_lag1"] is None
    assert after_gap["beds_lag1_status"] == "not_reported"
    assert after_gap["beds_lag2"] == 101  # week 1, not week 0
    gap_target = by_key[("B", week(2))]
    assert gap_target["y"] is None and gap_target["y_status"] == "not_reported"


def test_inputs_and_metadata_are_present(feats):
    by_key, _, _ = feats
    r = by_key[("A", week(10))]
    assert r["capacity_lag1"] == 60.0
    assert r["state"] == "PA"
    assert r["period"] == "train"
    assert 1 <= r["week_of_year"] <= 53


def test_baselines_score_persistence_exactly(feats, tmp_path):
    _, _, path = feats
    # Move hospital A's last weeks into "val" by scoring a hand-made table instead.
    duckdb.sql(
        f"""COPY (SELECT 'val' AS period, 'train' AS hospital_split, * EXCLUDE (period,
                         hospital_split) FROM '{path}/f.parquet' WHERE hospital_pk = 'A')
            TO '{tmp_path}/val.parquet' (FORMAT parquet)"""
    )
    res = score_baselines(tmp_path / "val.parquet", tmp_path / "b.json")
    v = res["val/train_hospitals"]
    # y - lag1 is always 1; y - lag52 is always 52.
    assert v["persistence_all_rows"]["mae"] == 1.0
    assert v["common_rows"]["persistence_mae"] == 1.0
    assert v["common_rows"]["seasonal_naive_mae"] == 52.0


def test_holiday_week_includes_new_year_in_the_following_calendar_year(tmp_path):
    # Synthetic rows exercise the real loader; no project test-period data is opened.
    frame = pd.DataFrame({name: [1.0, 1.0] for name in NUMERIC_FEATURES})
    frame["hospital_pk"] = "A"
    frame["state"] = "PA"
    frame["period"] = "test"
    frame["hospital_split"] = "train"
    frame["y"] = 10.0
    frame["target_week"] = pd.to_datetime(["2023-12-31", "2024-01-07"])
    frame["origin_week"] = frame["target_week"] - pd.Timedelta(days=7)
    frame["week_of_year"] = frame["target_week"].dt.isocalendar().week
    path = tmp_path / "features.parquet"
    frame.to_parquet(path, index=False)

    rows = load_model_rows(path, include_test=True)

    # Dec 31-Jan 6 includes New Year's Day 2024; Jan 7-13 contains no holiday.
    assert rows["holiday_week"].tolist() == [1, 0]
