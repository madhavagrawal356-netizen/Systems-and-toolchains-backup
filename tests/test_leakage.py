import pandas as pd
from test_data import row, target, week, write_csv

from smartops.data.clean import clean
from smartops.data.ingest import ingest
from smartops.features import STAGE_NUMERIC_FEATURES, build_features, load_model_rows


def make_features(tmp_path):
    records = [row("A", week(i), **target(float(i), 7 * i, 7)) for i in range(80)]
    records[-1]["state"] = "NY"
    records[-1]["hospital_subtype"] = "Changed in future"
    ingest(write_csv(tmp_path / "raw.csv", records), tmp_path / "raw.parquet")
    clean(tmp_path / "raw.parquet", tmp_path / "clean")
    build_features(tmp_path / "clean/beds_weekly.parquet", tmp_path / "features.parquet")
    return pd.read_parquet(tmp_path / "features.parquet")


def test_future_attributes_do_not_rewrite_history(tmp_path):
    frame = make_features(tmp_path)
    assert frame.iloc[0].state == "PA"
    assert frame.iloc[0].hospital_subtype == "Short Term"
    assert frame.state.eq("PA").all()  # last report is only a target, never an origin


def test_future_and_target_perturbation_preserves_earlier_features(tmp_path):
    frame = make_features(tmp_path)
    original = load_model_rows(
        tmp_path / "features.parquet", tmp_path / "clean/beds_weekly.parquet"
    )
    # y is a label; replacing all target columns must not alter engineered inputs.
    for column in [c for c in frame if c == "y" or c.startswith("y_")]:
        frame[column] = 999999
    frame.to_parquet(tmp_path / "changed.parquet")
    changed = load_model_rows(tmp_path / "changed.parquet", tmp_path / "clean/beds_weekly.parquet")
    pd.testing.assert_frame_equal(
        original[list(STAGE_NUMERIC_FEATURES)], changed[list(STAGE_NUMERIC_FEATURES)]
    )
    history = pd.read_parquet(tmp_path / "clean/beds_weekly.parquet")
    cutoff = pd.Timestamp(week(50))
    history.loc[pd.to_datetime(history.week) > cutoff, "beds_used"] = 1e9
    history.to_parquet(tmp_path / "future.parquet")
    future = load_model_rows(tmp_path / "features.parquet", tmp_path / "future.parquet")
    mask = original.origin_week <= cutoff
    pd.testing.assert_frame_equal(
        original.loc[mask, list(STAGE_NUMERIC_FEATURES)],
        future.loc[mask, list(STAGE_NUMERIC_FEATURES)],
    )
