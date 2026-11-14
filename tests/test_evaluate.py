import numpy as np
import pandas as pd
import pytest

from smartops.audit import OFFICIAL, align_predictions, evaluate_saved
from smartops.evaluate import evaluate_checks, row_digest, score_group, score_models


def rows():
    return pd.DataFrame(
        {
            "hospital_pk": ["a", "b", "c"],
            "target_week": pd.to_datetime(["2023-01-01"] * 3),
            "y": [10.0, 20.0, 30.0],
            "beds_lag1": [8.0, 18.0, 28.0],
            "one": [9.0, 19.0, 29.0],
            "two": [10.0, np.nan, 30.0],
        }
    )


def test_comparison_uses_one_mask_and_reports_exclusions():
    report, selected = score_models(rows(), {"one": "one", "two": "two"})
    assert report["common_rows"] == 2
    assert report["excluded_rows"] == 1
    assert {v["rows"] for v in report["models"].values()} == {2}
    assert report["models"]["one"]["mae"] == 1
    assert report["models"]["one"]["skill_vs_persistence"] == 0.5
    assert report["row_sha256"] == row_digest(selected.iloc[::-1])
    with pytest.raises(ValueError, match="every eligible"):
        score_models(rows(), {"one": "one", "two": "two"}, require_complete=True)


def test_alignment_checks_truth_and_keys_not_row_position():
    source = rows().drop(columns=["one", "two"])
    prediction = rows()[["hospital_pk", "target_week", "one"]].iloc[::-1]
    # Existing parquet artifacts mix Arrow dates and Pandas timestamps.
    source["target_week"] = source.target_week.dt.date
    assert align_predictions(source, prediction, ["one"])["one"].tolist() == [9, 19, 29]
    prediction["actual"] = 999
    with pytest.raises(ValueError, match="authoritative"):
        align_predictions(source, prediction, ["one"])
    with pytest.raises(ValueError, match="unique"):
        score_models(pd.concat([rows(), rows()]), {"one": "one"})


def test_strict_scoring_and_quantile_math():
    frame = rows()
    with pytest.raises(ValueError, match="Non-finite"):
        score_group(frame, frame.two)
    result = score_group(frame, frame.y, frame.y - 1, frame.y, frame.y + 1, 1.0)
    assert result["interval_80_coverage"] == 1
    assert result["interval_80_width"] == 2
    assert result["pinball_loss"]["p50"] == 0
    assert result["p90_staffing_shortfall_rate"] == 0
    assert result["persistence_p90_staffing_shortfall_rate"] == 1
    with pytest.raises(ValueError, match="cross"):
        score_group(frame, frame.y, frame.y + 1, frame.y, frame.y - 1)


def test_shared_proposal_checks_do_not_claim_unavailable_intervals():
    score = score_group(rows(), rows().y, rows().y - 1, rows().y, rows().y + 1, 1.0)
    checks = evaluate_checks({"model": score}, {"model": score})["model"]
    assert checks["point_skill_at_least_3_percent"]
    assert checks["positive_skill_on_unseen_hospitals"]
    assert checks["80_percent_interval_calibrated"] is False  # 100% is overcoverage
    assert checks["90_percent_interval_calibrated"] is None
    assert not checks["useful_for_staffing"]  # fewer shortfalls but more idle beds


@pytest.mark.parametrize("period", ["val", "test"])
def test_saved_comparison_requires_only_retained_model_files(tmp_path, period):
    frames = []
    for split_period, date in (
        ("train", "2022-12-25"),
        ("val", "2023-01-01"),
        ("test", "2023-07-02"),
    ):
        part = rows().drop(columns=["one", "two"])
        part["target_week"] = pd.Timestamp(date)
        part["origin_week"] = part.target_week - pd.Timedelta(days=7)
        part["period"] = split_period
        part["hospital_split"] = ["train", "train", "holdout"]
        frames.append(part)
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(tmp_path / "features.parquet")
    selected = frame.loc[frame.period.eq(period)].copy()
    for model in OFFICIAL:
        selected[model] = selected.y
    for quantile, offset in ((10, -1), (50, 0), (90, 1)):
        selected[f"quantile_mlp_p{quantile}"] = selected.y + offset
    filename = (
        "ladder_predictions.parquet" if period == "val" else "ladder_test_predictions.parquet"
    )
    selected.to_parquet(tmp_path / filename)
    report = evaluate_saved(tmp_path, tmp_path / "report", period=period)
    assert report["official"]["train"]["common_rows"] == 2
    assert report["official"]["holdout"]["common_rows"] == 1
    for group in report["official"].values():
        assert set(group["models"]) == {"persistence", *OFFICIAL}
        assert group["models"]["lightgbm"]["mae"] == 0
    manifest = pd.read_parquet(tmp_path / "report/evaluation_rows.parquet")
    assert set(manifest.comparison) == {"official/train", "official/holdout"}
