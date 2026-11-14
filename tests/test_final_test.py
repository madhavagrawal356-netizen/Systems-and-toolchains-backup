import numpy as np
import pandas as pd

from smartops import ladder_models
from smartops.cli import build_parser
from smartops.features import STAGE_NUMERIC_FEATURES, load_model_rows
from smartops.tracking import experiment_run


def test_final_test_is_explicit_and_default_loader_excludes_it(tmp_path):
    from test_leakage import make_features

    frame = make_features(tmp_path)
    frame.loc[frame.index[-3:], "period"] = "test"
    frame.to_parquet(tmp_path / "features.parquet")
    default = load_model_rows(tmp_path / "features.parquet")
    final = load_model_rows(tmp_path / "features.parquet", include_test=True)
    assert "test" not in set(default.period)
    assert final.period.eq("test").sum() == 3
    assert build_parser().parse_args(["final-test"]).command == "final-test"


def test_final_scoring_does_not_fit_or_use_test_targets_for_predictions(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        {
            "hospital_pk": ["a", "a", "b", "b"],
            "origin_week": pd.to_datetime(["2023-06-25", "2023-07-02"] * 2),
            "target_week": pd.to_datetime(["2023-07-02", "2023-07-09"] * 2),
            "period": ["test"] * 4,
            "hospital_split": ["train", "train", "holdout", "holdout"],
            "y": [11.0, 12.0, 21.0, 22.0],
            "beds_lag1": [10.0, 11.0, 20.0, 21.0],
            "state": ["PA"] * 4,
        }
    )
    for column in STAGE_NUMERIC_FEATURES:
        frame[column] = 1.0

    def loader(path, *, include_test=False):
        assert include_test
        return frame.copy()

    class Frozen:
        best_iteration_ = 1

        def predict(self, x, **kwargs):
            return np.zeros(len(x))

        def transform(self, x):
            return x

        def fit(self, *args, **kwargs):
            raise AssertionError("Test scoring must never fit")

    monkeypatch.setattr(ladder_models, "load_model_rows", loader)
    monkeypatch.setattr(
        ladder_models,
        "_predict_mlp_quantiles",
        lambda model, prep, states, part: np.tile([-1.0, 0.0, 1.0], (len(part), 1)),
    )
    model = Frozen()
    selection = {"model_run_ids": {}, "groups": {}, "persistence_p90_change_error": 1.0}
    outputs = []
    for index in range(2):
        with experiment_run("test-scoring-fixture"):
            result = ladder_models._score_final_models(
                tmp_path / "unused.parquet",
                tmp_path / f"p{index}.parquet",
                tmp_path / f"m{index}.json",
                phi=0.0,
                linear=model,
                tree=model,
                tree_preprocessor=model,
                mlp=model,
                mlp_preprocessor=model,
                state_map={},
                selection=selection,
            )
        assert set(result["groups"]) == {"test/train_hospitals", "test/holdout_hospitals"}
        assert all(
            score["rows"] == 2 for group in result["groups"].values() for score in group.values()
        )
        outputs.append(pd.read_parquet(tmp_path / f"p{index}.parquet"))
        frame["y"] += 1000  # Labels change scores, never forecasts.
    prediction_columns = [
        "momentum",
        "linear",
        "lightgbm",
        "quantile_mlp",
        "quantile_mlp_p10",
        "quantile_mlp_p50",
        "quantile_mlp_p90",
    ]
    pd.testing.assert_frame_equal(outputs[0][prediction_columns], outputs[1][prediction_columns])
