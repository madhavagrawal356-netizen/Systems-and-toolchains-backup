"""Exercise persistence through the real ladder using synthetic, validation-only data."""

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

from smartops import ladder_models
from smartops.features import STAGE_NUMERIC_FEATURES


def test_ladder_bundle_reproduces_validation_without_refitting(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    n_train, n_val, n_holdout = 320, 20, 20
    count = n_train + n_val + n_holdout
    frame = pd.DataFrame(
        rng.normal(size=(count, len(STAGE_NUMERIC_FEATURES))),
        columns=STAGE_NUMERIC_FEATURES,
    )
    frame["hospital_pk"] = [f"h{i}" for i in range(count)]
    frame["target_week"] = pd.Timestamp("2023-01-01")
    frame["period"] = ["train"] * n_train + ["val"] * (n_val + n_holdout)
    frame["hospital_split"] = ["train"] * (n_train + n_val) + ["holdout"] * n_holdout
    frame["state"] = ["PA", "NY"] * (count // 2)
    frame.loc[count - n_holdout :, "state"] = "UNSEEN"
    frame["beds_lag1"] = 20.0 + rng.random(count)
    frame["change"] = 2 * frame["momentum"] + rng.normal(size=count)
    frame["y"] = frame["beds_lag1"] + frame["change"]
    # Cover missing numeric inputs in both fit and inference, and an unseen state.
    frame.loc[[0, n_train, count - 1], "momentum"] = np.nan
    monkeypatch.setattr(ladder_models, "PROJECT_ROOT", tmp_path)
    features = tmp_path / "data" / "processed" / "features.parquet"
    features.parent.mkdir(parents=True)
    frame.to_parquet(features)
    monkeypatch.setattr(ladder_models, "_load_rows", lambda path: pd.read_parquet(path))
    original_fit = ladder_models._fit_mlp

    def short_fit(train, val):
        return original_fit(train, val, epochs=2, batch_size=64)

    monkeypatch.setattr(ladder_models, "_fit_mlp", short_fit)
    uri = f"sqlite:///{(tmp_path / 'runs.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment(
        "smartops-model-ladder", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    metrics = ladder_models.run_ladder_models(
        features, tmp_path / "predictions.parquet", tmp_path / "metrics.json", uri
    )
    bundle_path = tmp_path / "models"
    assert not (features.parent / "models").exists()
    bundles = list(bundle_path.iterdir())
    assert len(bundles) == 1
    saved_path = bundles[0]
    assert str(saved_path.resolve()) == metrics["models_path"]
    manifest = json.loads((saved_path / "manifest.json").read_text())
    assert manifest["model_run_ids"] == metrics["model_run_ids"]
    client = mlflow.MlflowClient(tracking_uri=uri)
    artifacts = client.list_artifacts(manifest["parent_run_id"], "models")
    assert {a.path for a in artifacts} == {
        "models/manifest.json",
        "models/preprocessors.pkl",
        "models/lightgbm.txt",
        "models/mlp_weights.pt",
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("Loading and predicting must not fit or start experiment runs")

    monkeypatch.setattr(ColumnTransformer, "fit_transform", forbidden)
    monkeypatch.setattr(LinearRegression, "fit", forbidden)
    monkeypatch.setattr(ladder_models.lgb.LGBMRegressor, "fit", forbidden)
    monkeypatch.setattr(ladder_models, "_fit_mlp", forbidden)
    monkeypatch.setattr(mlflow, "start_run", forbidden)
    models = ladder_models.load_ladder_models(saved_path)
    validation = frame.loc[frame.period.eq("val")]
    original = pd.read_parquet(tmp_path / "predictions.parquet")
    reloaded = models.predict(validation).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        original, reloaded[original.columns], check_exact=False, atol=1e-6, rtol=1e-6
    )
    assert "UNSEEN" not in models.state_map
    assert not models.mlp.training
    # Prediction does not require labels; index/order and missing-input behavior survive saving.
    unlabeled = models.predict(validation.drop(columns=["y", "change"]))
    assert unlabeled.index.equals(validation.index)
    pd.testing.assert_frame_equal(unlabeled, models.predict(validation).drop(columns="y"))
    assert models.predict(validation.iloc[:0]).empty
    with pytest.raises(ValueError, match="finite beds_lag1"):
        models.predict(validation.assign(beds_lag1=np.nan))

    manifest["format_version"] = 999
    (saved_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported"):
        ladder_models.load_ladder_models(saved_path)
    manifest["format_version"] = 1
    (saved_path / "manifest.json").write_text(json.dumps(manifest))
    with (saved_path / "mlp_weights.pt").open("ab") as stream:
        stream.write(b"damaged")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ladder_models.load_ladder_models(saved_path)
