"""Keep test experiment records separate from the project's real MLflow database."""

import mlflow
import pytest


@pytest.fixture(autouse=True)
def isolated_default_tracking(tmp_path, monkeypatch):
    from smartops import tracking

    uri = f"sqlite:///{(tmp_path / 'test-mlflow.db').as_posix()}"
    monkeypatch.setattr(tracking, "default_tracking_uri", lambda: uri)
    previous_uri = mlflow.get_tracking_uri()
    yield
    if mlflow.active_run() is not None:
        mlflow.end_run(status="KILLED")
    mlflow.set_tracking_uri(previous_uri)
