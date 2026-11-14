import json

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from smartops.audit import audit_mlflow
from smartops.tracking import experiment_run, tracked_experiment


def test_failure_has_real_failed_status_and_traceback(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'runs.db').as_posix()}"

    @tracked_experiment("intentional-test-failure")
    def fail(features_path, mlflow_tracking_uri=None):
        raise RuntimeError("failure before any model was fitted")

    with pytest.raises(RuntimeError, match="before any model"):
        fail(tmp_path / "missing.parquet", mlflow_tracking_uri=uri)
    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("smartops-model-ladder")
    run = client.search_runs([exp.experiment_id])[0]
    assert run.info.status == "FAILED"
    assert run.data.tags["error_type"] == "RuntimeError"
    assert "failure.txt" in {a.path for a in client.list_artifacts(run.info.run_id)}


def test_repair_only_explicit_failed_tag_and_is_idempotent(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'runs.db').as_posix()}"
    with experiment_run("historical-failure", uri):
        mlflow.set_tag("status", "FAILED")
    with experiment_run("successful", uri):
        pass
    report = audit_mlflow(tmp_path, tmp_path, uri, repair=True)
    assert len(report["repaired_runs"]) == 1
    assert not report["remaining_status_mismatches"]
    assert len([r for r in report["runs"] if r["status"] == "FINISHED"]) == 1
    assert not audit_mlflow(tmp_path, tmp_path, uri, repair=True)["repaired_runs"]


def test_legacy_import_keeps_provenance_and_is_idempotent(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'runs.db').as_posix()}"
    (tmp_path / "ladder_metrics.json").write_text(json.dumps({"mae": 2.5}))
    pd.DataFrame({"prediction": [1.0]}).to_parquet(tmp_path / "ladder_predictions.parquet")
    first = audit_mlflow(tmp_path, tmp_path, uri, backfill=True)
    second = audit_mlflow(tmp_path, tmp_path, uri, backfill=True)
    assert len(first["imported_runs"]) == 1
    assert not second["imported_runs"]
    imported = second["runs"][0]
    assert imported["tags"]["training_reproduced"] == "false"
    assert imported["tags"]["original_run_time"] == "unknown"
