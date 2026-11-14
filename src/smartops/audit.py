"""Reproducible validation comparisons, temporal checks and historical run inventory."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from smartops.config import TEST_START, TRAIN_START, VAL_START
from smartops.evaluate import KEYS, evaluate_checks, score_models, validate_keys
from smartops.features import FEATURES
from smartops.tracking import default_tracking_uri, experiment_run, file_digest, log_metrics_tree

OFFICIAL = ("momentum", "linear", "lightgbm", "quantile_mlp")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def align_predictions(rows: pd.DataFrame, predictions: pd.DataFrame, columns) -> pd.DataFrame:
    """Align by keys and verify saved truth/metadata against the authoritative table."""
    rows, predictions = rows.copy(), predictions.copy()
    for frame in (rows, predictions):
        frame["hospital_pk"] = frame["hospital_pk"].astype(str)
        for column in ("target_week", "origin_week"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column])
    validate_keys(rows)
    validate_keys(predictions)
    check = rows.merge(predictions, on=KEYS, suffixes=("", "_saved"), validate="one_to_one")
    for source, saved in (
        ("y", "y_saved"),
        ("y", "actual"),
        ("beds_lag1", "persistence"),
        ("beds_lag1", "beds_lag1_saved"),
    ):
        if saved in check and not np.allclose(check[source], check[saved], equal_nan=True):
            raise ValueError(f"Saved {saved} differs from authoritative {source}")
    for column in ("period", "hospital_split", "origin_week"):
        saved = f"{column}_saved"
        if saved in check and not check[column].eq(check[saved]).all():
            raise ValueError(f"Saved {column} differs from authoritative metadata")
    return rows.merge(predictions[[*KEYS, *columns]], on=KEYS, how="left", validate="one_to_one")


def evaluate_saved(processed: Path, output: Path, *, period: str = "val") -> dict:
    if period not in {"val", "test"}:
        raise ValueError("period must be val or test")
    rows = pd.read_parquet(processed / "features.parquet", filters=[("period", "=", period)])
    train = pd.read_parquet(
        processed / "features.parquet",
        filters=[("period", "=", "train"), ("hospital_split", "=", "train")],
        columns=["y", "beds_lag1"],
    )
    error_p90 = float((train.y - train.beds_lag1).replace([np.inf, -np.inf], np.nan).quantile(0.9))
    ladder_name = (
        "ladder_test_predictions.parquet" if period == "test" else "ladder_predictions.parquet"
    )
    ladder = pd.read_parquet(processed / ladder_name)
    qs = [f"quantile_mlp_p{q}" for q in (10, 50, 90)]
    aligned = align_predictions(rows, ladder, [*OFFICIAL, *qs])
    report = {
        "protocol": f"{period} only; key-aligned shared finite rows",
        "features_sha256": file_digest(processed / "features.parquet"),
        "test_status": (
            "Earlier experiments already scored test; "
            + (
                "official test scored on user request."
                if period == "test"
                else "no new test scoring in this audit."
            )
        ),
        "interval_90": "not available: p05 and p95 were not produced",
        "official": {},
    }
    manifests = []
    for split in ("train", "holdout"):
        part = aligned.loc[aligned.hospital_split.eq(split)]
        result, selected = score_models(
            part,
            {name: name for name in OFFICIAL},
            quantiles={"quantile_mlp": qs},
            persistence_error_p90=error_p90,
            require_complete=True,
        )
        report["official"][split] = result
        manifests.append(selected.assign(comparison=f"official/{split}"))
    report["checks"] = evaluate_checks(
        report["official"]["train"]["models"], report["official"]["holdout"]["models"]
    )
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(manifests, ignore_index=True).to_parquet(
        output / "evaluation_rows.parquet", index=False
    )
    write_json(output / "evaluation.json", report)
    return report


def audit_features(features_path: Path, clean_path: Path, raw_path: Path | None = None) -> dict:
    frame = pd.read_parquet(features_path)
    validate_keys(frame)
    frame["origin_week"] = pd.to_datetime(frame.origin_week)
    frame["target_week"] = pd.to_datetime(frame.target_week)
    expected_period = np.where(
        frame.target_week < pd.Timestamp(VAL_START),
        "train",
        np.where(frame.target_week < pd.Timestamp(TEST_START), "val", "test"),
    )
    checks = {
        "horizon_mismatches": int(((frame.target_week - frame.origin_week).dt.days != 7).sum()),
        "period_mismatches": int((frame.period != expected_period).sum()),
        "pre_training_targets": int((frame.target_week < pd.Timestamp(TRAIN_START)).sum()),
        "hospital_split_changes": int(
            (frame.groupby("hospital_pk").hospital_split.nunique() != 1).sum()  # noqa: PD101
        ),
        "invalid_hospital_splits": int((~frame.hospital_split.isin(["train", "holdout"])).sum()),
    }
    history = pd.read_parquet(clean_path)
    history["week"] = pd.to_datetime(history.week)
    history = history.set_index(["hospital_pk", "week"])
    if not history.index.is_unique:
        raise ValueError("Duplicate hospital/week in clean history")
    mapping = {f"beds_lag{k}": ("beds_used", k) for k in (1, 2, 3, 4, 52)}
    mapping.update(
        capacity_lag1=("capacity", 1),
        covid_lag1=("covid_beds", 1),
        flu_lag1=("flu_patients", 1),
        y=("beds_used", 0),
    )
    for column, (source, lag) in mapping.items():
        keys = pd.MultiIndex.from_arrays(
            [frame.hospital_pk, frame.target_week - pd.Timedelta(weeks=lag)]
        )
        expected = history[source].reindex(keys).to_numpy(dtype=float)
        checks[f"{column}_mismatches"] = int(
            (~np.isclose(frame[column].to_numpy(dtype=float), expected, equal_nan=True)).sum()
        )
    checks["calendar_mismatches"] = int(
        (frame.week_of_year != frame.target_week.dt.isocalendar().week).sum()
    )
    if raw_path is not None and raw_path.exists():
        # Compare static-looking attributes with the latest non-null report available at origin.
        with duckdb.connect() as con:
            con.register("forecast_rows", frame)
            attrs = con.execute(
                """
                WITH history AS (
                    SELECT hospital_pk, collection_week,
                        last_value(state IGNORE NULLS) OVER w AS known_state,
                        last_value(hospital_subtype IGNORE NULLS) OVER w AS known_subtype
                    FROM read_parquet(?)
                    WINDOW w AS (PARTITION BY hospital_pk ORDER BY collection_week
                                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                )
                SELECT f.state, f.hospital_subtype, h.known_state, h.known_subtype
                FROM forecast_rows f ASOF LEFT JOIN history h
                ON f.hospital_pk = h.hospital_pk AND f.origin_week >= h.collection_week
            """,
                [str(raw_path)],
            ).df()
        for source, known in (("state", "known_state"), ("hospital_subtype", "known_subtype")):
            checks[f"{source}_asof_mismatches"] = int(
                (attrs[source].fillna("__NULL__") != attrs[known].fillna("__NULL__")).sum()
            )
    return {
        "passed": all(value == 0 for value in checks.values()),
        "rows": len(frame),
        "features_sha256": file_digest(features_path),
        "checks": checks,
        "approved_inputs": list(FEATURES),
        "fit_scope": "train period AND train hospitals; validation only for selection",
        "rolling_history": (
            "Observed earlier validation weeks are allowed for later one-week forecasts."
        ),
        "limitations": [
            "Source publication/revision timestamps are unavailable; "
            "real-time availability cannot be certified.",
            "Hospital holdout membership is fixed from the full retrospective hospital roster.",
            *(
                []
                if raw_path is not None and raw_path.exists()
                else [
                    "Raw reports unavailable: origin-time categorical attributes were not verified."
                ]
            ),
        ],
    }


def audit_mlflow(
    processed: Path, output: Path, tracking_uri=None, repair=False, backfill=False
) -> dict:
    uri = tracking_uri or default_tracking_uri()
    client = MlflowClient(tracking_uri=uri)
    experiments = client.search_experiments()
    runs = [r for exp in experiments for r in client.search_runs([exp.experiment_id])]
    repaired = []
    for run in runs:
        if repair and run.data.tags.get("status") == "FAILED" and run.info.status != "FAILED":
            client.set_tag(run.info.run_id, "audit_previous_status", run.info.status)
            client.set_tag(run.info.run_id, "audit_repair", "explicit historical FAILED tag")
            client.set_terminated(run.info.run_id, status="FAILED", end_time=run.info.end_time)
            repaired.append(run.info.run_id)
    imported = []
    if backfill:
        for stem in ("ladder",):
            metrics_path = processed / f"{stem}_metrics.json"
            predictions_path = processed / f"{stem}_predictions.parquet"
            if not metrics_path.exists() or not predictions_path.exists():
                continue
            signature = file_digest(metrics_path) + ":" + file_digest(predictions_path)
            if any(r.data.tags.get("legacy_signature") == signature for r in runs):
                continue
            with experiment_run(f"legacy-import-{stem}", uri):
                mlflow.set_tags(
                    {
                        "run_kind": "legacy_artifact_import",
                        "legacy_signature": signature,
                        "training_reproduced": "false",
                        "original_run_time": "unknown",
                        "provenance": "saved artifacts only; not a new training run",
                    }
                )
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                log_metrics_tree(payload)
                mlflow.log_artifact(str(metrics_path))
                mlflow.log_artifact(str(predictions_path))
                imported.append(mlflow.active_run().info.run_id)
    experiments = client.search_experiments()
    runs = [r for exp in experiments for r in client.search_runs([exp.experiment_id])]
    inventory = []
    for run in runs:
        artifacts = client.list_artifacts(run.info.run_id)
        inventory.append(
            {
                "run_id": run.info.run_id,
                "name": run.data.tags.get("mlflow.runName"),
                "status": run.info.status,
                "tags": run.data.tags,
                "params": run.data.params,
                "metric_count": len(run.data.metrics),
                "artifacts": [a.path for a in artifacts],
            }
        )
    report = {
        "tracking_uri": uri,
        "run_count": len(runs),
        "repaired_runs": repaired,
        "imported_runs": imported,
        "previously_repaired_runs": [r.info.run_id for r in runs if "audit_repair" in r.data.tags],
        "legacy_import_runs": [
            r.info.run_id for r in runs if r.data.tags.get("run_kind") == "legacy_artifact_import"
        ],
        "runs": inventory,
        "remaining_status_mismatches": [
            r.info.run_id
            for r in runs
            if r.data.tags.get("status") == "FAILED" and r.info.status != "FAILED"
        ],
        "historical_limit": (
            "Unlogged attempts with no surviving artifacts or error records cannot be "
            "reconstructed; imported runs do not certify original training provenance."
        ),
    }
    write_json(output / "mlflow_audit.json", report)
    return report


def run_audit(processed: Path, *, repair_tracking=False, backfill=False) -> dict:
    output = processed / "audit"
    with experiment_run("validation-audit"):
        leakage = audit_features(
            processed / "features.parquet",
            processed.parent / "clean" / "beds_weekly.parquet",
            processed.parent / "interim" / "raw.parquet",
        )
        write_json(output / "leakage_audit.json", leakage)
        evaluation = evaluate_saved(processed, output)
        mlflow.log_artifact(str(output / "leakage_audit.json"))
        mlflow.log_artifact(str(output / "evaluation.json"))
        mlflow.log_artifact(str(output / "evaluation_rows.parquet"))
        mlflow.log_param("features_sha256", evaluation["features_sha256"])
        mlflow.log_param("official_row_sha256", evaluation["official"]["train"]["row_sha256"])
        log_metrics_tree(evaluation["official"], "official")
    tracking = audit_mlflow(processed, output, repair=repair_tracking, backfill=backfill)
    return {
        "leakage_passed": leakage["passed"],
        "official_rows": {
            key: value["common_rows"] for key, value in evaluation["official"].items()
        },
        "mlflow_runs": tracking["run_count"],
        "output": str(output),
    }
