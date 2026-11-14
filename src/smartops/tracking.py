"""MLflow lifecycle shared by every model entry point, including failed attempts."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import traceback
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import mlflow

from smartops.config import PATHS, RANDOM_SEED


def default_tracking_uri() -> str:
    PATHS.processed.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(PATHS.processed / 'mlflow.db').resolve().as_posix()}"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_metrics_tree(values: dict, prefix: str = "") -> None:
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            log_metrics_tree(value, name)
        elif isinstance(value, (int, float)) and value is not None:
            mlflow.log_metric(name, value)


@contextmanager
def experiment_run(name: str, tracking_uri=None, params=None):
    if mlflow.active_run() is None:
        mlflow.set_tracking_uri(tracking_uri or default_tracking_uri())
        mlflow.set_experiment("smartops-model-ladder")
    with mlflow.start_run(run_name=name, nested=mlflow.active_run() is not None):
        mlflow.log_params(params or {})
        mlflow.set_tag("tracking_schema", "2")
        try:
            yield
        except BaseException as error:
            mlflow.set_tag("error_type", type(error).__name__)
            mlflow.set_tag("error_message", str(error)[:1000])
            mlflow.log_text(traceback.format_exc(), "failure.txt")
            # Reraising inside start_run gives the run a real FAILED lifecycle status.
            raise


def tracked_experiment(name: str):
    """Start tracking before input validation or fitting, and propagate failures."""

    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            bound = inspect.signature(function).bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            params = {
                k: str(v)
                for k, v in values.items()
                if not k.endswith("_path") and k != "mlflow_tracking_uri"
            }
            params["seed"] = params.get("random_state", RANDOM_SEED)
            with experiment_run(name, values.get("mlflow_tracking_uri"), params):
                source = Path(inspect.getfile(function))
                mlflow.log_param("source_sha256", file_digest(source))
                mlflow.log_artifact(str(source), "source")
                for filename in ("features.py", "evaluate.py", "tracking.py", "config.py"):
                    shared = Path(__file__).parent / filename
                    if shared != source:
                        mlflow.log_artifact(str(shared), "source")
                environment = {}
                for package in (
                    "numpy",
                    "pandas",
                    "torch",
                    "scikit-learn",
                    "lightgbm",
                    "mlflow",
                    "pyarrow",
                    "duckdb",
                ):
                    try:
                        environment[package] = version(package)
                    except PackageNotFoundError:
                        environment[package] = "not installed"
                mlflow.log_dict(environment, "environment.json")
                features = values.get("features_path")
                if features is not None and Path(features).exists():
                    mlflow.log_param("features_sha256", file_digest(Path(features)))
                result = function(*args, **kwargs)
                for key in ("predictions_path", "metrics_path", "model_path", "out_path"):
                    path = values.get(key)
                    if path is not None and Path(path).exists():
                        mlflow.log_artifact(str(path))
                        if key in ("metrics_path", "out_path"):
                            log_metrics_tree(json.loads(Path(path).read_text(encoding="utf-8")))
                return result

        return wrapped

    return decorate
