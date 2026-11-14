"""Command-line entry point: `uv run smartops <command>`.

Each pipeline stage registers a subcommand here as it is built, so the whole
project can be reproduced end to end without opening a notebook.

Data stage, in order:  download -> ingest -> clean -> features   (or `data` to run all four)
Then:                  baselines  (persistence and seasonal naive MAE on val and test)
"""

from __future__ import annotations

import argparse
import json

from smartops import __version__
from smartops.config import PATHS

RAW_PARQUET = "raw.parquet"
FEATURES_PARQUET = "features.parquet"


def _info(_: argparse.Namespace) -> None:
    print(f"smartops {__version__}")
    print(f"data dir: {PATHS.data}")


def _download(args: argparse.Namespace) -> None:
    from smartops.data.download import download

    download(PATHS.raw, force=args.force)


def _verify(_: argparse.Namespace) -> None:
    from smartops.data.download import FILENAME, verify

    print(f"checksum OK: {FILENAME} {verify(PATHS.raw)}")


def _ingest(_: argparse.Namespace) -> None:
    from smartops.data.download import FILENAME, verify
    from smartops.data.ingest import ingest

    verify(PATHS.raw)
    ingest(PATHS.raw / FILENAME, PATHS.interim / RAW_PARQUET)


def _clean(_: argparse.Namespace) -> None:
    from smartops.data.clean import clean
    from smartops.data.download import FILENAME, read_sums

    counts = clean(PATHS.interim / RAW_PARQUET, PATHS.clean, read_sums(PATHS.raw).get(FILENAME))
    for step in counts["waterfall"]:
        print(f"  {step['rows']:>12,}  {step['step']}")


def _features(_: argparse.Namespace) -> None:
    from smartops.features import build_features

    build_features(PATHS.clean / "beds_weekly.parquet", PATHS.processed / FEATURES_PARQUET)


def _baselines(_: argparse.Namespace) -> None:
    from smartops.baselines import score_baselines

    score_baselines(PATHS.processed / FEATURES_PARQUET, PATHS.processed / "baselines.json")


def _data(args: argparse.Namespace) -> None:
    _download(args)
    _ingest(args)
    _clean(args)
    _features(args)


def _ladder(args: argparse.Namespace) -> None:
    from smartops.ladder_models import run_ladder_models

    metrics = run_ladder_models(
        features_path=PATHS.processed / FEATURES_PARQUET,
        predictions_path=PATHS.processed / "ladder_predictions.parquet",
        metrics_path=PATHS.processed / "ladder_metrics.json",
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )
    print(json.dumps(metrics, indent=2))


def _audit(args: argparse.Namespace) -> None:
    from smartops.audit import run_audit

    print(
        json.dumps(
            run_audit(
                PATHS.processed, repair_tracking=args.repair_tracking, backfill=args.backfill
            ),
            indent=2,
        )
    )


def _final_test(args: argparse.Namespace) -> None:
    import mlflow
    import pandas as pd

    from smartops.audit import audit_mlflow, evaluate_saved
    from smartops.ladder_models import run_final_test
    from smartops.tracking import experiment_run, log_metrics_tree

    metrics = run_final_test(
        PATHS.processed / FEATURES_PARQUET,
        PATHS.processed / "ladder_test_predictions.parquet",
        PATHS.processed / "ladder_test_metrics.json",
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )
    output = PATHS.processed / "final_test"
    pd.json_normalize(
        [
            dict(group=group, model=model, **score)
            for group, models in metrics["groups"].items()
            for model, score in models.items()
        ]
    ).to_csv(output / "metrics.csv", index=False)
    with experiment_run("final-test-comparison", args.mlflow_tracking_uri):
        report = evaluate_saved(PATHS.processed, output, period="test")
        mlflow.set_tag("ladder_test_run_id", metrics["run_id"])
        log_metrics_tree(report["official"], "official")
        mlflow.log_artifact(str(output / "evaluation.json"))
        mlflow.log_artifact(str(output / "evaluation_rows.parquet"))
        mlflow.log_artifact(str(output / "metrics.csv"))
    audit_mlflow(PATHS.processed, PATHS.processed / "audit", args.mlflow_tracking_uri)
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartops", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="show version and paths").set_defaults(func=_info)

    p = sub.add_parser("download", help="download the raw HHS CSV and verify its checksum")
    p.add_argument("--force", action="store_true", help="download again even if present")
    p.set_defaults(func=_download)
    sub.add_parser("verify", help="check data/raw against SHA256SUMS").set_defaults(func=_verify)
    sub.add_parser("ingest", help="raw CSV -> data/interim/raw.parquet").set_defaults(func=_ingest)
    sub.add_parser("clean", help="interim -> data/clean/ tables + counts.json").set_defaults(
        func=_clean
    )
    sub.add_parser(
        "features", help="data/clean -> data/processed/features.parquet (shared model inputs)"
    ).set_defaults(func=_features)
    sub.add_parser(
        "baselines", help="persistence and seasonal naive MAE -> data/processed/baselines.json"
    ).set_defaults(func=_baselines)
    p = sub.add_parser("data", help="run download, ingest, clean and features in order")
    p.add_argument("--force", action="store_true", help="download again even if present")
    p.set_defaults(func=_data)
    p = sub.add_parser(
        "ladder",
        help="fit momentum, linear, ladder LightGBM and quantile MLP; score validation only",
    )
    p.add_argument("--mlflow-tracking-uri", default=None, help="optional MLflow tracking URI")
    p.set_defaults(func=_ladder)
    p = sub.add_parser("final-test", help="score validation-selected ladder models on final test")
    p.add_argument("--mlflow-tracking-uri", default=None, help="optional MLflow tracking URI")
    p.set_defaults(func=_final_test)
    p = sub.add_parser("audit", help="audit validation rows, temporal leakage and MLflow runs")
    p.add_argument(
        "--repair-tracking",
        action="store_true",
        help="correct runs explicitly tagged FAILED but marked FINISHED",
    )
    p.add_argument(
        "--backfill",
        action="store_true",
        help="import saved experiments as explicitly labelled historical artifacts",
    )
    p.set_defaults(func=_audit)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
