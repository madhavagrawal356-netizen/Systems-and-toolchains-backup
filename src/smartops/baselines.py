"""The two trivial baselines from the proposal, scored as MAE in beds.

    persistence     y_hat = beds_lag1    ("next week looks like this week")
    seasonal naive  y_hat = beds_lag52   ("next week looks like the same week last year")

Both are scored on the same rows (target and both lags known), so their numbers compare
directly. Persistence is also scored on every row where it can predict, because that is the
number a model has to beat. These are point forecasts; quantile metrics (pinball loss,
coverage) come with the models.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from smartops.tracking import tracked_experiment

PERIODS = ("val", "test")
SPLITS = ("train", "holdout")


def _r(x: float | None) -> float | None:
    """Round a metric; an empty group (no rows) has no MAE."""
    return None if x is None else round(x, 3)


def _f(x: float | None) -> str:
    return f"{'-':>10}" if x is None else f"{x:>10.2f}"


@tracked_experiment("baselines")
def score_baselines(features_path: Path, out_path: Path) -> dict:
    con = duckdb.connect()
    results = {}
    for period in PERIODS:
        for split in SPLITS:
            where = f"period = '{period}' AND hospital_split = '{split}' AND y IS NOT NULL"
            n_p, mae_p = con.execute(
                f"""SELECT count(*), avg(abs(y - beds_lag1)) FROM '{features_path}'
                    WHERE {where} AND beds_lag1 IS NOT NULL"""
            ).fetchone()
            n_c, mae_pc, mae_sc = con.execute(
                f"""SELECT count(*), avg(abs(y - beds_lag1)), avg(abs(y - beds_lag52))
                    FROM '{features_path}'
                    WHERE {where} AND beds_lag1 IS NOT NULL AND beds_lag52 IS NOT NULL"""
            ).fetchone()
            results[f"{period}/{split}_hospitals"] = {
                "persistence_all_rows": {"n": int(n_p), "mae": _r(mae_p)},
                "common_rows": {
                    "n": int(n_c),
                    "persistence_mae": _r(mae_pc),
                    "seasonal_naive_mae": _r(mae_sc),
                },
            }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    print(
        f"{'period/hospitals':<24}{'n':>9}{'persist':>10}{'n common':>10}"
        f"{'persist':>10}{'seasonal':>10}   (MAE, beds)"
    )
    for key, r in results.items():
        a, c = r["persistence_all_rows"], r["common_rows"]
        print(
            f"{key:<24}{a['n']:>9,}{_f(a['mae'])}{c['n']:>10,}"
            f"{_f(c['persistence_mae'])}{_f(c['seasonal_naive_mae'])}"
        )
    print(f"wrote {out_path}")
    return results
