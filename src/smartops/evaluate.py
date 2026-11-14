"""Shared validation scoring. Row identity, exclusions and baseline are explicit."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

KEYS = ["hospital_pk", "target_week"]
QUANTILES = (0.1, 0.5, 0.9)


def row_digest(frame: pd.DataFrame) -> str:
    # Sorting makes this fingerprint independent of file order, so comparisons
    # can prove they used the same hospital-week population.
    keys = frame[KEYS].copy()
    keys["hospital_pk"] = keys["hospital_pk"].astype(str)
    keys["target_week"] = pd.to_datetime(keys["target_week"]).dt.strftime("%Y-%m-%d")
    return hashlib.sha256(keys.sort_values(KEYS).to_csv(index=False).encode()).hexdigest()


def validate_keys(frame: pd.DataFrame) -> None:
    # Duplicate forecasts would overweight a hospital-week in every aggregate.
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Forecast keys must be non-null and unique (hospital_pk, target_week)")


def score_group(part, prediction, p10=None, p50=None, p90=None, persistence_error_p90=None) -> dict:
    """Score exactly the supplied rows; never silently drop a model's bad predictions."""
    actual = part["y"].to_numpy(dtype=float)
    baseline = part["beds_lag1"].to_numpy(dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != actual.shape:
        raise ValueError("Prediction shape does not match evaluation rows")
    if not all(np.isfinite(x).all() for x in (actual, baseline, prediction)):
        raise ValueError("Non-finite evaluation values; select common rows before scoring")
    result = {
        "rows": len(part),
        "mae": None,
        "persistence_mae": None,
        "skill_vs_persistence": None,
        "hospital_win_share": None,
    }
    if not len(part):
        return result
    # Pooled MAE weights hospital-weeks equally. Skill is the relative reduction
    # in that error against persistence on these exact same rows.
    error = np.abs(actual - prediction)
    baseline_error = np.abs(actual - baseline)
    mae, baseline_mae = float(error.mean()), float(baseline_error.mean())
    hospitals = pd.DataFrame(
        {
            "hospital": part["hospital_pk"].astype(str).to_numpy(),
            "model": error,
            "baseline": baseline_error,
        }
    )
    by_hospital = hospitals.groupby("hospital")[["model", "baseline"]].mean()
    # Win share gives each hospital one vote, complementing pooled bed error.
    result.update(
        mae=mae,
        persistence_mae=baseline_mae,
        skill_vs_persistence=1 - mae / baseline_mae if baseline_mae else None,
        hospital_win_share=float((by_hospital.model < by_hospital.baseline).mean()),
    )
    supplied = [x is not None for x in (p10, p50, p90)]
    if any(supplied) and not all(supplied):
        raise ValueError("Supply all three quantiles")
    if all(supplied):
        q = np.column_stack((p10, p50, p90)).astype(float)
        if q.shape != (len(part), 3) or not np.isfinite(q).all():
            raise ValueError("Invalid quantile predictions")
        if (np.diff(q, axis=1) < 0).any():
            raise ValueError("Quantile predictions cross")
        # Pinball loss scores individual quantiles. The p10-to-p90 interval is
        # an 80% interval; coverage measures calibration and width measures spread.
        residual = actual[:, None] - q
        # Treat p90 as planned bed capacity: excess demand is a shortfall,
        # excess planned capacity is idle. Magnitudes average over ALL rows,
        # including zeros, whereas shortfall rate counts how often demand exceeds p90.
        # These are demand proxies, not a nurse scheduling or cost simulation.
        shortfall = np.maximum(actual - q[:, 2], 0)
        result.update(
            pinball_loss={
                f"p{int(t * 100)}": float(
                    np.maximum(t * residual[:, i], (t - 1) * residual[:, i]).mean()
                )
                for i, t in enumerate(QUANTILES)
            },
            interval_80_coverage=float(((actual >= q[:, 0]) & (actual <= q[:, 2])).mean()),
            interval_80_width=float((q[:, 2] - q[:, 0]).mean()),
            p90_staffing_shortfall_rate=float((shortfall > 0).mean()),
            mean_p90_staffing_shortfall_beds=float(shortfall.mean()),
            mean_p90_idle_beds=float(np.maximum(q[:, 2] - actual, 0).mean()),
        )
        if persistence_error_p90 is not None:
            # The baseline safety margin is a training-residual quantile supplied
            # by the caller; never estimate it from the period being evaluated.
            upper = np.maximum(0, baseline + persistence_error_p90)
            result.update(
                persistence_p90_staffing_shortfall_rate=float((actual > upper).mean()),
                persistence_mean_p90_staffing_shortfall_beds=float(
                    np.maximum(actual - upper, 0).mean()
                ),
                persistence_mean_p90_idle_beds=float(np.maximum(upper - actual, 0).mean()),
            )
    return result


def score_models(
    frame: pd.DataFrame,
    models: dict[str, str],
    *,
    quantiles=None,
    persistence_error_p90=None,
    require_complete=False,
) -> tuple[dict, pd.DataFrame]:
    """One common finite mask for every compared model, including persistence.

    The caller aligns predictions by key first. Missing predictions are reported,
    never filled with a baseline or silently scored on a different population.
    """
    validate_keys(frame)
    quantiles = quantiles or {}
    columns = list(
        dict.fromkeys(
            ["y", "beds_lag1", *models.values(), *(c for cols in quantiles.values() for c in cols)]
        )
    )
    finite = np.isfinite(frame[columns].to_numpy(dtype=float))
    # Intersect validity across ALL outputs, including interval endpoints. In
    # strict official comparisons, a model cannot drop difficult eligible rows.
    common = finite.all(axis=1)
    eligible = np.isfinite(frame[["y", "beds_lag1"]].to_numpy(dtype=float)).all(axis=1)
    if require_complete and np.any(eligible & ~common):
        raise ValueError("Official models must predict every eligible validation row")
    selected = frame.loc[common].copy()
    if selected.empty:
        raise ValueError("No common evaluation rows")
    scores = {"persistence": score_group(selected, selected["beds_lag1"])}
    for name, column in models.items():
        qs = [selected[c].to_numpy() for c in quantiles.get(name, [])]
        scores[name] = score_group(
            selected, selected[column], *qs, persistence_error_p90=persistence_error_p90
        )
    return {
        "candidate_rows": len(frame),
        "eligible_rows": int(eligible.sum()),
        "common_rows": len(selected),
        "excluded_rows": int((~common).sum()),
        "row_sha256": row_digest(selected),
        "nonfinite_by_column": dict(zip(columns, (~finite).sum(axis=0).tolist(), strict=True)),
        "models": scores,
    }, selected


def evaluate_checks(train_scores: dict, holdout_scores: dict) -> dict:
    """Proposal thresholds, applied to the same shared-harness metrics everywhere."""
    checks = {}
    for name, score in train_scores.items():
        # Here "train" means training hospitals in the evaluation period,
        # not in-sample training scores. Holdout scores test unseen facilities.
        skill = score.get("skill_vs_persistence")
        unseen_skill = holdout_scores.get(name, {}).get("skill_vs_persistence")
        wins = score.get("hospital_win_share")
        checks[name] = {
            "point_skill_at_least_3_percent": skill is not None and skill >= 0.03,
            "positive_skill_on_unseen_hospitals": (
                unseen_skill > 0 if unseen_skill is not None else None
            ),
            "beats_persistence_for_over_half_train_hospitals": wins is not None and wins > 0.5,
        }
        if "interval_80_coverage" in score:
            # Keep the proposal's fixed tolerance; p10/p90 cannot assess a 90% band.
            checks[name]["80_percent_interval_calibrated"] = (
                0.77 <= score["interval_80_coverage"] <= 0.83
            )
            checks[name]["90_percent_interval_calibrated"] = None  # p05/p95 not available
        if "persistence_p90_staffing_shortfall_rate" in score:
            # The proposal requires BOTH fewer shortfall events and no extra idle
            # beds; a lower mean shortfall alone does not satisfy this check.
            checks[name]["useful_for_staffing"] = (
                score["p90_staffing_shortfall_rate"]
                < score["persistence_p90_staffing_shortfall_rate"]
                and score["mean_p90_idle_beds"] <= score["persistence_mean_p90_idle_beds"]
            )
    return checks
