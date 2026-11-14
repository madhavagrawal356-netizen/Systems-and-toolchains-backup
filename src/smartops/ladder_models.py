"""Stage 1 and 2 models for the escalation ladder.

All fitting and model selection use training and validation periods only. Test
rows are opened only by the explicit final-test path, after model selection.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from smartops.config import PROJECT_ROOT, RANDOM_SEED
from smartops.evaluate import evaluate_checks, score_models
from smartops.features import STAGE_NUMERIC_FEATURES, load_model_rows
from smartops.tracking import experiment_run, file_digest, log_metrics_tree, tracked_experiment

QUANTILES = (0.10, 0.50, 0.90)
_load_rows = load_model_rows


def _linear_features(df: pd.DataFrame) -> pd.DataFrame:
    # Stage 1 adds momentum to the shared lagged inputs. Missing values are
    # imputed from train rows only by the estimator pipeline.
    columns = [*STAGE_NUMERIC_FEATURES, "state"]
    return df[columns].copy()


class QuantileMLP(nn.Module):
    """Predict occupancy-change quantiles from numeric inputs and a learned state code."""

    def __init__(self, n_numeric: int, n_states: int, embedding_size: int = 8):
        super().__init__()
        self.state_embedding = nn.Embedding(n_states, embedding_size)
        self.network = nn.Sequential(
            nn.Linear(n_numeric + embedding_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, numeric, state):
        raw = self.network(torch.cat((numeric, self.state_embedding(state)), dim=1))
        # Positive softplus gaps anchor p10/p90 around p50, preventing crossing
        # without sorting predictions or changing the pinball training objective.
        median = raw[:, 1]
        lower = median - torch.nn.functional.softplus(raw[:, 0])
        upper = median + torch.nn.functional.softplus(raw[:, 2])
        return torch.stack((lower, median, upper), dim=1)


def _predict_mlp_quantiles(model, preprocessor, state_map, frame):
    # Reuse training statistics and reserve code 0 for states absent during fitting.
    numeric = list(STAGE_NUMERIC_FEATURES)
    matrix = preprocessor.transform(frame[numeric])
    states = (
        frame["state"]
        .fillna("__UNKNOWN__")
        .astype(str)
        .map(state_map)
        .fillna(0)
        .to_numpy(dtype=np.int64)
    )
    model.eval()
    with torch.no_grad():
        return model(
            torch.as_tensor(np.asarray(matrix), dtype=torch.float32),
            torch.as_tensor(states.copy(), dtype=torch.long),
        ).numpy()


def _pinball_loss(prediction, target):
    # Higher quantiles penalize underprediction more; average over rows and quantiles.
    quantiles = prediction.new_tensor(QUANTILES).view(1, -1)
    error = target.unsqueeze(1) - prediction
    return torch.maximum(quantiles * error, (quantiles - 1) * error).mean()


def _fit_mlp(train, val, epochs=60, batch_size=1024, learning_rate=1e-3):
    train = train.loc[train["change"].notna() & train["beds_lag1"].notna()].copy()
    val = val.loc[val["change"].notna() & val["beds_lag1"].notna()].copy()
    numeric = list(STAGE_NUMERIC_FEATURES)
    # Fit imputation, scaling and the state vocabulary on training rows only.
    prep = ColumnTransformer(
        [("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric)],
        remainder="drop",
    )
    train_x = prep.fit_transform(train)
    val_x = prep.transform(val)
    state_names = sorted(train["state"].fillna("__UNKNOWN__").astype(str).unique())
    state_map = {value: idx + 1 for idx, value in enumerate(state_names)}

    def state_ids(frame):
        return (
            frame["state"]
            .fillna("__UNKNOWN__")
            .astype(str)
            .map(state_map)
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )

    train_state = state_ids(train)
    val_state = state_ids(val)
    train_ds = TensorDataset(
        torch.as_tensor(np.asarray(train_x), dtype=torch.float32),
        torch.as_tensor(train_state.copy(), dtype=torch.long),
        torch.as_tensor(train["change"].to_numpy(dtype=np.float32)),
    )
    val_numeric = torch.as_tensor(np.asarray(val_x), dtype=torch.float32)
    val_states = torch.as_tensor(val_state.copy(), dtype=torch.long)
    val_target = torch.as_tensor(val["change"].to_numpy(dtype=np.float32))
    torch.manual_seed(RANDOM_SEED)
    model = QuantileMLP(train_x.shape[1], len(state_names) + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    best_state, best_loss, stale = None, float("inf"), 0
    # Only training batches update weights. Validation selects the checkpoint;
    # detached copies prevent later optimizer steps from overwriting the best one.
    for _ in range(epochs):
        model.train()
        for x, state, y in loader:
            loss = _pinball_loss(model(x, state), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(_pinball_loss(model(val_numeric, val_states), val_target))
        if val_loss < best_loss:
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_loss, stale = val_loss, 0
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        quantiles = model(val_numeric, val_states).numpy()
    return model, prep, state_map, quantiles, best_loss


@dataclass
class FittedLadderModels:
    """Frozen fitted objects; predict accepts engineered rows from load_model_rows()."""

    phi: float
    linear: object
    tree: lgb.Booster
    tree_preprocessor: ColumnTransformer
    mlp: QuantileMLP
    mlp_preprocessor: ColumnTransformer
    state_map: dict[str, int]
    metadata: dict

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return occupancy predictions in input order without fitting or reading labels.

        The caller selects the period and eligible rows. A finite beds_lag1 is
        required; missing engineered numeric values use the saved training imputer.
        """
        base = frame["beds_lag1"].to_numpy(dtype=float)
        if not np.isfinite(base).all():
            raise ValueError("Prediction requires finite beds_lag1; select eligible rows first")
        columns = [
            c
            for c in (
                "hospital_pk",
                "origin_week",
                "target_week",
                "period",
                "hospital_split",
                "y",
                "beds_lag1",
            )
            if c in frame
        ]
        out = frame[columns].copy()
        names = (
            "momentum",
            "linear",
            "lightgbm",
            "quantile_mlp",
            "quantile_mlp_p10",
            "quantile_mlp_p50",
            "quantile_mlp_p90",
        )
        if frame.empty:
            return out.assign(**{name: np.empty(0, dtype=float) for name in names})
        # Every estimator predicts a change; restore bed levels with the same
        # nonnegative clipping used by the official scoring path.
        inputs = _linear_features(frame)
        out["momentum"] = np.maximum(0, base + self.phi * frame.momentum.fillna(0).to_numpy())
        out["linear"] = np.maximum(0, base + self.linear.predict(inputs))
        out["lightgbm"] = np.maximum(
            0, base + self.tree.predict(self.tree_preprocessor.transform(inputs))
        )
        change_q = _predict_mlp_quantiles(self.mlp, self.mlp_preprocessor, self.state_map, frame)
        for i, quantile in enumerate((10, 50, 90)):
            out[f"quantile_mlp_p{quantile}"] = np.maximum(0, base + change_q[:, i])
        out["quantile_mlp"] = out["quantile_mlp_p50"]
        return out


def _save_ladder_models(
    path, *, phi, linear, tree, tree_preprocessor, mlp, mlp_preprocessor, state_map, metadata
) -> None:
    """Save one run's selected weights and fitted transformations before scoring."""
    path = Path(path)
    # A run-specific directory avoids replacing earlier evaluated model evidence.
    path.mkdir(parents=True, exist_ok=False)
    with (path / "preprocessors.pkl").open("wb") as stream:
        pickle.dump({"linear": linear, "tree": tree_preprocessor, "mlp": mlp_preprocessor}, stream)
    tree.booster_.save_model(str(path / "lightgbm.txt"), num_iteration=tree.best_iteration_)
    torch.save(mlp.state_dict(), path / "mlp_weights.pt")
    manifest = {
        **metadata,
        "format_version": 1,
        "numeric_features": list(STAGE_NUMERIC_FEATURES),
        "quantiles": list(QUANTILES),
        "momentum_phi": phi,
        "state_map": state_map,
        "mlp": {
            "n_numeric": mlp.network[0].in_features - mlp.state_embedding.embedding_dim,
            "n_states": mlp.state_embedding.num_embeddings,
            "embedding_size": mlp.state_embedding.embedding_dim,
        },
        "packages": {name: version(name) for name in ("scikit-learn", "torch", "lightgbm")},
        "sha256": {
            name: file_digest(path / name)
            for name in ("preprocessors.pkl", "lightgbm.txt", "mlp_weights.pt")
        },
    }
    # Write the manifest last: an interrupted save is not a loadable bundle.
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_ladder_models(models_path: Path) -> FittedLadderModels:
    """Load a trusted run bundle without fitting or starting an MLflow run.

    Use the training environment (uv.lock). The sklearn pipelines use pickle;
    load only bundles from a trusted source. Hashes detect damage, not authenticity.
    """
    path = Path(models_path)
    metadata = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if metadata["format_version"] != 1:
        raise ValueError("Unsupported ladder model bundle version")
    if metadata["numeric_features"] != list(STAGE_NUMERIC_FEATURES):
        raise ValueError("Saved numeric feature schema differs from the current ladder")
    if metadata["quantiles"] != list(QUANTILES):
        raise ValueError("Saved quantiles differ from the current ladder")
    for name in ("preprocessors.pkl", "lightgbm.txt", "mlp_weights.pt"):
        if file_digest(path / name) != metadata["sha256"][name]:
            raise ValueError(f"Model bundle checksum mismatch: {name}")
    with (path / "preprocessors.pkl").open("rb") as stream:
        fitted = pickle.load(stream)
    mlp = QuantileMLP(**metadata["mlp"])
    mlp.load_state_dict(torch.load(path / "mlp_weights.pt", map_location="cpu", weights_only=True))
    mlp.eval()
    return FittedLadderModels(
        phi=metadata["momentum_phi"],
        linear=fitted["linear"],
        tree=lgb.Booster(model_file=str(path / "lightgbm.txt")),
        tree_preprocessor=fitted["tree"],
        mlp=mlp,
        mlp_preprocessor=fitted["mlp"],
        state_map=metadata["state_map"],
        metadata=metadata,
    )


def _run_ladder_models(
    features_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    mlflow_tracking_uri: str | None = None,
    test_outputs: tuple[Path, Path] | None = None,
) -> dict:
    """Fit Stage 1 and Stage 2 on train, select on validation, and score val only."""
    import mlflow

    _seed_everything()
    mlflow.set_tag("active_stage", "load_features")
    df = _load_rows(Path(features_path))
    # Hospital holdouts never enter fitting or early stopping, even in train weeks.
    train = df.loc[
        df["period"].eq("train")
        & df["hospital_split"].eq("train")
        & df["y"].notna()
        & df["beds_lag1"].notna()
    ].copy()
    val_train = df.loc[df["period"].eq("val") & df["hospital_split"].eq("train")].copy()
    val_holdout = df.loc[df["period"].eq("val") & df["hospital_split"].eq("holdout")].copy()
    val_train = val_train.loc[val_train["y"].notna() & val_train["beds_lag1"].notna()].copy()
    val_holdout = val_holdout.loc[
        val_holdout["y"].notna() & val_holdout["beds_lag1"].notna()
    ].copy()
    persistence_error_p90 = float(train["change"].quantile(0.90))

    model_run_ids = {}
    mlflow.log_params(
        {
            "train_rows": len(train),
            "validation_rows": len(val_train),
            "holdout_rows": len(val_holdout),
            "target": "y - beds_lag1",
        }
    )
    # Stage 1: the one-parameter momentum rule and a linear change model.
    mlflow.set_tag("active_stage", "momentum")
    with experiment_run("momentum-fit", params={}):
        model_run_ids["momentum"] = mlflow.active_run().info.run_id
        mlflow.set_tag("run_kind", "model_fit")
        mlflow.log_param("seed", RANDOM_SEED)
        usable = train[["change", "momentum"]].dropna()
        # Closed-form least squares through the origin; no recent movement means
        # no predicted correction to persistence.
        denominator = float(np.dot(usable["momentum"], usable["momentum"]))
        phi = (
            float(np.dot(usable["momentum"], usable["change"]) / denominator)
            if denominator
            else 0.0
        )
        mlflow.log_param("phi", phi)

    with experiment_run(
        "linear-fit", params={"estimator": "LinearRegression", "imputation": "train median"}
    ):
        model_run_ids["linear"] = mlflow.active_run().info.run_id
        mlflow.set_tag("run_kind", "model_fit")
        mlflow.log_param("seed", RANDOM_SEED)
        linear_preprocessor = ColumnTransformer(
            [
                (
                    "numeric",
                    make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                    list(STAGE_NUMERIC_FEATURES),
                ),
                ("state", OneHotEncoder(handle_unknown="ignore"), ["state"]),
            ]
        )
        linear = make_pipeline(linear_preprocessor, LinearRegression())
        mlflow.set_tag("active_stage", "linear")
        linear.fit(_linear_features(train), train["change"])

    # Stage 2 comparison: same change target and train/validation split.
    with experiment_run(
        "lightgbm-fit",
        params={
            "objective": "quantile",
            "alpha": 0.5,
            "n_estimators": 3000,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 100,
            "patience": 100,
        },
    ):
        model_run_ids["lightgbm"] = mlflow.active_run().info.run_id
        mlflow.set_tag("run_kind", "model_fit")
        mlflow.log_param("seed", RANDOM_SEED)
        lgbm = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.5,
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=100,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
        )
        lgbm_preprocessor = ColumnTransformer(
            [
                (
                    "numeric",
                    make_pipeline(SimpleImputer(strategy="median")),
                    list(STAGE_NUMERIC_FEATURES),
                ),
                ("state", OneHotEncoder(handle_unknown="ignore"), ["state"]),
            ]
        )
        train_lgb = lgbm_preprocessor.fit_transform(_linear_features(train))
        val_lgb = lgbm_preprocessor.transform(_linear_features(val_train))
        mlflow.set_tag("active_stage", "lightgbm")
        mlflow.log_params(
            {
                "tree_estimators": 3000,
                "tree_learning_rate": 0.03,
                "mlp_epochs": 60,
                "mlp_batch_size": 1024,
                "mlp_learning_rate": 0.001,
            }
        )
        lgbm.fit(
            train_lgb,
            train["change"],
            eval_X=val_lgb,
            eval_y=val_train["change"],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        mlflow.log_param("best_iteration", lgbm.best_iteration_)

    mlflow.set_tag("active_stage", "quantile_mlp")
    with experiment_run(
        "quantile_mlp-fit",
        params={
            "epochs": 60,
            "batch_size": 1024,
            "learning_rate": 0.001,
            "patience": 8,
            "embedding_size": 8,
            "hidden_sizes": "64,32",
            "dropout": 0.1,
            "weight_decay": 1e-5,
        },
    ):
        model_run_ids["quantile_mlp"] = mlflow.active_run().info.run_id
        mlflow.set_tag("run_kind", "model_fit")
        mlflow.log_param("seed", RANDOM_SEED)
        mlp, mlp_preprocessor, state_map, mlp_change_q, best_val_pinball = _fit_mlp(
            train, val_train
        )
        mlflow.log_metric("best_validation_pinball", best_val_pinball)

    mlflow.set_tag("active_stage", "save_models")
    parent_run_id = mlflow.active_run().info.run_id
    # Keep fitted bundles outside ignored data so the team can commit and reuse them.
    models_path = PROJECT_ROOT / "models" / parent_run_id
    _save_ladder_models(
        models_path,
        phi=phi,
        linear=linear,
        tree=lgbm,
        tree_preprocessor=lgbm_preprocessor,
        mlp=mlp,
        mlp_preprocessor=mlp_preprocessor,
        state_map=state_map,
        metadata={
            "parent_run_id": parent_run_id,
            "model_run_ids": model_run_ids,
            "features_sha256": file_digest(Path(features_path)),
            "source_sha256": file_digest(Path(__file__)),
            "seed": RANDOM_SEED,
            "persistence_p90_change_error": persistence_error_p90,
            "protocol": "train-only fit; validation-only selection; no train-plus-val refit",
        },
    )
    mlflow.log_artifacts(str(models_path), artifact_path="models")

    mlflow.set_tag("active_stage", "evaluation")
    # Add each change forecast to observed origin occupancy; bed predictions
    # cannot be negative but are not capped at the reported capacity.
    val_linear_change = linear.predict(_linear_features(val_train))
    val_momentum_change = phi * val_train["momentum"].fillna(0.0).to_numpy(dtype=float)
    val_tree_change = lgbm.predict(val_lgb, num_iteration=lgbm.best_iteration_)
    val_base = val_train["beds_lag1"].to_numpy(dtype=float)
    val_predictions = {
        "momentum": np.maximum(0.0, val_base + val_momentum_change),
        "linear": np.maximum(0.0, val_base + val_linear_change),
        "lightgbm": np.maximum(0.0, val_base + val_tree_change),
        "quantile_mlp": np.maximum(0.0, val_base + mlp_change_q[:, 1]),
    }

    # A single common validation mask makes persistence comparisons fair.
    common = val_train["y"].notna() & val_train["beds_lag1"].notna()
    val_scored = val_train.loc[common].copy()
    # Evaluate the same fitted models on unseen hospitals in validation.
    holdout_linear = np.maximum(
        0.0,
        val_holdout["beds_lag1"].to_numpy(dtype=float)
        + linear.predict(_linear_features(val_holdout)),
    )
    holdout_momentum = np.maximum(
        0.0,
        val_holdout["beds_lag1"].to_numpy(dtype=float)
        + phi * val_holdout["momentum"].fillna(0.0).to_numpy(dtype=float),
    )
    holdout_tree_x = lgbm_preprocessor.transform(_linear_features(val_holdout))
    holdout_tree = np.maximum(
        0.0,
        val_holdout["beds_lag1"].to_numpy(dtype=float)
        + lgbm.predict(holdout_tree_x, num_iteration=lgbm.best_iteration_),
    )
    holdout_mlp_q = _predict_mlp_quantiles(mlp, mlp_preprocessor, state_map, val_holdout)
    holdout_base = val_holdout["beds_lag1"].to_numpy(dtype=float)
    predictions = val_scored[
        ["hospital_pk", "target_week", "period", "hospital_split", "y", "beds_lag1"]
    ].copy()
    for name, values in val_predictions.items():
        predictions[name] = values[common.to_numpy()]
    predictions["quantile_mlp_p10"] = np.maximum(
        0.0, val_base[common.to_numpy()] + mlp_change_q[common.to_numpy(), 0]
    )
    predictions["quantile_mlp_p50"] = predictions["quantile_mlp"]
    predictions["quantile_mlp_p90"] = np.maximum(
        0.0, val_base[common.to_numpy()] + mlp_change_q[common.to_numpy(), 2]
    )

    if not val_holdout.empty:
        holdout_predictions = val_holdout[
            ["hospital_pk", "target_week", "period", "hospital_split", "y", "beds_lag1"]
        ].copy()
        holdout_predictions["momentum"] = holdout_momentum
        holdout_predictions["linear"] = holdout_linear
        holdout_predictions["lightgbm"] = holdout_tree
        holdout_predictions["quantile_mlp"] = np.maximum(0.0, holdout_base + holdout_mlp_q[:, 1])
        holdout_predictions["quantile_mlp_p10"] = np.maximum(
            0.0, holdout_base + holdout_mlp_q[:, 0]
        )
        holdout_predictions["quantile_mlp_p50"] = holdout_predictions["quantile_mlp"]
        holdout_predictions["quantile_mlp_p90"] = np.maximum(
            0.0, holdout_base + holdout_mlp_q[:, 2]
        )
        predictions = pd.concat([predictions, holdout_predictions], ignore_index=True)
    predictions_path = Path(predictions_path)
    metrics_path = Path(metrics_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_path, index=False)

    metrics = {
        "models_path": str(models_path.resolve()),
        "model_run_ids": model_run_ids,
        "protocol": "train-only fit; validation-only selection and scoring; test not opened",
        "target": "change = y - beds_lag1",
        "momentum_phi": phi,
        "persistence_p90_change_error": persistence_error_p90,
        "quantile_mlp_best_validation_pinball": best_val_pinball,
        "groups": {},
    }
    metrics["shared_evaluation"] = {}
    # Score both hospital populations through the shared harness so every model
    # uses the same rows and persistence comparator within each population.
    for split in ("train", "holdout"):
        part = predictions.loc[predictions["hospital_split"].eq(split)]
        report, _ = score_models(
            part,
            {name: name for name in val_predictions},
            quantiles={
                "quantile_mlp": ["quantile_mlp_p10", "quantile_mlp_p50", "quantile_mlp_p90"]
            },
            persistence_error_p90=persistence_error_p90,
            require_complete=True,
        )
        key = f"val/{split}_hospitals"
        metrics["shared_evaluation"][key] = report
        metrics["groups"][key] = report["models"]
    metrics["checks"] = evaluate_checks(
        metrics["groups"]["val/train_hospitals"], metrics["groups"]["val/holdout_hospitals"]
    )
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _log_mlflow(metrics, predictions_path, metrics_path, mlflow_tracking_uri)
    if test_outputs is not None:
        return _score_final_models(
            features_path,
            *test_outputs,
            phi=phi,
            linear=linear,
            tree=lgbm,
            tree_preprocessor=lgbm_preprocessor,
            mlp=mlp,
            mlp_preprocessor=mlp_preprocessor,
            state_map=state_map,
            selection=metrics,
        )
    return metrics


def _seed_everything() -> None:
    np.random.seed(RANDOM_SEED)  # noqa: NPY002 -- seed libraries using NumPy's legacy global RNG
    torch.manual_seed(RANDOM_SEED)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def _log_mlflow(metrics, predictions_path, metrics_path, tracking_uri):
    import mlflow

    # Score runs refer back to fit runs, distinguishing fitting from evaluation.
    for name in metrics["groups"]["val/train_hospitals"]:
        with experiment_run(
            name,
            tracking_uri,
            {"stage": 0 if name == "persistence" else 1 if name in {"momentum", "linear"} else 2},
        ):
            mlflow.set_tag("run_kind", "validation_score")
            if name in metrics["model_run_ids"]:
                mlflow.set_tag("fit_run_id", metrics["model_run_ids"][name])
            mlflow.log_param(
                "validation_row_sha256",
                metrics["shared_evaluation"]["val/train_hospitals"]["row_sha256"],
            )
            for group, scores in metrics["groups"].items():
                log_metrics_tree(scores[name], group)
            mlflow.log_artifact(str(predictions_path))
            mlflow.log_artifact(str(metrics_path))


@tracked_experiment("ladder-training")
def run_ladder_models(
    features_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    mlflow_tracking_uri: str | None = None,
) -> dict:
    """Track the entire attempt, including validation, fitting and scoring failures."""
    return _run_ladder_models(features_path, predictions_path, metrics_path, mlflow_tracking_uri)


def _score_final_models(
    features_path,
    predictions_path,
    metrics_path,
    *,
    phi,
    linear,
    tree,
    tree_preprocessor,
    mlp,
    mlp_preprocessor,
    state_map,
    selection,
):
    """Score frozen validation-selected models; no fitting happens in this function."""
    import mlflow

    mlflow.set_tag("active_stage", "final_test")
    frame = load_model_rows(features_path, include_test=True)
    frame = frame.loc[frame["period"].eq("test")].copy()
    names = ("momentum", "linear", "lightgbm", "quantile_mlp")
    quantiles = [f"quantile_mlp_p{q}" for q in (10, 50, 90)]
    metrics = {
        "run_id": mlflow.active_run().info.run_id,
        "models_path": selection.get("models_path"),
        "model_run_ids": selection["model_run_ids"],
        "protocol": (
            "train-only fit; validation-only selection; frozen weights; rolling one-week test"
        ),
        "refit_train_plus_validation": False,
        "test_start": str(frame.target_week.min().date()),
        "test_end": str(frame.target_week.max().date()),
        "selection_validation_groups": selection["groups"],
        "persistence_p90_change_error": selection["persistence_p90_change_error"],
        "groups": {},
        "shared_evaluation": {},
    }
    predictions = []
    for split in ("train", "holdout"):
        candidates = frame.loc[frame.hospital_split.eq(split)]
        # Eligibility depends on observed targets and persistence, never on which
        # model happens to perform well. Missing model outputs fail the scorer.
        eligible = np.isfinite(candidates[["y", "beds_lag1"]].to_numpy(dtype=float)).all(axis=1)
        part = candidates.loc[eligible].copy()
        base = part.beds_lag1.to_numpy(dtype=float)
        change_q = _predict_mlp_quantiles(mlp, mlp_preprocessor, state_map, part)
        out = part[
            [
                "hospital_pk",
                "origin_week",
                "target_week",
                "period",
                "hospital_split",
                "y",
                "beds_lag1",
            ]
        ].copy()
        out["momentum"] = np.maximum(0, base + phi * part.momentum.fillna(0).to_numpy())
        out["linear"] = np.maximum(0, base + linear.predict(_linear_features(part)))
        out["lightgbm"] = np.maximum(
            0,
            base
            + tree.predict(
                tree_preprocessor.transform(_linear_features(part)),
                num_iteration=tree.best_iteration_,
            ),
        )
        for i, column in enumerate(quantiles):
            out[column] = np.maximum(0, base + change_q[:, i])
        out["quantile_mlp"] = out["quantile_mlp_p50"]
        report, _ = score_models(
            out,
            {name: name for name in names},
            quantiles={"quantile_mlp": quantiles},
            persistence_error_p90=selection["persistence_p90_change_error"],
            require_complete=True,
        )
        report["total_test_rows"] = len(candidates)
        report["excluded_missing_target_or_baseline"] = int((~eligible).sum())
        key = f"test/{split}_hospitals"
        metrics["shared_evaluation"][key] = report
        metrics["groups"][key] = report["models"]
        predictions.append(out)
    metrics["checks"] = evaluate_checks(
        metrics["groups"]["test/train_hospitals"], metrics["groups"]["test/holdout_hospitals"]
    )
    pd.concat(predictions, ignore_index=True).to_parquet(predictions_path, index=False)
    Path(metrics_path).write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for name in ("persistence", *names):
        with experiment_run(f"test-{name}"):
            mlflow.set_tag("run_kind", "test_score")
            if name in metrics["model_run_ids"]:
                mlflow.set_tag("fit_run_id", metrics["model_run_ids"][name])
            for group, scores in metrics["groups"].items():
                log_metrics_tree(scores[name], group)
                mlflow.log_param(
                    f"{group}/row_sha256", metrics["shared_evaluation"][group]["row_sha256"]
                )
            mlflow.log_artifact(str(predictions_path))
            mlflow.log_artifact(str(metrics_path))
    return metrics


@tracked_experiment("ladder-final-test")
def run_final_test(
    features_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    mlflow_tracking_uri: str | None = None,
) -> dict:
    """Reproduce model selection, then open test; keep validation results separate."""
    output = Path(metrics_path).parent / "final_test"
    return _run_ladder_models(
        features_path,
        output / "selection_predictions.parquet",
        output / "selection_metrics.json",
        mlflow_tracking_uri,
        test_outputs=(Path(predictions_path), Path(metrics_path)),
    )
