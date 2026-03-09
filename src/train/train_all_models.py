from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from src.common import ensure_parent_dir, load_config, load_json, resolve_path, save_json

sns.set_theme(style="whitegrid")


def _build_preprocessor(X: pd.DataFrame, sparse_output: bool = True) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=sparse_output,
                    min_frequency=0.01,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["auc_roc"] = float("nan")
    return metrics


def _plot_roc_curve(y_true: pd.Series, y_proba: np.ndarray, title: str, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _train_grid_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    estimator: Any,
    param_grid: dict[str, list[Any]],
    cv_folds: int,
    scoring: str,
) -> GridSearchCV:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(X_train, sparse_output=True)),
            ("model", estimator),
        ]
    )
    grid = {f"model__{k}": v for k, v in param_grid.items()}
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def _run_mlp_subprocess(config_path: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [sys.executable, "-m", "src.train.train_mlp", "--config", config_path],
        check=True,
        cwd=str(resolve_path(".")),
        env=env,
    )

    mlp_metrics = load_json("artifacts/metrics/mlp_metrics.json")
    mlp_params = load_json("artifacts/metrics/mlp_best_params.json")
    mlp_roc = load_json("artifacts/metrics/mlp_roc.json")
    return mlp_metrics, mlp_params, mlp_roc


def run_training(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    random_state = int(config["project"]["random_state"])
    target_col = config["problem"]["target_column"]
    cv_folds = int(config["modeling"]["cv_folds"])
    scoring = str(config["modeling"]["scoring"])

    modeling_table_path = resolve_path(config["paths"]["modeling_table"])
    if not modeling_table_path.exists():
        raise FileNotFoundError(
            f"Missing modeling table: {modeling_table_path}. Run python -m src.data.prepare_business_table first."
        )

    df = pd.read_parquet(modeling_table_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in modeling table.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].astype(int).copy()

    if "stars" in X.columns:
        raise AssertionError("Leakage guard failed: 'stars' should not be in feature matrix.")
    if target_col in X.columns:
        raise AssertionError("Leakage guard failed: target column present in feature matrix.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(config["problem"]["test_size"]),
        random_state=random_state,
        stratify=y,
    )

    resolve_path(config["paths"]["shap_background"]).parent.mkdir(parents=True, exist_ok=True)
    X_train.sample(min(1500, len(X_train)), random_state=random_state).to_parquet(
        resolve_path(config["paths"]["shap_background"]), index=False
    )

    metrics_rows: list[dict[str, Any]] = []
    best_params: dict[str, Any] = {}
    roc_payload: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    models_dir = resolve_path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = resolve_path("artifacts/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Logistic regression baseline.
    start = time.perf_counter()
    logreg_pipe = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(X_train, sparse_output=True)),
            (
                "model",
                LogisticRegression(
                    max_iter=int(config["modeling"]["logistic"]["max_iter"]),
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    logreg_pipe.fit(X_train, y_train)
    logreg_proba = logreg_pipe.predict_proba(X_test)[:, 1]
    logreg_pred = (logreg_proba >= 0.5).astype(int)
    elapsed = time.perf_counter() - start
    logreg_metrics = _classification_metrics(y_test, logreg_pred, logreg_proba)
    logreg_metrics.update({"model": "logreg", "train_time_sec": elapsed})
    metrics_rows.append(logreg_metrics)
    joblib.dump(logreg_pipe, models_dir / "logreg.joblib")
    _plot_roc_curve(y_test, logreg_proba, "Logistic Regression ROC", figures_dir / "roc_logreg.png")
    fpr, tpr, _ = roc_curve(y_test, logreg_proba)
    roc_payload["logreg"] = (fpr, tpr, logreg_metrics["auc_roc"])
    best_params["logreg"] = {}

    # Decision tree.
    start = time.perf_counter()
    dt_search = _train_grid_model(
        X_train,
        y_train,
        DecisionTreeClassifier(class_weight="balanced", random_state=random_state),
        config["modeling"]["decision_tree_grid"],
        cv_folds=cv_folds,
        scoring=scoring,
    )
    dt_model = dt_search.best_estimator_
    dt_proba = dt_model.predict_proba(X_test)[:, 1]
    dt_pred = (dt_proba >= 0.5).astype(int)
    elapsed = time.perf_counter() - start
    dt_metrics = _classification_metrics(y_test, dt_pred, dt_proba)
    dt_metrics.update({"model": "dt", "train_time_sec": elapsed})
    metrics_rows.append(dt_metrics)
    joblib.dump(dt_model, models_dir / "dt.joblib")
    _plot_roc_curve(y_test, dt_proba, "Decision Tree ROC", figures_dir / "roc_dt.png")
    fpr, tpr, _ = roc_curve(y_test, dt_proba)
    roc_payload["dt"] = (fpr, tpr, dt_metrics["auc_roc"])
    best_params["dt"] = {
        key.replace("model__", ""): value for key, value in dt_search.best_params_.items()
    }
    dt_preprocessor = dt_model.named_steps["preprocessor"]
    dt_estimator = dt_model.named_steps["model"]
    dt_feature_names = dt_preprocessor.get_feature_names_out()
    plt.figure(figsize=(24, 10))
    plot_tree(
        dt_estimator,
        max_depth=3,
        feature_names=dt_feature_names,
        class_names=["low_rating", "high_rating"],
        filled=True,
        rounded=True,
        fontsize=7,
    )
    plt.title("Best Decision Tree (Top 3 Levels)")
    plt.tight_layout()
    plt.savefig(figures_dir / "dt_best_tree.png", dpi=160)
    plt.close()

    # Random forest.
    start = time.perf_counter()
    rf_search = _train_grid_model(
        X_train,
        y_train,
        RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        config["modeling"]["random_forest_grid"],
        cv_folds=cv_folds,
        scoring=scoring,
    )
    rf_model = rf_search.best_estimator_
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)
    elapsed = time.perf_counter() - start
    rf_metrics = _classification_metrics(y_test, rf_pred, rf_proba)
    rf_metrics.update({"model": "rf", "train_time_sec": elapsed})
    metrics_rows.append(rf_metrics)
    joblib.dump(rf_model, models_dir / "rf.joblib")
    _plot_roc_curve(y_test, rf_proba, "Random Forest ROC", figures_dir / "roc_rf.png")
    fpr, tpr, _ = roc_curve(y_test, rf_proba)
    roc_payload["rf"] = (fpr, tpr, rf_metrics["auc_roc"])
    best_params["rf"] = {
        key.replace("model__", ""): value for key, value in rf_search.best_params_.items()
    }

    # XGBoost.
    try:
        from xgboost import XGBClassifier  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError("xgboost is required. Install dependencies from requirements.txt.") from exc

    start = time.perf_counter()
    class_counts = y_train.value_counts()
    scale_pos_weight = float(class_counts.get(0, 1) / max(class_counts.get(1, 1), 1))
    xgb_search = _train_grid_model(
        X_train,
        y_train,
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
        ),
        config["modeling"]["xgboost_grid"],
        cv_folds=cv_folds,
        scoring=scoring,
    )
    xgb_model = xgb_search.best_estimator_
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_proba >= 0.5).astype(int)
    elapsed = time.perf_counter() - start
    xgb_metrics = _classification_metrics(y_test, xgb_pred, xgb_proba)
    xgb_metrics.update({"model": "xgb", "train_time_sec": elapsed})
    metrics_rows.append(xgb_metrics)
    joblib.dump(xgb_model, models_dir / "xgb.joblib")
    _plot_roc_curve(y_test, xgb_proba, "XGBoost ROC", figures_dir / "roc_xgb.png")
    fpr, tpr, _ = roc_curve(y_test, xgb_proba)
    roc_payload["xgb"] = (fpr, tpr, xgb_metrics["auc_roc"])
    best_params["xgb"] = {
        key.replace("model__", ""): value for key, value in xgb_search.best_params_.items()
    }

    # Run MLP in a fresh process to avoid post-GridSearch hangs from the worker pool.
    mlp_metrics, mlp_params, mlp_roc = _run_mlp_subprocess(config_path)
    metrics_rows.append(mlp_metrics)
    roc_payload["mlp"] = (
        np.asarray(mlp_roc["fpr"]),
        np.asarray(mlp_roc["tpr"]),
        float(mlp_roc["auc_roc"]),
    )
    best_params.update(mlp_params)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("f1", ascending=False)
    ensure_parent_dir(config["paths"]["metrics_csv"])
    metrics_df.to_csv(resolve_path(config["paths"]["metrics_csv"]), index=False)
    save_json(best_params, config["paths"]["best_params_json"])

    # Comparison chart.
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=metrics_df,
        x="model",
        y="f1",
        hue="model",
        palette="crest",
        dodge=False,
        legend=False,
    )
    plt.title("Model Comparison by Test F1")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison_f1.png", dpi=160)
    plt.close()

    # Combined ROC chart.
    plt.figure(figsize=(8, 6))
    for model_name, (fpr_arr, tpr_arr, roc_auc) in roc_payload.items():
        plt.plot(fpr_arr, tpr_arr, label=f"{model_name.upper()} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Across Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curves_all_models.png", dpi=160)
    plt.close()

    print(f"Saved metrics to {resolve_path(config['paths']['metrics_csv'])}")
    return metrics_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all HW1 models and save artifacts.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_df = run_training(config_path=args.config)
    print(metrics_df)


if __name__ == "__main__":
    main()
