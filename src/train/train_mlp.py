from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.common import ensure_parent_dir, load_config, resolve_path, save_json
from src.train.torch_mlp import YelpMLP
from src.train.train_all_models import _build_preprocessor, _classification_metrics, _plot_roc_curve

sns.set_theme(style="whitegrid")


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32)


def _evaluate_model(
    model: nn.Module,
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    criterion: nn.Module,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor).squeeze(-1)
        loss = criterion(logits, y_tensor).item()
        proba = torch.sigmoid(logits).cpu().numpy()
        pred = (proba >= 0.5).astype(int)
        acc = float((pred == y_tensor.cpu().numpy().astype(int)).mean())
    return loss, acc, proba


def _train_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_units: int,
    dropout_rate: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
) -> tuple[nn.Module, dict[str, list[float]]]:
    _set_torch_seed(seed)

    X_train_t = _to_tensor(X_train)
    y_train_t = _to_tensor(y_train)
    X_val_t = _to_tensor(X_val)
    y_val_t = _to_tensor(y_val)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    model = YelpMLP(
        input_dim=X_train.shape[1],
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        train_loss, train_acc, _ = _evaluate_model(model, X_train_t, y_train_t, criterion)
        val_loss, val_acc, _ = _evaluate_model(model, X_val_t, y_val_t, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _plot_mlp_history(history: dict[str, list[float]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].set_title("MLP Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_accuracy"], label="train_accuracy")
    axes[1].plot(history["val_accuracy"], label="val_accuracy")
    axes[1].set_title("MLP Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _run_bonus_mlp_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict[str, Any],
    random_state: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    tuning_cfg = config["mlp"]["bonus_tuning"]
    candidates = list(
        product(
            tuning_cfg["hidden_units"],
            tuning_cfg["dropout_rates"],
            tuning_cfg["learning_rates"],
        )
    )[: int(tuning_cfg["max_trials"])]

    # Use a subset for the bonus sweep to keep runtime reasonable.
    sample_size = min(20000, len(X_train))
    rng = np.random.default_rng(random_state)
    train_idx = rng.choice(len(X_train), size=sample_size, replace=False)
    val_size = min(5000, len(X_val))
    val_idx = rng.choice(len(X_val), size=val_size, replace=False)

    X_train_small = X_train[train_idx]
    y_train_small = y_train[train_idx]
    X_val_small = X_val[val_idx]
    y_val_small = y_val[val_idx]

    results: list[dict[str, Any]] = []
    for trial_idx, (hidden_units, dropout_rate, learning_rate) in enumerate(candidates):
        model, _ = _train_torch_mlp(
            X_train_small,
            y_train_small,
            X_val_small,
            y_val_small,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            batch_size=int(config["mlp"]["batch_size"]),
            epochs=max(8, int(config["mlp"]["epochs"]) // 2),
            patience=max(2, int(config["mlp"]["early_stopping_patience"]) - 1),
            seed=random_state + trial_idx + 1,
        )
        logits = model(_to_tensor(X_val_small)).squeeze(-1)
        val_proba = torch.sigmoid(logits).detach().cpu().numpy()
        val_pred = (val_proba >= 0.5).astype(int)
        results.append(
            {
                "hidden_units": hidden_units,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
                "val_f1": float(f1_score(y_val_small, val_pred, zero_division=0)),
                "val_auc_roc": float(roc_auc_score(y_val_small, val_proba)),
            }
        )

    if not results:
        return results, None
    best = sorted(results, key=lambda row: row["val_f1"], reverse=True)[0]
    return results, best


def run_mlp_training(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    random_state = int(config["project"]["random_state"])
    target_col = config["problem"]["target_column"]
    modeling_table_path = resolve_path(config["paths"]["modeling_table"])

    if not modeling_table_path.exists():
        raise FileNotFoundError(
            f"Missing modeling table: {modeling_table_path}. Run python -m src.data.prepare_business_table first."
        )

    df = pd.read_parquet(modeling_table_path)
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(config["problem"]["test_size"]),
        random_state=random_state,
        stratify=y,
    )

    preprocessor_dense = _build_preprocessor(X_train, sparse_output=False)
    X_train_dense = preprocessor_dense.fit_transform(X_train).astype(np.float32)
    X_test_dense = preprocessor_dense.transform(X_test).astype(np.float32)
    y_train_np = y_train.to_numpy().astype(np.float32)
    y_test_np = y_test.to_numpy().astype(int)

    X_mlp_train, X_mlp_val, y_mlp_train, y_mlp_val = train_test_split(
        X_train_dense,
        y_train_np,
        test_size=float(config["mlp"]["validation_split"]),
        random_state=random_state,
        stratify=y_train_np.astype(int),
    )

    start = time.perf_counter()
    model, history = _train_torch_mlp(
        X_mlp_train,
        y_mlp_train,
        X_mlp_val,
        y_mlp_val,
        hidden_units=128,
        dropout_rate=0.0,
        learning_rate=0.001,
        batch_size=int(config["mlp"]["batch_size"]),
        epochs=int(config["mlp"]["epochs"]),
        patience=int(config["mlp"]["early_stopping_patience"]),
        seed=random_state,
    )
    logits = model(_to_tensor(X_test_dense)).squeeze(-1)
    mlp_proba = torch.sigmoid(logits).detach().cpu().numpy()
    mlp_pred = (mlp_proba >= 0.5).astype(int)
    elapsed = time.perf_counter() - start
    mlp_metrics = _classification_metrics(pd.Series(y_test_np), mlp_pred, mlp_proba)
    mlp_metrics.update({"model": "mlp", "train_time_sec": elapsed})

    models_dir = resolve_path("artifacts/models")
    figures_dir = resolve_path("artifacts/figures")
    metrics_dir = resolve_path("artifacts/metrics")
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor_dense, models_dir / "mlp_preprocessor.joblib")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(X_train_dense.shape[1]),
            "hidden_units": 128,
            "dropout_rate": 0.0,
        },
        models_dir / "mlp.pt",
    )
    _plot_mlp_history(history, figures_dir / "mlp_training_history.png")
    _plot_roc_curve(pd.Series(y_test_np), mlp_proba, "MLP ROC", figures_dir / "roc_mlp.png")

    best_params: dict[str, Any] = {
        "mlp": {
            "framework": "pytorch",
            "hidden_layers": [128, 128],
            "dropout_rate": 0.0,
            "learning_rate": 0.001,
            "optimizer": "adam",
        }
    }

    tuning_rows, tuning_best = _run_bonus_mlp_tuning(
        X_mlp_train,
        y_mlp_train,
        X_mlp_val,
        y_mlp_val,
        config=config,
        random_state=random_state,
    )
    if tuning_rows:
        tuning_df = pd.DataFrame(tuning_rows)
        ensure_parent_dir(config["paths"]["mlp_tuning_csv"])
        tuning_df.to_csv(resolve_path(config["paths"]["mlp_tuning_csv"]), index=False)
        best_params["mlp_bonus_best"] = tuning_best

        plot_df = tuning_df.sort_values("val_f1", ascending=False).reset_index(drop=True)
        plot_df["trial"] = [f"trial_{i+1}" for i in plot_df.index]
        plt.figure(figsize=(8, 5))
        sns.barplot(data=plot_df, x="val_f1", y="trial")
        plt.title("Bonus MLP Tuning Results (Validation F1)")
        plt.xlabel("Validation F1")
        plt.ylabel("Trial")
        plt.tight_layout()
        plt.savefig(figures_dir / "mlp_tuning_results.png", dpi=160)
        plt.close()

    fpr, tpr, _ = roc_curve(y_test_np, mlp_proba)
    save_json(mlp_metrics, metrics_dir / "mlp_metrics.json")
    save_json(best_params, metrics_dir / "mlp_best_params.json")
    save_json(
        {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc_roc": float(mlp_metrics["auc_roc"])},
        metrics_dir / "mlp_roc.json",
    )
    return mlp_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train only the PyTorch MLP model.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_mlp_training(config_path=args.config)
    print(metrics)


if __name__ == "__main__":
    main()
