from __future__ import annotations

import argparse
import json
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import ensure_parent_dir, load_config, resolve_path


def _to_dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _select_best_tree_model(metrics_df: pd.DataFrame) -> str:
    tree_candidates = metrics_df[metrics_df["model"].isin(["rf", "xgb"])].copy()
    if tree_candidates.empty:
        raise ValueError("No RF/XGB rows found in metrics for SHAP analysis.")
    return tree_candidates.sort_values("f1", ascending=False).iloc[0]["model"]


def _load_tree_pipeline(model_name: str):
    pipeline_path = resolve_path(f"artifacts/models/{model_name}.joblib")
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {pipeline_path}")
    pipeline = joblib.load(pipeline_path)
    return pipeline, pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]


def _format_feature_label(feature_name: str) -> str:
    raw = feature_name
    for prefix in ("num__", "cat__"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]

    direct_labels = {
        "review_count": "Review Count",
        "is_open": "Business Currently Open",
        "num_days_open": "Days Open per Week",
        "avg_open_hours": "Average Open Hours per Day",
        "weekend_open": "Weekend Availability",
    }
    if raw in direct_labels:
        return direct_labels[raw]

    if raw.startswith("attr_"):
        return raw.removeprefix("attr_").replace("_", " ").title()
    if raw.startswith("cat_"):
        return f"{raw.removeprefix('cat_').replace('_', ' ').title()} Category"
    if raw.startswith("state_"):
        return f"State = {raw.split('_', 1)[1].upper()}"
    if raw.startswith("city_"):
        return f"City = {raw.split('_', 1)[1].replace('_', ' ').title()}"

    return raw.replace("_", " ").title()


def _describe_direction(feature_values: np.ndarray, shap_column: np.ndarray) -> str:
    feature_values = np.asarray(feature_values, dtype=float)
    shap_column = np.asarray(shap_column, dtype=float)
    if np.allclose(feature_values, feature_values[0]) or np.allclose(shap_column, shap_column[0]):
        return "Its effect varies with the broader business profile."

    corr = np.corrcoef(feature_values, shap_column)[0, 1]
    if np.isnan(corr):
        return "Its effect varies with the broader business profile."
    if corr >= 0.15:
        return "Higher values generally push predictions toward the high-rating class."
    if corr <= -0.15:
        return "Higher values generally push predictions away from the high-rating class."
    return "Its effect depends on interactions with the rest of the business profile."


def run_shap_analysis(config_path: str = "configs/config.yaml") -> None:
    config = load_config(config_path)
    target_col = config["problem"]["target_column"]
    random_state = int(config["project"]["random_state"])

    metrics_path = resolve_path(config["paths"]["metrics_csv"])
    modeling_path = resolve_path(config["paths"]["modeling_table"])
    shap_dir = resolve_path("artifacts/shap")
    shap_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}. Run model training first.")
    if not modeling_path.exists():
        raise FileNotFoundError(f"Missing modeling table: {modeling_path}. Run data preparation first.")

    metrics_df = pd.read_csv(metrics_path)
    requested_model_name = _select_best_tree_model(metrics_df)

    try:
        import shap  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError("shap is required. Install dependencies from requirements.txt.") from exc

    df = pd.read_parquet(modeling_path)
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].astype(int).copy()
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=float(config["problem"]["test_size"]),
        random_state=random_state,
        stratify=y,
    )

    background_path = resolve_path(config["paths"]["shap_background"])
    if background_path.exists():
        X_background = pd.read_parquet(background_path)
    else:
        X_background = X.sample(min(1000, len(X)), random_state=random_state)

    X_explain = X_test.sample(min(400, len(X_test)), random_state=random_state)
    selected_model_name = requested_model_name
    fallback_reason: str | None = None
    last_error: Exception | None = None

    for candidate_model in ([requested_model_name, "rf"] if requested_model_name == "xgb" else [requested_model_name]):
        pipeline, preprocessor, tree_model = _load_tree_pipeline(candidate_model)
        X_background_t = _to_dense(preprocessor.transform(X_background))
        X_explain_t = _to_dense(preprocessor.transform(X_explain))
        feature_names = preprocessor.get_feature_names_out()

        try:
            explainer = shap.TreeExplainer(tree_model, X_background_t)
            shap_values_raw = explainer(X_explain_t, check_additivity=False)
            shap_values = shap_values_raw.values if hasattr(shap_values_raw, "values") else shap_values_raw
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, -1]
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(np.asarray(base_value).ravel()[-1])
            else:
                base_value = float(base_value)
            selected_model_name = candidate_model
            if candidate_model != requested_model_name:
                fallback_reason = f"Used {candidate_model.upper()} because SHAP failed for {requested_model_name.upper()}: {last_error}"
            break
        except ValueError as exc:
            last_error = exc
            if candidate_model == "rf":
                raise
    else:
        raise RuntimeError("Unable to initialize SHAP explainer for available tree models.") from last_error

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_explain_t, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(shap_dir / "summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_explain_t, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(shap_dir / "bar.png", dpi=160, bbox_inches="tight")
    plt.close()

    proba = pipeline.predict_proba(X_explain)[:, 1]
    sample_idx = int(np.argmax(proba))
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=base_value,
        data=X_explain_t[sample_idx],
        feature_names=feature_names,
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(shap_dir / "waterfall_example.png", dpi=160, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:8]
    top_features: list[dict[str, Any]] = []
    lines = [
        f"# SHAP Interpretation ({selected_model_name.upper()})",
        "",
        "This section explains which business characteristics most influenced the model's high-rating predictions.",
        "",
    ]
    if fallback_reason:
        lines.append(f"- Note: {fallback_reason}")
        lines.append("")

    lines.append("Top drivers and what they mean:")
    for idx in top_idx:
        feature_label = _format_feature_label(str(feature_names[idx]))
        direction_summary = _describe_direction(X_explain_t[:, idx], shap_values[:, idx])
        top_features.append(
            {
                "feature_raw": str(feature_names[idx]),
                "feature_label": feature_label,
                "mean_abs_shap": float(mean_abs_shap[idx]),
                "direction_summary": direction_summary,
            }
        )
        lines.append(f"- **{feature_label}**: {direction_summary}")

    lines.append("")
    lines.append("Decision-maker takeaway:")
    lines.append(
        "- The strongest signals come from operating intensity, customer engagement, and business type. "
        "A stakeholder can use these drivers to understand whether a business profile looks structurally strong "
        "or whether it is missing signals commonly associated with highly rated businesses."
    )
    interpretation_path = shap_dir / "interpretation.md"
    ensure_parent_dir(interpretation_path)
    interpretation_path.write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "requested_tree_model": requested_model_name,
        "selected_tree_model": selected_model_name,
        "fallback_reason": fallback_reason,
        "n_background_rows": int(X_background.shape[0]),
        "n_explained_rows": int(X_explain.shape[0]),
        "waterfall_example_true_label": int(y_test.loc[X_explain.index].iloc[sample_idx]),
        "waterfall_example_pred_proba": float(proba[sample_idx]),
        "top_features": top_features,
    }
    ensure_parent_dir(shap_dir / "metadata.json")
    (shap_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved SHAP artifacts to: {shap_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP explainability on best tree model.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_shap_analysis(config_path=args.config)


if __name__ == "__main__":
    main()
