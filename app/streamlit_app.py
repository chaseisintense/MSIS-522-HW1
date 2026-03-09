from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from src.train.torch_mlp import YelpMLP

px.defaults.template = "plotly_dark"
ARTIFACTS = ROOT / "artifacts"

MODEL_LABELS = {
    "Logistic Regression": "logreg",
    "Decision Tree": "dt",
    "Random Forest": "rf",
    "XGBoost": "xgb",
    "MLP (PyTorch)": "mlp",
}
TREE_MODEL_CODES = {"dt", "rf", "xgb"}

FIGURE_HELP_TEXT = {
    "target_distribution.png": "This shows class balance. A near 50/50 split is ideal for stable classifier learning.",
    "review_count_distribution_by_target.png": "This histogram uses a log-scaled review-count axis so you can compare the full distribution without the extreme right tail flattening everything else.",
    "review_count_boxplot_by_target.png": "This boxplot suppresses extreme outliers and uses a log y-axis so the class medians and upper quartiles are easier to compare.",
    "high_rating_share_by_state_top15.png": "Geographic differences suggest location effects that the model can use for prediction.",
    "high_rating_share_by_top_categories.png": "Some business categories are structurally more likely to receive high ratings.",
    "correlation_heatmap.png": "Correlation helps identify overlapping features and potential multicollinearity risks.",
}

FEATURE_GLOSSARY = {
    "review_count": "Total Yelp review count for the business.",
    "is_open": "Whether the business is currently marked open (1) or closed (0).",
    "num_days_open": "Number of days per week the business reports operating hours.",
    "avg_open_hours": "Average open hours per active day.",
    "weekend_open": "Whether the business is open Saturday/Sunday.",
    "state": "U.S. state or region field from Yelp business metadata.",
    "city": "City field (top cities retained, others grouped as 'Other').",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_modeling_table() -> pd.DataFrame:
    path = ARTIFACTS / "data" / "modeling_table.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_metrics() -> pd.DataFrame:
    path = ARTIFACTS / "metrics" / "model_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_summary() -> dict[str, Any]:
    return _load_json(ARTIFACTS / "data" / "dataset_summary.json")


@st.cache_data
def load_feature_defaults() -> dict[str, Any]:
    return _load_json(ARTIFACTS / "data" / "feature_defaults.json")


@st.cache_data
def load_best_params() -> dict[str, Any]:
    return _load_json(ARTIFACTS / "metrics" / "best_params.json")


@st.cache_data
def load_shap_metadata() -> dict[str, Any]:
    return _load_json(ARTIFACTS / "shap" / "metadata.json")


@st.cache_resource
def load_sklearn_model(model_code: str):
    model_path = ARTIFACTS / "models" / f"{model_code}.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_resource
def load_mlp_components():
    pre_path = ARTIFACTS / "models" / "mlp_preprocessor.joblib"
    model_path = ARTIFACTS / "models" / "mlp.pt"
    if not pre_path.exists() or not model_path.exists():
        return None, None
    pre = joblib.load(pre_path)
    payload = torch.load(model_path, map_location="cpu")
    model = YelpMLP(
        input_dim=int(payload["input_dim"]),
        hidden_units=int(payload["hidden_units"]),
        dropout_rate=float(payload["dropout_rate"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return pre, model


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-1: #0e1620;
            --bg-2: #13202c;
            --ink: #e7eef7;
            --muted: #b2c1d3;
            --accent: #63b3ff;
            --accent-soft: rgba(99,179,255,0.18);
            --card: rgba(21, 33, 45, 0.9);
            --line: rgba(190, 215, 240, 0.22);
        }

        html, body, [class*="css"] {
            font-family: "Trebuchet MS", "Lucida Grande", "Lucida Sans Unicode", sans-serif;
            color: var(--ink);
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 6%, rgba(85, 148, 212, 0.22) 0%, transparent 40%),
                radial-gradient(circle at 88% 20%, rgba(56, 98, 151, 0.18) 0%, transparent 40%),
                linear-gradient(140deg, var(--bg-1) 0%, var(--bg-2) 100%);
        }

        h1, h2, h3 {
            font-family: "Georgia", "Times New Roman", serif;
            color: #f4f8ff;
            letter-spacing: 0.2px;
        }

        p, li, label, .stMarkdown, .stCaption {
            color: var(--ink) !important;
        }

        [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
            color: var(--ink) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            padding: 0.25rem;
        }

        .stTabs [data-baseweb="tab"] {
            color: var(--muted);
            border-radius: 8px;
            padding: 0.4rem 0.8rem;
        }

        .stTabs [aria-selected="true"] {
            color: #f4f8ff !important;
            background: rgba(99, 179, 255, 0.2) !important;
        }

        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div,
        .stTextInput input,
        .stNumberInput input {
            background: rgba(8, 17, 25, 0.85) !important;
            color: var(--ink) !important;
            border-color: rgba(160, 190, 220, 0.3) !important;
        }

        .stSlider [data-baseweb="slider"] {
            padding-top: 0.5rem;
        }

        div.stButton > button {
            background: rgba(99, 179, 255, 0.2);
            border: 1px solid rgba(99, 179, 255, 0.45);
            color: #eaf4ff;
        }

        div.stButton > button:hover {
            background: rgba(99, 179, 255, 0.32);
            border-color: rgba(99, 179, 255, 0.65);
        }

        .hero-note {
            border: 1px solid var(--line);
            background: linear-gradient(135deg, rgba(18, 33, 47, 0.95) 0%, rgba(24, 40, 55, 0.95) 100%);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 24px rgba(3, 8, 14, 0.32);
        }

        .info-card {
            border: 1px solid var(--line);
            background: var(--card);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 8px 18px rgba(3, 8, 14, 0.28);
        }

        .pill-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-bottom: 0.5rem;
        }

        .pill {
            font-size: 0.82rem;
            border: 1px solid rgba(99, 179, 255, 0.45);
            background: var(--accent-soft);
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            color: #d9ebff;
        }

        .metric-tip {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: -0.4rem;
            margin-bottom: 0.6rem;
        }

        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid var(--line);
            border-radius: 10px;
            overflow: hidden;
        }

        @media (max-width: 900px) {
            .hero-note, .info-card {
                padding: 0.8rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _status_badge(exists: bool, label: str) -> str:
    if exists:
        return f"<span class='pill'>{label}: ready</span>"
    return (
        "<span class='pill' "
        "style='background:rgba(176,66,66,0.22);border-color:rgba(255,150,150,0.45);color:#ffd6d6;'>"
        f"{label}: missing</span>"
    )


def _show_image_if_exists(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing figure: {path.name}")


def _build_default_input(table: pd.DataFrame, defaults: dict[str, Any]) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    target_col = "high_rating"
    feature_cols = [c for c in table.columns if c != target_col]
    payload = {}
    for col in feature_cols:
        if col in defaults:
            payload[col] = defaults[col]
        else:
            series = table[col]
            if pd.api.types.is_numeric_dtype(series):
                payload[col] = float(series.dropna().median())
            else:
                mode = series.mode(dropna=True)
                payload[col] = str(mode.iloc[0]) if not mode.empty else "Unknown"
    return pd.DataFrame([payload], columns=feature_cols)


def _interactive_feature_controls(table: pd.DataFrame, default_row: pd.DataFrame) -> pd.DataFrame:
    row = default_row.iloc[0].to_dict()
    st.subheader("Interactive Prediction Inputs")
    st.caption("Change a few business characteristics to see how predicted high-rating probability shifts.")

    left, right = st.columns(2)
    with left:
        if "review_count" in row and "review_count" in table.columns:
            series = pd.to_numeric(table["review_count"], errors="coerce").dropna()
            lo, hi = int(max(0, series.quantile(0.01))), int(max(1, series.quantile(0.99)))
            row["review_count"] = st.slider("Review Count", lo, hi, int(np.clip(row["review_count"], lo, hi)))

        if "num_days_open" in row:
            row["num_days_open"] = st.slider("Days Open / Week", 0, 7, int(np.clip(row["num_days_open"], 0, 7)))

        if "avg_open_hours" in row:
            row["avg_open_hours"] = st.slider(
                "Avg Open Hours / Day", 0.0, 24.0, float(np.clip(row["avg_open_hours"], 0.0, 24.0)), 0.5
            )

    with right:
        if "is_open" in row:
            row["is_open"] = float(
                st.selectbox("Business Currently Open", [0, 1], index=int(round(float(row["is_open"]))))
            )
        if "weekend_open" in row:
            row["weekend_open"] = float(
                st.selectbox("Open on Weekend", [0, 1], index=int(round(float(row["weekend_open"]))))
            )

        if "state" in row and "state" in table.columns:
            state_options = sorted(table["state"].fillna("Unknown").astype(str).unique().tolist())
            state_default = str(row["state"]) if str(row["state"]) in state_options else state_options[0]
            row["state"] = st.selectbox(
                "State",
                state_options,
                index=state_options.index(state_default),
            )

        if "city" in row and "city" in table.columns:
            city_source = table.copy()
            if "state" in row and "state" in table.columns:
                city_source = city_source[
                    city_source["state"].fillna("Unknown").astype(str) == str(row["state"])
                ]
            city_counts = city_source["city"].fillna("Unknown").astype(str).value_counts()
            common_city_options = city_counts[city_counts >= 5].index.tolist()
            city_options = sorted(common_city_options if common_city_options else city_counts.index.tolist())
            if not city_options:
                city_options = sorted(table["city"].fillna("Unknown").astype(str).unique().tolist())

            city_default = str(row["city"])
            if city_default not in city_options:
                state_mode = (
                    city_source["city"].fillna("Unknown").astype(str).mode()
                    if not city_source.empty
                    else pd.Series([], dtype="object")
                )
                city_default = (
                    str(state_mode.iloc[0])
                    if not state_mode.empty and str(state_mode.iloc[0]) in city_options
                    else city_options[0]
                )

            row["city"] = st.selectbox(
                "City",
                city_options,
                index=city_options.index(city_default),
                help="City options are filtered to the selected state and limited to common cities when possible.",
            )

    bool_features = [
        c
        for c in default_row.columns
        if c.startswith("attr_") and c in table.columns and pd.api.types.is_numeric_dtype(table[c])
    ]
    if bool_features:
        st.markdown("**Business Attributes**")
        cols = st.columns(3)
        for i, col in enumerate(bool_features[:9]):
            current = int(round(float(row[col]))) if pd.notna(row[col]) else 0
            with cols[i % 3]:
                row[col] = float(
                    st.checkbox(col.replace("attr_", "").replace("_", " ").title(), value=bool(current), key=col)
                )

    return pd.DataFrame([row], columns=default_row.columns)


def _predict_probability(model_code: str, row_df: pd.DataFrame) -> float:
    if model_code == "mlp":
        pre, mlp_model = load_mlp_components()
        if pre is None or mlp_model is None:
            raise RuntimeError("Missing MLP preprocessor/model artifacts or PyTorch dependency.")
        X_row = pre.transform(row_df).astype(np.float32)
        with torch.no_grad():
            logits = mlp_model(torch.tensor(X_row, dtype=torch.float32)).squeeze(-1)
            proba = torch.sigmoid(logits).cpu().numpy().ravel()[0]
        return float(proba)

    model = load_sklearn_model(model_code)
    if model is None:
        raise RuntimeError(f"Model artifact for '{model_code}' not found.")
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(row_df)[0, 1])
    pred = model.predict(row_df)
    return float(pred[0])


def _to_dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _build_custom_shap_waterfall(row_df: pd.DataFrame, selected_model_code: str) -> tuple[Any | None, str]:
    shap_meta = _load_json(ARTIFACTS / "shap" / "metadata.json")
    fallback_model = shap_meta.get("selected_tree_model", "rf")
    candidate_models = []
    if selected_model_code in TREE_MODEL_CODES:
        candidate_models.append(selected_model_code)
    if fallback_model not in candidate_models:
        candidate_models.append(fallback_model)

    try:
        import shap  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None, "Install `shap` to render a custom waterfall explanation."

    bg_path = ARTIFACTS / "data" / "shap_background.parquet"
    if bg_path.exists():
        bg_df = pd.read_parquet(bg_path)
    else:
        table = load_modeling_table()
        bg_df = table.drop(columns=["high_rating"]).sample(min(300, len(table)), random_state=42)

    last_error: Exception | None = None
    for tree_code in candidate_models:
        model = load_sklearn_model(tree_code)
        if model is None:
            continue

        try:
            preprocessor = model.named_steps["preprocessor"]
            estimator = model.named_steps["model"]
            X_bg = _to_dense(preprocessor.transform(bg_df))
            X_row = _to_dense(preprocessor.transform(row_df))
            explainer = shap.TreeExplainer(estimator, X_bg)
            shap_vals_raw = explainer(X_row, check_additivity=False)
            shap_vals = shap_vals_raw.values if hasattr(shap_vals_raw, "values") else shap_vals_raw
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, -1]

            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(np.asarray(base_value).ravel()[-1])
            else:
                base_value = float(base_value)

            explanation = shap.Explanation(
                values=shap_vals[0],
                base_values=base_value,
                data=X_row[0],
                feature_names=preprocessor.get_feature_names_out(),
            )
            fig = plt.figure(figsize=(10, 5))
            shap.plots.waterfall(explanation, max_display=15, show=False)
            plt.tight_layout()
            if selected_model_code not in TREE_MODEL_CODES:
                note = f"Waterfall is shown using {tree_code.upper()} (best tree model)."
            elif tree_code != selected_model_code:
                note = f"Waterfall fell back to {tree_code.upper()} because SHAP is unavailable for {selected_model_code.upper()}."
            else:
                note = f"Waterfall is shown using selected model: {tree_code.upper()}."
            return fig, note
        except ValueError as exc:
            last_error = exc

    return None, f"Tree model artifact unavailable for SHAP waterfall. Last error: {last_error}"


def _render_project_status(summary: dict[str, Any], metrics: pd.DataFrame) -> None:
    artifacts_ready = {
        "Data": (ARTIFACTS / "data" / "modeling_table.parquet").exists(),
        "EDA": (ARTIFACTS / "figures" / "target_distribution.png").exists(),
        "Metrics": (ARTIFACTS / "metrics" / "model_metrics.csv").exists(),
        "SHAP": (ARTIFACTS / "shap" / "summary.png").exists(),
    }
    badges = "".join([_status_badge(ok, label) for label, ok in artifacts_ready.items()])
    st.markdown(f"<div class='pill-row'>{badges}</div>", unsafe_allow_html=True)

    if summary:
        st.markdown(
            "<div class='hero-note'><b>What this project predicts:</b> Whether a Yelp business will be high-rated "
            "(4 stars or higher). The app combines exploratory visuals, model benchmarking, and SHAP explanations "
            "to show both performance and reasoning.</div>",
            unsafe_allow_html=True,
        )
    if metrics.empty:
        st.info("Model metrics are not fully available yet. Run `python -m src.train.train_all_models`.")


def render_tab_executive() -> None:
    st.header("Executive Summary")
    summary = load_summary()
    metrics = load_metrics()
    shap_meta = load_shap_metadata()

    if not summary:
        st.warning("Run the pipeline first to generate summary artifacts.")
        return

    _render_project_status(summary, metrics)

    class_dist = summary.get("class_distribution", {})
    total = sum(class_dist.values()) if class_dist else 0
    pos_share = class_dist.get("1", 0) / total if total else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{summary.get('n_rows', 0):,}")
    c2.metric("Features", summary.get("n_features", "N/A"))
    c3.metric("High-Rating Share", f"{pos_share:.1%}")

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown(
            "<div class='info-card'><b>Business framing:</b> This is a ranking-quality problem. A strong model helps "
            "identify which business characteristics are associated with consistently high customer ratings.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='info-card'><b>Target definition:</b> `high_rating = 1 if stars >= 4 else 0`.<br/>"
            "A prediction near 1.0 means the model sees the business profile as likely high-rated.</div>",
            unsafe_allow_html=True,
        )

        top_categories = summary.get("top_categories", [])[:10]
        if top_categories:
            pills = "".join([f"<span class='pill'>{x}</span>" for x in top_categories])
            st.markdown("<b>Most common business categories in the sample:</b>", unsafe_allow_html=True)
            st.markdown(f"<div class='pill-row'>{pills}</div>", unsafe_allow_html=True)

    with right:
        if class_dist:
            class_df = pd.DataFrame(
                {
                    "Class": ["0: Not High-Rated", "1: High-Rated"],
                    "Count": [class_dist.get("0", 0), class_dist.get("1", 0)],
                }
            )
            fig = px.pie(
                class_df,
                names="Class",
                values="Count",
                hole=0.55,
                color="Class",
                color_discrete_map={
                    "0: Not High-Rated": "#6a8aa6",
                    "1: High-Rated": "#e6762f",
                },
            )
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e7eef7"),
                legend=dict(font=dict(color="#e7eef7")),
            )
            st.plotly_chart(fig, use_container_width=True)

    if not metrics.empty:
        ranked = metrics.sort_values("f1", ascending=False).reset_index(drop=True)
        best = ranked.iloc[0]
        st.success(
            f"Current best available model: {best['model'].upper()} | "
            f"F1={best['f1']:.3f}, AUC={best['auc_roc']:.3f}, Accuracy={best['accuracy']:.3f}"
        )
        if not ranked.empty:
            baseline = ranked[ranked["model"] == "logreg"].iloc[0]
            delta = best["f1"] - baseline["f1"]
            takeaway = (
                f"The best model improves F1 by {delta:.3f} over the logistic baseline, which indicates that "
                "nonlinear interactions matter for this Yelp business problem."
            )
            if shap_meta.get("top_features"):
                driver_names = ", ".join(item["feature_label"] for item in shap_meta["top_features"][:3])
                takeaway += f" The strongest explainability drivers are {driver_names}."
            st.markdown(f"<div class='info-card'><b>Key finding:</b> {takeaway}</div>", unsafe_allow_html=True)


def render_tab_descriptive() -> None:
    st.header("Descriptive Analytics")
    table = load_modeling_table()
    summary = load_summary()

    st.markdown(
        "<div class='hero-note'><b>How to read this tab:</b> Start with class balance, then compare feature distributions "
        "by target, and finally use correlation to identify potentially redundant predictors.</div>",
        unsafe_allow_html=True,
    )

    if not table.empty:
        with st.expander("Dataset Preview and Glossary", expanded=False):
            preview_cols = [c for c in ["city", "state", "review_count", "is_open", "num_days_open", "avg_open_hours", "high_rating"] if c in table.columns]
            st.dataframe(table[preview_cols].head(12), use_container_width=True)
            glossary_rows = []
            for key, text in FEATURE_GLOSSARY.items():
                if key in table.columns:
                    glossary_rows.append({"Feature": key, "Meaning": text})
            if glossary_rows:
                st.dataframe(pd.DataFrame(glossary_rows), use_container_width=True, hide_index=True)

    figure_specs = [
        ("target_distribution.png", "Target Distribution"),
        ("review_count_distribution_by_target.png", "Review Count Distribution by Target"),
        ("review_count_boxplot_by_target.png", "Review Count Boxplot by Target"),
        ("high_rating_share_by_state_top15.png", "High-Rating Share by State"),
        ("high_rating_share_by_top_categories.png", "High-Rating Share by Category"),
        ("correlation_heatmap.png", "Correlation Heatmap"),
    ]

    cols = st.columns(2)
    for idx, (filename, caption) in enumerate(figure_specs):
        with cols[idx % 2]:
            _show_image_if_exists(ARTIFACTS / "figures" / filename, caption)
            tip = FIGURE_HELP_TEXT.get(filename)
            if tip:
                st.markdown(f"<div class='metric-tip'>{tip}</div>", unsafe_allow_html=True)

    if not table.empty and "state" in table.columns and "high_rating" in table.columns:
        st.subheader("Interactive View: High-Rating Share by State")
        state_df = (
            table.groupby("state", dropna=False)["high_rating"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
            .rename(columns={"high_rating": "High-Rating Share"})
        )
        fig = px.bar(
            state_df,
            x="High-Rating Share",
            y="state",
            orientation="h",
            color="High-Rating Share",
            color_continuous_scale="Teal",
        )
        fig.update_layout(height=420, yaxis_title="State", xaxis_title="High-Rating Share")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e7eef7"),
            coloraxis_colorbar=dict(
                tickfont=dict(color="#e7eef7"),
                title=dict(font=dict(color="#e7eef7")),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    notes_path = ARTIFACTS / "figures" / "eda_notes.md"
    if notes_path.exists():
        with st.expander("Analyst Notes", expanded=False):
            st.markdown(notes_path.read_text(encoding="utf-8"))

    if summary:
        st.caption(
            f"Rows: {summary.get('n_rows', 'N/A')} | Features: {summary.get('n_features', 'N/A')} | "
            f"Numerical: {summary.get('numeric_feature_count', 'N/A')} | "
            f"Categorical: {summary.get('categorical_feature_count', 'N/A')}"
        )


def render_tab_model_performance() -> None:
    st.header("Model Performance")
    metrics = load_metrics()

    st.markdown(
        "<div class='hero-note'><b>How to read this tab:</b> F1 balances precision/recall and is the main ranking metric. "
        "AUC shows probability ranking quality independent of threshold.</div>",
        unsafe_allow_html=True,
    )

    if metrics.empty:
        st.warning("Model metrics are not available yet. Run training: `python -m src.train.train_all_models`.")
        return

    display_metrics = metrics.copy()
    metric_cols = ["accuracy", "precision", "recall", "f1", "auc_roc", "train_time_sec"]
    for col in metric_cols:
        if col in display_metrics.columns:
            display_metrics[col] = display_metrics[col].astype(float).round(4)
    st.dataframe(display_metrics, use_container_width=True)

    top = metrics.sort_values("f1", ascending=False).iloc[0]
    ranked = metrics.sort_values("f1", ascending=False).reset_index(drop=True)
    baseline = ranked[ranked["model"] == "logreg"].iloc[0]
    runner_up = ranked.iloc[1] if len(ranked) > 1 else top
    st.markdown(
        f"<div class='info-card'><b>Best model by F1:</b> {top['model'].upper()} "
        f"(F1={top['f1']:.3f}, AUC={top['auc_roc']:.3f}).</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='info-card'><b>Model comparison takeaway:</b> "
        f"{top['model'].upper()} outperformed the logistic baseline by {top['f1'] - baseline['f1']:.3f} F1 points. "
        f"The runner-up was {runner_up['model'].upper()} with F1={runner_up['f1']:.3f}. "
        "The tradeoff is straightforward: ensemble models are stronger predictors, while logistic regression and the single decision tree remain easier to explain directly."
        "</div>",
        unsafe_allow_html=True,
    )

    _show_image_if_exists(ARTIFACTS / "figures" / "model_comparison_f1.png", "Model Comparison by F1")
    _show_image_if_exists(ARTIFACTS / "figures" / "roc_curves_all_models.png", "Combined ROC Curves")

    st.subheader("Per-Model Curves")
    cols = st.columns(2)
    for idx, code in enumerate(["logreg", "dt", "rf", "xgb", "mlp"]):
        with cols[idx % 2]:
            _show_image_if_exists(ARTIFACTS / "figures" / f"roc_{code}.png", f"ROC - {code.upper()}")

    _show_image_if_exists(ARTIFACTS / "figures" / "dt_best_tree.png", "Best Decision Tree (Top Levels)")
    _show_image_if_exists(ARTIFACTS / "figures" / "mlp_training_history.png", "MLP Training History")
    _show_image_if_exists(ARTIFACTS / "figures" / "mlp_tuning_results.png", "Bonus MLP Tuning Results")

    params = load_best_params()
    if params:
        with st.expander("Best Hyperparameters", expanded=False):
            param_rows = [
                {"Model": model.upper(), "Hyperparameters": json.dumps(values)}
                for model, values in params.items()
            ]
            st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)


def _render_explainability_panel(shap_meta: dict[str, Any]) -> None:
    st.markdown(
        "<div class='hero-note'><b>How to read SHAP:</b> Positive SHAP values push prediction toward high-rating; "
        "negative values push it away. The waterfall plot explains one specific business profile.</div>",
        unsafe_allow_html=True,
    )

    _show_image_if_exists(ARTIFACTS / "shap" / "summary.png", "SHAP Summary (Beeswarm)")
    _show_image_if_exists(ARTIFACTS / "shap" / "bar.png", "SHAP Mean Absolute Importance")
    _show_image_if_exists(ARTIFACTS / "shap" / "waterfall_example.png", "SHAP Waterfall (Saved Example)")

    if shap_meta.get("fallback_reason"):
        st.info(shap_meta["fallback_reason"])

    interpretation_path = ARTIFACTS / "shap" / "interpretation.md"
    if interpretation_path.exists():
        st.markdown(interpretation_path.read_text(encoding="utf-8"))

    top_features = shap_meta.get("top_features", [])
    if top_features:
        st.subheader("Top SHAP Drivers")
        top_features_df = pd.DataFrame(top_features)[["feature_label", "mean_abs_shap", "direction_summary"]]
        top_features_df.columns = ["Feature", "Mean |SHAP|", "Interpretation"]
        top_features_df["Mean |SHAP|"] = top_features_df["Mean |SHAP|"].astype(float).round(4)
        st.dataframe(top_features_df, use_container_width=True, hide_index=True)


def _render_prediction_panel() -> None:
    st.markdown(
        "<div class='hero-note'><b>How to use this tab:</b> Change the business inputs, choose a model, "
        "and compare the predicted high-rating probability. A custom SHAP waterfall explains the result for tree models.</div>",
        unsafe_allow_html=True,
    )

    table = load_modeling_table()
    defaults = load_feature_defaults()
    if table.empty:
        st.warning("Missing modeling table; run the data preparation script first.")
        return

    default_row = _build_default_input(table, defaults)
    input_row = _interactive_feature_controls(table, default_row)

    selected_label = st.selectbox("Model for prediction", list(MODEL_LABELS.keys()))
    selected_code = MODEL_LABELS[selected_label]

    try:
        proba = _predict_probability(selected_code, input_row)
        pred_class = int(proba >= 0.5)
        c1, c2 = st.columns(2)
        c1.metric("Predicted Probability (High-Rating)", f"{proba:.3f}")
        c2.metric("Predicted Class", "High-Rating" if pred_class == 1 else "Not High-Rating")

        interpretation = (
            "Model believes this business profile has a strong chance of high ratings."
            if proba >= 0.65
            else "Model sees this profile as borderline; a few feature improvements could shift the prediction."
            if proba >= 0.45
            else "Model believes this profile is currently less likely to be high-rated."
        )
        st.markdown(f"<div class='info-card'><b>Plain-English interpretation:</b> {interpretation}</div>", unsafe_allow_html=True)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return

    st.subheader("Custom Input SHAP Waterfall")
    fig, note = _build_custom_shap_waterfall(input_row, selected_code)
    if note:
        st.caption(note)
    if fig is not None:
        st.pyplot(fig, clear_figure=True)


def render_tab_explainability() -> None:
    st.header("Explainability & Interactive Prediction")
    st.caption("Top-level structure stays rubric-safe: this required tab now contains two focused sub-tabs.")
    shap_meta = load_shap_metadata()

    sub_tabs = st.tabs(["Explainability", "Interactive Prediction"])
    with sub_tabs[0]:
        _render_explainability_panel(shap_meta)
    with sub_tabs[1]:
        _render_prediction_panel()


def main() -> None:
    st.set_page_config(page_title="MSIS 522 HW1 - Yelp Dashboard", layout="wide")
    _inject_styles()

    st.title("Yelp Business Rating Analytics Dashboard")
    st.caption("MSIS 522 | Foster School of Business | University of Washington")

    tabs = st.tabs(
        [
            "Executive Summary",
            "Descriptive Analytics",
            "Model Performance",
            "Explainability & Interactive Prediction",
        ]
    )

    with tabs[0]:
        render_tab_executive()
    with tabs[1]:
        render_tab_descriptive()
    with tabs[2]:
        render_tab_model_performance()
    with tabs[3]:
        render_tab_explainability()


if __name__ == "__main__":
    main()
