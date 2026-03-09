from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.common import ensure_parent_dir, load_config, resolve_path, save_json

sns.set_theme(style="whitegrid")


def _sanitize_column_name(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", value.strip().lower()).strip("_")


def _clean_text_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("Unknown")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", "Unknown")
    )


def _normalize_city_series(series: pd.Series) -> pd.Series:
    cleaned = _clean_text_series(series)
    canonical = cleaned.str.casefold()
    canonical_map = (
        pd.DataFrame({"canonical": canonical, "display": cleaned})
        .groupby("canonical")["display"]
        .agg(lambda values: values.value_counts().index[0])
        .to_dict()
    )
    return canonical.map(canonical_map)


def _parse_boolish(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return 1.0
    if text in {"false", "no", "0"}:
        return 0.0
    return np.nan


def _hours_to_duration(hours_text: str) -> float:
    """
    Convert strings like '10:0-23:30' into open duration hours.
    """
    try:
        start, end = hours_text.split("-")
        sh, sm = [int(x) for x in start.split(":")]
        eh, em = [int(x) for x in end.split(":")]
        start_minutes = sh * 60 + sm
        end_minutes = eh * 60 + em
        if end_minutes < start_minutes:
            end_minutes += 24 * 60
        return float(max(end_minutes - start_minutes, 0)) / 60.0
    except Exception:  # noqa: BLE001
        return np.nan


def _derive_hours_features(hours_obj: Any) -> tuple[float, float, float]:
    if not isinstance(hours_obj, dict) or not hours_obj:
        return np.nan, np.nan, np.nan

    days = list(hours_obj.keys())
    durations = [_hours_to_duration(str(v)) for v in hours_obj.values()]
    durations = [d for d in durations if not np.isnan(d)]
    avg_hours = float(np.mean(durations)) if durations else np.nan
    weekend_open = float(any(day in {"Saturday", "Sunday"} for day in days))
    return float(len(days)), avg_hours, weekend_open


def _extract_attributes(df: pd.DataFrame, attribute_names: list[str]) -> pd.DataFrame:
    attrs = df["attributes"].apply(lambda x: x if isinstance(x, dict) else {})
    for name in attribute_names:
        col = f"attr_{_sanitize_column_name(name)}"
        df[col] = attrs.apply(lambda d: _parse_boolish(d.get(name)))
    return df


def _extract_category_features(df: pd.DataFrame, top_k: int) -> tuple[pd.DataFrame, list[str]]:
    category_lists = (
        df["categories"]
        .fillna("")
        .astype(str)
        .apply(lambda x: [c.strip() for c in x.split(",") if c.strip()])
    )
    exploded = category_lists.explode()
    top_categories = exploded.value_counts().head(top_k).index.tolist()

    created_columns: list[str] = []
    for category in top_categories:
        col = f"cat_{_sanitize_column_name(category)}"
        df[col] = category_lists.apply(lambda lst: float(category in lst))
        created_columns.append(col)

    return df, top_categories


def _plot_target_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    counts = df["high_rating"].value_counts().sort_index()
    counts_df = pd.DataFrame(
        {
            "class_label": counts.index.map({0: "Not High-Rated", 1: "High-Rated"}),
            "count": counts.values,
        }
    )
    sns.barplot(
        data=counts_df,
        x="class_label",
        y="count",
        hue="class_label",
        palette="Blues_d",
        dodge=False,
        legend=False,
    )
    plt.title("Target Distribution: High Rating (stars >= 4)")
    plt.xlabel("High Rating Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png", dpi=160)
    plt.close()


def _plot_review_count_hist(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = df[["review_count", "high_rating"]].dropna().copy()
    plot_df["target_label"] = plot_df["high_rating"].map({0: "Not High-Rated", 1: "High-Rated"})
    plot_df["review_count_log1p"] = np.log10(plot_df["review_count"] + 1.0)

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=plot_df,
        x="review_count_log1p",
        hue="target_label",
        bins=45,
        kde=True,
        stat="density",
        common_norm=False,
        element="step",
    )
    tick_labels = np.array([5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000])
    plt.xticks(np.log10(tick_labels + 1.0), tick_labels)
    plt.title("Review Count Distribution by Target (Log Scale)")
    plt.xlabel("Review Count")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(output_dir / "review_count_distribution_by_target.png", dpi=160)
    plt.close()


def _plot_review_count_box(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = df[["review_count", "high_rating"]].dropna().copy()
    plot_df["target_label"] = plot_df["high_rating"].map({0: "Not High-Rated", 1: "High-Rated"})
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=plot_df, x="target_label", y="review_count", showfliers=False)
    plt.yscale("log")
    plt.title("Review Count by Target Class (Log Scale)")
    plt.xlabel("Target Class")
    plt.ylabel("Review Count")
    plt.tight_layout()
    plt.savefig(output_dir / "review_count_boxplot_by_target.png", dpi=160)
    plt.close()


def _plot_state_target_rate(df: pd.DataFrame, output_dir: Path) -> None:
    state_rate = (
        df.groupby("state", dropna=False)["high_rating"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=state_rate,
        x="high_rating",
        y="state",
        hue="state",
        palette="viridis",
        dodge=False,
        legend=False,
    )
    plt.title("Top States by High-Rating Share")
    plt.xlabel("High-Rating Share")
    plt.ylabel("State")
    plt.tight_layout()
    plt.savefig(output_dir / "high_rating_share_by_state_top15.png", dpi=160)
    plt.close()


def _plot_top_category_rate(df: pd.DataFrame, output_dir: Path) -> None:
    cat_cols = [c for c in df.columns if c.startswith("cat_")]
    if not cat_cols:
        return
    rows = []
    for col in cat_cols:
        subset = df[df[col] == 1.0]
        if len(subset) < 200:
            continue
        rows.append({"category": col.replace("cat_", ""), "high_rating_share": subset["high_rating"].mean()})
    if not rows:
        return
    cat_rates = (
        pd.DataFrame(rows)
        .sort_values("high_rating_share", ascending=False)
        .head(12)
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=cat_rates,
        x="high_rating_share",
        y="category",
        hue="category",
        palette="mako",
        dodge=False,
        legend=False,
    )
    plt.title("Top Category Indicators by High-Rating Share")
    plt.xlabel("High-Rating Share")
    plt.ylabel("Category Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "high_rating_share_by_top_categories.png", dpi=160)
    plt.close()


def _plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if "high_rating" not in numeric_df.columns:
        return
    corr = numeric_df.corr(numeric_only=True)
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Correlation Heatmap (Numerical Features)")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close()


def _save_dataset_summary(
    df: pd.DataFrame,
    config: dict[str, Any],
    top_categories: list[str],
    output_dir: Path,
) -> None:
    target_col = config["problem"]["target_column"]
    summary = {
        "dataset_name": "Yelp Academic Dataset (business table)",
        "target_column": target_col,
        "target_definition": f"{target_col} = 1 if stars >= {config['problem']['high_rating_threshold']}",
        "n_rows": int(df.shape[0]),
        "n_features": int(df.drop(columns=[target_col]).shape[1]),
        "numeric_feature_count": int(df.drop(columns=[target_col]).select_dtypes(include=["number"]).shape[1]),
        "categorical_feature_count": int(
            df.drop(columns=[target_col]).select_dtypes(exclude=["number"]).shape[1]
        ),
        "class_distribution": {
            str(int(k)): int(v)
            for k, v in df[target_col].value_counts().sort_index().to_dict().items()
        },
        "top_categories": top_categories,
        "figures": [
            "target_distribution.png",
            "review_count_distribution_by_target.png",
            "review_count_boxplot_by_target.png",
            "high_rating_share_by_state_top15.png",
            "high_rating_share_by_top_categories.png",
            "correlation_heatmap.png",
        ],
    }
    save_json(summary, config["paths"]["dataset_summary"])

    feature_df = df.drop(columns=[target_col]).copy()
    defaults: dict[str, Any] = {}
    for col in feature_df.columns:
        series = feature_df[col]
        if pd.api.types.is_numeric_dtype(series):
            defaults[col] = float(series.fillna(series.median()).median())
        else:
            mode = series.mode(dropna=True)
            defaults[col] = str(mode.iloc[0]) if not mode.empty else "Unknown"
    save_json(defaults, config["paths"]["feature_defaults"])

    ensure_parent_dir(output_dir / "eda_notes.md")
    with (output_dir / "eda_notes.md").open("w", encoding="utf-8") as f:
        f.write(
            "# EDA Notes\n\n"
            "- Target distribution indicates whether class imbalance handling is needed.\n"
            "- Review count visuals show highly skewed engagement patterns across businesses.\n"
            "- Geographic plot highlights where high-rated businesses are concentrated.\n"
            "- Category-level rates reveal business-type differences tied to higher ratings.\n"
            "- Correlation heatmap informs potential multicollinearity before modeling.\n"
        )


def prepare_business_table(config_path: str = "configs/config.yaml") -> Path:
    config = load_config(config_path)
    paths = config["paths"]
    feature_cfg = config["features"]
    problem_cfg = config["problem"]
    random_state = config["project"]["random_state"]

    raw_business_file = resolve_path(paths["raw_dir"]) / paths["business_filename"]
    if not raw_business_file.exists():
        raise FileNotFoundError(
            f"Missing input file: {raw_business_file}. Run python -m src.data.download_yelp first "
            "or place the file manually."
        )

    df = pd.read_json(raw_business_file, lines=True)
    df = df[feature_cfg["keep_columns"]].copy()

    # Normalize key text fields and keep cardinality manageable.
    df["city"] = _normalize_city_series(df["city"])
    df["state"] = _clean_text_series(df["state"]).str.upper()

    df["is_open"] = pd.to_numeric(df["is_open"], errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")

    df = _extract_attributes(df, feature_cfg["attribute_booleans"])

    hours_df = df["hours"].apply(_derive_hours_features).apply(pd.Series)
    hours_df.columns = ["num_days_open", "avg_open_hours", "weekend_open"]
    df = pd.concat([df, hours_df], axis=1)

    df, top_categories = _extract_category_features(df, feature_cfg["top_categories_k"])

    df[problem_cfg["target_column"]] = (df["stars"] >= problem_cfg["high_rating_threshold"]).astype(int)

    # Remove direct leakage columns before writing the modeling table.
    drop_cols = ["stars", "business_id", "categories", "attributes", "hours"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    max_rows = int(problem_cfg["max_rows"])
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state)

    model_table_path = resolve_path(paths["modeling_table"])
    ensure_parent_dir(model_table_path)
    df.to_parquet(model_table_path, index=False)

    figure_dir = resolve_path("artifacts/figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_target_distribution(df, figure_dir)
    _plot_review_count_hist(df, figure_dir)
    _plot_review_count_box(df, figure_dir)
    _plot_state_target_rate(df, figure_dir)
    _plot_top_category_rate(df, figure_dir)
    _plot_correlation_heatmap(df, figure_dir)

    _save_dataset_summary(df, config, top_categories, figure_dir)
    print(f"Saved modeling table to {model_table_path}")
    return model_table_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Yelp business modeling table and EDA artifacts.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = prepare_business_table(config_path=args.config)
    print(f"Prepared data at: {output_path}")


if __name__ == "__main__":
    main()
