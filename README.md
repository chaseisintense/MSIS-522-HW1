# MSIS 522 Homework 1: Yelp High-Rating Classification

End-to-end data science workflow for MSIS 522 (Foster School of Business, UW), including:
- descriptive analytics and visual storytelling,
- predictive modeling (Logistic, Decision Tree, Random Forest, XGBoost, PyTorch MLP),
- SHAP explainability,
- deployed-ready Streamlit app with interactive predictions.

## Repository Structure

```text
MSIS522/
├── app/
│   └── streamlit_app.py
├── configs/
│   └── config.yaml
├── notebooks/
│   └── msis522_hw1_yelp_workflow.ipynb
├── src/
│   ├── common.py
│   ├── data/
│   │   ├── download_yelp.py
│   │   └── prepare_business_table.py
│   ├── train/
│   │   └── train_all_models.py
│   ├── explain/
│   │   └── run_shap.py
│   └── pipeline/
│       └── run_all.py
├── artifacts/
│   ├── data/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── shap/
├── data/
│   └── raw/  # ignored in git
├── requirements.txt
└── README.md
```

## Problem Setup

- **Target**: `high_rating = 1 if stars >= 4 else 0`
- **Data source**: Yelp Academic Dataset (`yelp_academic_dataset_business.json`)
- **Task type**: Binary classification
- **Random seed**: `42`

## Quickstart

### 1) Create environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Run the full pipeline

```bash
python -m src.pipeline.run_all
```

This performs:
1. Dataset download via KaggleHub
2. Feature preparation + EDA figure generation
3. Model training/tuning + metrics + saved artifacts
4. SHAP explainability outputs

### 3) Launch Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## Individual CLI Entrypoints

```bash
python -m src.data.download_yelp
python -m src.data.prepare_business_table
python -m src.train.train_all_models
python -m src.explain.run_shap
python -m src.pipeline.run_all --skip-download
python -m src.pipeline.run_all --skip-training
```

## KaggleHub Authentication Notes

The downloader uses:

```python
kagglehub.dataset_download("yelp-dataset/yelp-dataset")
```

If Kaggle credentials are not configured, place this file manually:

```text
data/raw/yelp/yelp_academic_dataset_business.json
```

Then run:

```bash
python -m src.data.prepare_business_table
python -m src.train.train_all_models
python -m src.explain.run_shap
```

## Saved Artifact Contract

The pipeline writes the following:

- `artifacts/data/modeling_table.parquet`
- `artifacts/data/dataset_summary.json`
- `artifacts/data/feature_defaults.json`
- `artifacts/data/shap_background.parquet`
- `artifacts/models/{logreg,dt,rf,xgb}.joblib`
- `artifacts/models/mlp_preprocessor.joblib`
- `artifacts/models/mlp.pt`
- `artifacts/metrics/model_metrics.csv`
- `artifacts/metrics/best_params.json`
- `artifacts/metrics/mlp_tuning_results.csv` (bonus)
- `artifacts/figures/*.png`
- `artifacts/shap/{summary.png,bar.png,waterfall_example.png,interpretation.md,metadata.json}`

## Streamlit App Tabs

The app includes required tabs:
1. Executive Summary
2. Descriptive Analytics
3. Model Performance
4. Explainability & Interactive Prediction

## Current Local Results

Latest verified local artifacts show:

- `xgb`: Accuracy `0.684`, F1 `0.683`, ROC-AUC `0.753`
- `mlp` (PyTorch): Accuracy `0.672`, F1 `0.665`, ROC-AUC `0.738`
- `rf`: Accuracy `0.650`, F1 `0.652`, ROC-AUC `0.714`
- `dt`: Accuracy `0.646`, F1 `0.645`, ROC-AUC `0.702`
- `logreg`: Accuracy `0.641`, F1 `0.641`, ROC-AUC `0.696`

Interpretation:
- XGBoost is the strongest predictive model on the held-out test set.
- The PyTorch MLP is competitive but does not beat the best tree ensemble.
- Simpler models remain useful as interpretability baselines.

## SHAP Note

- The predictive winner is XGBoost.
- In this environment, SHAP fails on the saved XGBoost artifact with a parsing error.
- The explainability artifacts therefore fall back to the Random Forest model, which still satisfies the assignment requirement of using a tree-based model for SHAP analysis.

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Ensure `requirements.txt` is in the repo root.
3. In Streamlit Community Cloud, create a new app from the GitHub repo.
4. Set main file path to `app/streamlit_app.py`.
5. Deploy and verify public access in an incognito browser window.

## Submission Checklist

- [ ] GitHub repo includes code, notebook, saved models, figures, and requirements.
- [ ] Streamlit app publicly accessible via cloud URL (not localhost).
- [ ] Model comparison and SHAP outputs visible in app.
- [ ] README contains reproducible run instructions.
# MSIS-522-HW1
