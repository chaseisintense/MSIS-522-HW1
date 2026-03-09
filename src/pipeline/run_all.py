from __future__ import annotations

import argparse

from src.data.download_yelp import download_dataset
from src.data.prepare_business_table import prepare_business_table
from src.explain.run_shap import run_shap_analysis
from src.train.train_all_models import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full HW1 pipeline.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument("--skip-download", action="store_true", help="Skip Kaggle download step.")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP explainability step.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_download:
        download_dataset(config_path=args.config)
    prepare_business_table(config_path=args.config)
    if not args.skip_training:
        run_training(config_path=args.config)
    if not args.skip_shap and not args.skip_training:
        run_shap_analysis(config_path=args.config)


if __name__ == "__main__":
    main()

