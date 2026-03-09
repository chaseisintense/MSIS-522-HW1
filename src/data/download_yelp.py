from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.common import load_config, resolve_path


DATASET_HANDLE = "yelp-dataset/yelp-dataset"
BUSINESS_FILENAME = "yelp_academic_dataset_business.json"


def _find_business_file(search_root: Path) -> Path | None:
    matches = list(search_root.rglob(BUSINESS_FILENAME))
    return matches[0] if matches else None


def _manual_fallback_message(target_path: Path) -> str:
    return (
        "Could not download via KaggleHub.\n"
        "Manual fallback:\n"
        "1) Download the Yelp dataset from Kaggle.\n"
        f"2) Place {BUSINESS_FILENAME} at: {target_path}\n"
        "3) Re-run: python -m src.data.prepare_business_table\n"
    )


def download_dataset(config_path: str = "configs/config.yaml") -> Path:
    config = load_config(config_path)
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_path = raw_dir / BUSINESS_FILENAME

    if target_path.exists():
        print(f"Dataset file already exists: {target_path}")
        return target_path

    try:
        import kagglehub  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(_manual_fallback_message(target_path)) from exc

    try:
        downloaded_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(_manual_fallback_message(target_path)) from exc

    source_file = _find_business_file(downloaded_path)
    if source_file is None:
        raise FileNotFoundError(
            f"{BUSINESS_FILENAME} was not found under downloaded path: {downloaded_path}"
        )

    shutil.copy2(source_file, target_path)
    print(f"Copied {BUSINESS_FILENAME} to: {target_path}")
    return target_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Yelp dataset using KaggleHub.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to project config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = download_dataset(config_path=args.config)
    print(f"Path to dataset file: {path}")


if __name__ == "__main__":
    main()

