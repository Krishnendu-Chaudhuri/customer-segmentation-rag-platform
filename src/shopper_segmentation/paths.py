"""Repository-relative path constants for data, output, and bundled resources."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "output"))
SAMPLE_OUTPUT_DIR = PACKAGE_ROOT / "resources" / "sample_output"

PIPELINE_SOURCE_FILES = (
    "transaction_data.csv",
    "product.csv",
    "hh_demographic.csv",
    "campaign_table.csv",
    "campaign_desc.csv",
    "coupon.csv",
    "coupon_redempt.csv",
    "causal_data.csv",
)

REQUIRED_JSON_ARTIFACTS = (
    "segment_profiles.json",
    "segment_recommendations.json",
    "uplift_report.json",
)


def profiles_path(output_dir: Path | None = None) -> Path:
    """Return path to segment profiles JSON."""
    return (output_dir or OUTPUT_DIR) / "segment_profiles.json"


def recommendations_path(output_dir: Path | None = None) -> Path:
    """Return path to segment recommendations JSON."""
    return (output_dir or OUTPUT_DIR) / "segment_recommendations.json"


def uplift_path(output_dir: Path | None = None) -> Path:
    """Return path to uplift report JSON."""
    return (output_dir or OUTPUT_DIR) / "uplift_report.json"


def raw_features_path(output_dir: Path | None = None) -> Path:
    """Return path to raw household features parquet."""
    return (output_dir or OUTPUT_DIR) / "household_features_raw.parquet"


def features_path(output_dir: Path | None = None) -> Path:
    """Return path to engineered household features parquet."""
    return (output_dir or OUTPUT_DIR) / "household_features.parquet"


def segments_path(output_dir: Path | None = None) -> Path:
    """Return path to household segment assignments parquet."""
    return (output_dir or OUTPUT_DIR) / "household_segments.parquet"
