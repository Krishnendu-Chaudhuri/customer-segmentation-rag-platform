"""Unit tests for ETL schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shopper_segmentation.schema import SchemaValidationError, validate_csv_schema


def _write_minimal_fixtures(data_dir: Path) -> None:
    """Write valid minimal CSV fixtures for schema validation."""
    pd.DataFrame(
        {
            "household_key": [1],
            "BASKET_ID": [1],
            "DAY": [1],
            "PRODUCT_ID": [101],
            "SALES_VALUE": [10.0],
            "WEEK_NO": [1],
            "STORE_ID": [1],
        }
    ).to_csv(data_dir / "transaction_data.csv", index=False)

    pd.DataFrame(
        {"PRODUCT_ID": [101], "DEPARTMENT": ["GROCERY"], "BRAND": ["BrandA"]}
    ).to_csv(data_dir / "product.csv", index=False)

    pd.DataFrame(
        {
            "household_key": [1],
            "AGE_DESC": ["35-44"],
            "MARITAL_STATUS_CODE": ["A"],
            "INCOME_DESC": ["Mid"],
            "HOMEOWNER_DESC": ["Owner"],
            "HH_COMP_DESC": ["Adults"],
            "HOUSEHOLD_SIZE_DESC": ["2"],
            "KID_CATEGORY_DESC": ["None"],
        }
    ).to_csv(data_dir / "hh_demographic.csv", index=False)

    pd.DataFrame({"household_key": [1], "CAMPAIGN": [1]}).to_csv(
        data_dir / "campaign_table.csv", index=False
    )
    pd.DataFrame(
        {
            "CAMPAIGN": [1],
            "DESCRIPTION": ["Promo"],
            "START_DAY": [1],
            "END_DAY": [10],
        }
    ).to_csv(data_dir / "campaign_desc.csv", index=False)
    pd.DataFrame({"COUPON_UPC": [1001], "DESCRIPTION": ["Coupon"]}).to_csv(
        data_dir / "coupon.csv", index=False
    )
    pd.DataFrame({"household_key": [1], "COUPON_UPC": [1001], "DAY": [2]}).to_csv(
        data_dir / "coupon_redempt.csv", index=False
    )
    pd.DataFrame(
        {
            "PRODUCT_ID": [101],
            "STORE_ID": [1],
            "WEEK_NO": [1],
            "DISPLAY": [1],
            "MAILER": [0],
        }
    ).to_csv(data_dir / "causal_data.csv", index=False)


def test_validate_csv_schema_passes_with_valid_fixtures(tmp_path: Path) -> None:
    """Valid fixtures should pass schema validation."""
    _write_minimal_fixtures(tmp_path)
    validate_csv_schema(tmp_path)


def test_validate_csv_schema_raises_on_missing_columns(tmp_path: Path) -> None:
    """Malformed fixtures should raise SchemaValidationError with file details."""
    _write_minimal_fixtures(tmp_path)
    bad_txn = pd.read_csv(tmp_path / "transaction_data.csv").drop(columns=["WEEK_NO"])
    bad_txn.to_csv(tmp_path / "transaction_data.csv", index=False)

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_csv_schema(tmp_path)

    message = str(exc_info.value)
    assert "transaction_data.csv" in message
    assert "WEEK_NO" in message
