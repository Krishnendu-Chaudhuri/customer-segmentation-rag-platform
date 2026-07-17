"""Input schema validation for dunnhumby CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEMO_COLUMNS = [
    "AGE_DESC",
    "MARITAL_STATUS_CODE",
    "INCOME_DESC",
    "HOMEOWNER_DESC",
    "HH_COMP_DESC",
    "HOUSEHOLD_SIZE_DESC",
    "KID_CATEGORY_DESC",
]

REQUIRED_SCHEMAS: dict[str, list[str]] = {
    "transaction_data.csv": [
        "household_key",
        "BASKET_ID",
        "DAY",
        "PRODUCT_ID",
        "SALES_VALUE",
        "WEEK_NO",
        "STORE_ID",
    ],
    "product.csv": ["PRODUCT_ID", "DEPARTMENT", "BRAND"],
    "hh_demographic.csv": ["household_key", *DEMO_COLUMNS],
    "campaign_table.csv": ["household_key", "CAMPAIGN"],
    "campaign_desc.csv": ["CAMPAIGN", "DESCRIPTION", "START_DAY", "END_DAY"],
    "coupon.csv": ["COUPON_UPC", "DESCRIPTION"],
    "coupon_redempt.csv": ["household_key", "COUPON_UPC", "DAY"],
    "causal_data.csv": ["PRODUCT_ID", "STORE_ID", "WEEK_NO", "DISPLAY", "MAILER"],
}


class SchemaValidationError(ValueError):
    """Raised when a source CSV does not match the expected schema."""


def validate_csv_schema(
    data_dir: Path,
    schemas: dict[str, list[str]] | None = None,
) -> None:
    """Validate required columns exist in each expected CSV file.

    Args:
        data_dir: Directory containing source CSV files.
        schemas: Optional override of filename-to-columns mapping.

    Raises:
        SchemaValidationError: If a file is missing or columns do not match.
        FileNotFoundError: If a required CSV file does not exist.
    """
    expected = schemas or REQUIRED_SCHEMAS
    errors: list[str] = []

    for filename, required_columns in expected.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

        header = pd.read_csv(path, nrows=0).columns.tolist()
        missing = [col for col in required_columns if col not in header]
        if missing:
            errors.append(f"{filename}: missing columns {missing}")

    if errors:
        message = "Schema validation failed:\n" + "\n".join(f"- {err}" for err in errors)
        raise SchemaValidationError(message)
