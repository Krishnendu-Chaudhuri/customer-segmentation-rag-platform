"""Unit tests for feature engineering helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from shopper_segmentation.features import (
    compute_price_sensitivity,
    compute_promo_responsiveness,
    compute_rfm,
    encode_demographics,
    pivot_department_mix,
    sanitize_column_name,
)


def test_compute_rfm_basic() -> None:
    """RFM metrics should match hand-calculated values."""
    raw = pd.DataFrame(
        {
            "household_key": [1, 2],
            "total_spend": [100.0, 50.0],
            "n_trips": [10, 5],
            "tenure_weeks": [20, 10],
        }
    )
    last_trip_week = pd.Series({1: 95, 2: 90})
    rfm = compute_rfm(raw, last_trip_week, max_week=100)

    assert rfm.loc[rfm["household_key"] == 1, "recency"].iloc[0] == 5
    assert rfm.loc[rfm["household_key"] == 1, "frequency"].iloc[0] == pytest.approx(0.5)
    assert rfm.loc[rfm["household_key"] == 1, "monetary"].iloc[0] == pytest.approx(10.0)
    assert rfm.loc[rfm["household_key"] == 2, "monetary"].iloc[0] == pytest.approx(10.0)


def test_compute_price_sensitivity() -> None:
    """Price sensitivity should use absolute discount over sales."""
    assert compute_price_sensitivity(total_discount=-12.0, total_sales=100.0) == pytest.approx(0.12)
    assert compute_price_sensitivity(total_discount=0.0, total_sales=0.0) == 0.0


def test_compute_promo_responsiveness() -> None:
    """Promo rate should divide coupons by campaigns and handle zeros."""
    coupon_count = pd.Series({1: 4, 2: 0, 3: 2})
    campaign_count = pd.Series({1: 8, 2: 0, 3: 0})
    rates = compute_promo_responsiveness(coupon_count, campaign_count)

    assert rates[1] == pytest.approx(0.5)
    assert rates[2] == pytest.approx(0.0)
    assert rates[3] == pytest.approx(0.0)


def test_encode_demographics_fills_missing_households() -> None:
    """Households without demographics should get all-zero dummy columns."""
    demographics = pd.DataFrame(
        {
            "household_key": [1],
            "AGE_DESC": ["45-54"],
            "MARITAL_STATUS_CODE": ["A"],
            "INCOME_DESC": ["50-74K"],
            "HOMEOWNER_DESC": ["Homeowner"],
            "HH_COMP_DESC": ["2 Adults No Kids"],
            "HOUSEHOLD_SIZE_DESC": ["2"],
            "KID_CATEGORY_DESC": ["None/Unknown"],
        }
    )
    household_keys = pd.Series([1, 2])
    encoded = encode_demographics(demographics, household_keys)

    assert set(encoded["household_key"]) == {1, 2}
    age_col = "AGE_DESC_45-54"
    assert encoded.loc[encoded["household_key"] == 1, age_col].iloc[0] == 1
    assert encoded.loc[encoded["household_key"] == 2, age_col].iloc[0] == 0


def test_pivot_department_mix() -> None:
    """Department mix should pivot to wide percentage columns."""
    long_df = pd.DataFrame(
        {
            "household_key": [1, 1, 2],
            "total_spend": [100.0, 100.0, 50.0],
            "price_sensitivity": [0.1, 0.1, 0.2],
            "department": ["GROCERY", "DAIRY", None],
            "dept_pct": [0.6, 0.4, None],
        }
    )
    wide = pivot_department_mix(long_df)

    assert "dept_grocery_pct" in wide.columns
    assert "dept_dairy_pct" in wide.columns
    assert wide.loc[wide["household_key"] == 1, "dept_grocery_pct"].iloc[0] == pytest.approx(0.6)
    assert wide.loc[wide["household_key"] == 2, "dept_grocery_pct"].iloc[0] == pytest.approx(0.0)


def test_sanitize_column_name() -> None:
    """Category labels should become safe snake-case suffixes."""
    assert sanitize_column_name("MISC. TRANS.") == "misc_trans"
    assert sanitize_column_name("  ") == "unknown"
