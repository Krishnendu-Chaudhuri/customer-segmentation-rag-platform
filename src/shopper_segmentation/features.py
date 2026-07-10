"""Feature engineering module for household shopper segmentation.

Builds RFM, category mix, price sensitivity, promo responsiveness, and
demographic features per household_key. Outputs household_features.parquet.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

from shopper_segmentation.etl import DATA_DIR, OUTPUT_DIR

RAW_INPUT = OUTPUT_DIR / "household_features_raw.parquet"
DEFAULT_OUTPUT = OUTPUT_DIR / "household_features.parquet"
TOP_DEPARTMENTS = 8

DEMO_COLUMNS = [
    "AGE_DESC",
    "MARITAL_STATUS_CODE",
    "INCOME_DESC",
    "HOMEOWNER_DESC",
    "HH_COMP_DESC",
    "HOUSEHOLD_SIZE_DESC",
    "KID_CATEGORY_DESC",
]


def get_data_paths(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """Return paths to CSV and parquet files used by feature engineering.

    Args:
        data_dir: Directory containing source CSV files.

    Returns:
        Mapping of logical names to file paths.
    """
    return {
        "transaction_data": data_dir / "transaction_data.csv",
        "product": data_dir / "product.csv",
        "hh_demographic": data_dir / "hh_demographic.csv",
        "campaign_table": data_dir / "campaign_table.csv",
        "coupon_redempt": data_dir / "coupon_redempt.csv",
        "causal_data": data_dir / "causal_data.csv",
        "raw_features": RAW_INPUT,
    }


def compute_rfm(
    raw: pd.DataFrame,
    last_trip_week: pd.Series,
    max_week: int,
) -> pd.DataFrame:
    """Compute RFM features from raw household aggregates.

    Args:
        raw: DataFrame with household_key, total_spend, n_trips, tenure_weeks.
        last_trip_week: Series indexed by household_key with last trip WEEK_NO.
        max_week: Global maximum WEEK_NO in the dataset.

    Returns:
        DataFrame with household_key, recency, frequency, monetary columns.
    """
    rfm = raw[["household_key"]].copy()
    rfm["recency"] = rfm["household_key"].map(
        lambda hk: max_week - last_trip_week.get(hk, max_week)
    )
    tenure = raw.set_index("household_key")["tenure_weeks"]
    trips = raw.set_index("household_key")["n_trips"]
    spend = raw.set_index("household_key")["total_spend"]
    rfm["frequency"] = rfm["household_key"].map(
        lambda hk: trips[hk] / tenure[hk] if tenure[hk] > 0 else 0.0
    )
    rfm["monetary"] = rfm["household_key"].map(
        lambda hk: spend[hk] / trips[hk] if trips[hk] > 0 else 0.0
    )
    return rfm


def compute_promo_responsiveness(
    coupon_count: pd.Series,
    campaign_count: pd.Series,
) -> pd.Series:
    """Compute coupon redemption rate per household.

    Args:
        coupon_count: Redemption counts indexed by household_key.
        campaign_count: Campaign exposure counts indexed by household_key.

    Returns:
        Series of promo responsiveness rates in [0, 1].
    """
    aligned = pd.DataFrame({"coupons": coupon_count, "campaigns": campaign_count}).fillna(0)
    rates = aligned["coupons"] / aligned["campaigns"].replace(0, pd.NA)
    return rates.fillna(0.0)


def compute_price_sensitivity(
    total_discount: float,
    total_sales: float,
) -> float:
    """Compute average price sensitivity for a household.

    Args:
        total_discount: Sum of RETAIL_DISC + COUPON_DISC across transactions.
        total_sales: Sum of SALES_VALUE across transactions.

    Returns:
        Ratio of total discount magnitude to total sales, or 0 if sales is 0.
    """
    if total_sales <= 0:
        return 0.0
    return abs(total_discount) / total_sales


def sanitize_column_name(value: str) -> str:
    """Convert a category label into a safe parquet column name.

    Args:
        value: Raw category string.

    Returns:
        Snake-case column suffix safe for parquet/ML pipelines.
    """
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip()).strip("_").lower()
    return cleaned or "unknown"


def encode_demographics(
    demographics: pd.DataFrame,
    household_keys: pd.Series,
) -> pd.DataFrame:
    """One-hot encode demographic columns for all households.

    Households without demographic records receive zeros for all dummy columns.

    Args:
        demographics: Demographic records with household_key column.
        household_keys: All household keys to include in output.

    Returns:
        One-hot encoded demographic features indexed by household_key.
    """
    demo = demographics.drop_duplicates(subset=["household_key"]).set_index("household_key")
    encoded_parts: list[pd.DataFrame] = []

    for column in DEMO_COLUMNS:
        if column not in demo.columns:
            continue
        dummies = pd.get_dummies(demo[column], prefix=column, dtype=int)
        encoded_parts.append(dummies)

    if not encoded_parts:
        return pd.DataFrame(index=household_keys.unique())

    encoded = pd.concat(encoded_parts, axis=1)
    encoded = encoded.reindex(household_keys.unique(), fill_value=0)
    encoded.index.name = "household_key"
    return encoded.reset_index()


def build_feature_query(paths: dict[str, Path], top_departments: int = TOP_DEPARTMENTS) -> str:
    """Build DuckDB SQL for transaction-derived household features.

    Args:
        paths: Mapping of logical names to file paths.
        top_departments: Number of top departments to pivot for category mix.

    Returns:
        SQL query returning one row per household_key with engineered features.
    """
    txn = paths["transaction_data"].as_posix()
    product = paths["product"].as_posix()
    campaign = paths["campaign_table"].as_posix()
    coupon = paths["coupon_redempt"].as_posix()
    causal = paths["causal_data"].as_posix()
    raw = paths["raw_features"].as_posix()

    return f"""
WITH global_bounds AS (
    SELECT MAX(WEEK_NO) AS max_week
    FROM read_csv_auto('{txn}', header=true)
),
txn_enriched AS (
    SELECT
        t.household_key,
        t.PRODUCT_ID,
        t.STORE_ID,
        t.WEEK_NO,
        t.SALES_VALUE,
        t.RETAIL_DISC,
        t.COUPON_DISC,
        COALESCE(p.DEPARTMENT, 'UNKNOWN') AS department
    FROM read_csv_auto('{txn}', header=true) AS t
    LEFT JOIN read_csv_auto('{product}', header=true) AS p
        ON t.PRODUCT_ID = p.PRODUCT_ID
),
last_trip AS (
    SELECT household_key, MAX(WEEK_NO) AS last_trip_week
    FROM txn_enriched
    GROUP BY household_key
),
price_features AS (
    SELECT
        household_key,
        SUM(RETAIL_DISC + COUPON_DISC) AS total_discount,
        SUM(SALES_VALUE) AS txn_sales
    FROM txn_enriched
    GROUP BY household_key
),
dept_spend AS (
    SELECT
        household_key,
        department,
        SUM(SALES_VALUE) AS dept_spend
    FROM txn_enriched
    GROUP BY household_key, department
),
household_spend AS (
    SELECT household_key, SUM(dept_spend) AS total_spend
    FROM dept_spend
    GROUP BY household_key
),
top_departments AS (
    SELECT department
    FROM (
        SELECT department, SUM(dept_spend) AS global_dept_spend
        FROM dept_spend
        GROUP BY department
        ORDER BY global_dept_spend DESC
        LIMIT {top_departments}
    )
),
dept_mix AS (
    SELECT
        d.household_key,
        d.department,
        d.dept_spend / NULLIF(h.total_spend, 0) AS dept_pct
    FROM dept_spend AS d
    JOIN household_spend AS h ON d.household_key = h.household_key
    WHERE d.department IN (SELECT department FROM top_departments)
),
campaign_counts AS (
    SELECT household_key, COUNT(*) AS campaign_count
    FROM read_csv_auto('{campaign}', header=true)
    GROUP BY household_key
),
coupon_counts AS (
    SELECT household_key, COUNT(*) AS coupon_count
    FROM read_csv_auto('{coupon}', header=true)
    GROUP BY household_key
),
txn_causal AS (
    SELECT
        t.household_key,
        CASE
            WHEN TRY_CAST(c.display AS DOUBLE) IS NOT NULL
                 AND TRY_CAST(c.display AS DOUBLE) != 0
            THEN 1.0 ELSE 0.0
        END AS display_hit,
        CASE
            WHEN c.mailer IS NOT NULL
                 AND TRIM(CAST(c.mailer AS VARCHAR)) NOT IN ('0', '')
            THEN 1.0 ELSE 0.0
        END AS mailer_hit
    FROM read_csv_auto('{txn}', header=true) AS t
    INNER JOIN read_csv_auto('{causal}', header=true) AS c
        ON t.PRODUCT_ID = c.PRODUCT_ID
       AND t.STORE_ID = c.STORE_ID
       AND t.WEEK_NO = c.WEEK_NO
),
exposure_features AS (
    SELECT
        household_key,
        AVG(display_hit) AS display_exposure_rate,
        AVG(mailer_hit) AS mailer_exposure_rate
    FROM txn_causal
    GROUP BY household_key
)
SELECT
    r.household_key,
    r.total_spend,
    r.n_baskets,
    r.n_trips,
    r.avg_basket_size,
    r.n_unique_products,
    r.tenure_weeks,
    gb.max_week - lt.last_trip_week AS recency,
    CASE WHEN r.tenure_weeks > 0 THEN r.n_trips * 1.0 / r.tenure_weeks ELSE 0 END AS frequency,
    CASE WHEN r.n_trips > 0 THEN r.total_spend * 1.0 / r.n_trips ELSE 0 END AS monetary,
    CASE
        WHEN pf.txn_sales > 0
        THEN ABS(pf.total_discount) / pf.txn_sales
        ELSE 0
    END AS price_sensitivity,
    COALESCE(cc.coupon_count, 0) AS coupon_redemption_count,
    COALESCE(camp.campaign_count, 0) AS campaign_count,
    CASE
        WHEN COALESCE(camp.campaign_count, 0) > 0
        THEN COALESCE(cc.coupon_count, 0) * 1.0 / camp.campaign_count
        ELSE 0
    END AS promo_responsiveness,
    COALESCE(ex.display_exposure_rate, 0) AS display_exposure_rate,
    COALESCE(ex.mailer_exposure_rate, 0) AS mailer_exposure_rate,
    dm.department,
    dm.dept_pct
FROM read_parquet('{raw}') AS r
CROSS JOIN global_bounds AS gb
LEFT JOIN last_trip AS lt ON r.household_key = lt.household_key
LEFT JOIN price_features AS pf ON r.household_key = pf.household_key
LEFT JOIN campaign_counts AS camp ON r.household_key = camp.household_key
LEFT JOIN coupon_counts AS cc ON r.household_key = cc.household_key
LEFT JOIN exposure_features AS ex ON r.household_key = ex.household_key
LEFT JOIN dept_mix AS dm ON r.household_key = dm.household_key
ORDER BY r.household_key, dm.department
"""


def pivot_department_mix(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long department mix rows into wide percentage columns.

    Args:
        long_df: Query result with household_key, department, dept_pct columns.

    Returns:
        Wide DataFrame with dept_*_pct columns per household.
    """
    base_cols = [
        c
        for c in long_df.columns
        if c not in {"department", "dept_pct"}
    ]
    base = long_df[base_cols].drop_duplicates(subset=["household_key"])
    mix = long_df[["household_key", "department", "dept_pct"]].dropna(subset=["department"])
    if mix.empty:
        return base

    mix = mix.copy()
    mix["department_col"] = mix["department"].map(
        lambda d: f"dept_{sanitize_column_name(str(d))}_pct"
    )
    wide = mix.pivot_table(
        index="household_key",
        columns="department_col",
        values="dept_pct",
        aggfunc="first",
        fill_value=0.0,
    ).reset_index()
    return base.merge(wide, on="household_key", how="left").fillna(0.0)


def run_features(
    output_path: Path = DEFAULT_OUTPUT,
    data_dir: Path = DATA_DIR,
    raw_input: Path = RAW_INPUT,
) -> pd.DataFrame:
    """Execute feature engineering and write household_features.parquet.

    Args:
        output_path: Destination parquet file path.
        data_dir: Directory containing source CSV files.
        raw_input: Path to Module 1 raw household features parquet.

    Returns:
        Final feature DataFrame.
    """
    paths = get_data_paths(data_dir)
    paths["raw_features"] = raw_input

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required file for {name}: {path}")

    query = build_feature_query(paths)
    conn = duckdb.connect()
    try:
        long_df = conn.execute(query).fetchdf()
        demographics = conn.execute(
            f"SELECT * FROM read_csv_auto('{paths['hh_demographic'].as_posix()}', header=true)"
        ).fetchdf()
    finally:
        conn.close()

    features = pivot_department_mix(long_df)
    demo_encoded = encode_demographics(demographics, features["household_key"])
    features = features.merge(demo_encoded, on="household_key", how="left")

    demo_dummy_cols = [c for c in features.columns if c.startswith(tuple(DEMO_COLUMNS))]
    features[demo_dummy_cols] = features[demo_dummy_cols].fillna(0).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return features


def main() -> None:
    """Run feature engineering and print schema plus summary statistics."""
    print("=" * 72)
    print("Module 2: Feature Engineering")
    print("=" * 72)

    df = run_features()
    print(f"\n--- Row Count: {len(df):,} ---\n")
    print("--- Schema ---\n")
    print(df.dtypes.to_string())

    print("\n--- Sample Rows (first 5) ---\n")
    print(df.head(5).to_string(index=False))

    print("\n--- Summary Statistics (describe) ---\n")
    print(df.describe().to_string())

    print(f"\n--- Output written to: {DEFAULT_OUTPUT} ---")


if __name__ == "__main__":
    main()
