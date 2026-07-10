"""ETL module: load and join dunnhumby transaction data via DuckDB.

Joins transaction_data, product, and hh_demographic CSVs, aggregates
household-level metrics, and writes household_features_raw.parquet.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_OUTPUT = OUTPUT_DIR / "household_features_raw.parquet"


def get_data_paths(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """Return paths to the CSV files used by the ETL pipeline.

    Args:
        data_dir: Directory containing the source CSV files.

    Returns:
        Mapping of logical names to file paths.
    """
    return {
        "transaction_data": data_dir / "transaction_data.csv",
        "product": data_dir / "product.csv",
        "hh_demographic": data_dir / "hh_demographic.csv",
    }


def build_etl_query(paths: dict[str, Path]) -> str:
    """Build the DuckDB SQL query for household-level aggregation.

    Joins transactions with product metadata and demographics (LEFT JOINs
    preserve all transaction households). Aggregates spend, trip, basket,
    and tenure metrics per household_key.

    Args:
        paths: Mapping of logical names to CSV file paths.

    Returns:
        Complete SQL query string.
    """
    txn = paths["transaction_data"].as_posix()
    product = paths["product"].as_posix()
    demo = paths["hh_demographic"].as_posix()

    return f"""
WITH txn_enriched AS (
    SELECT
        t.household_key,
        t.BASKET_ID,
        t.DAY,
        t.PRODUCT_ID,
        t.SALES_VALUE,
        t.WEEK_NO,
        p.DEPARTMENT,
        p.BRAND
    FROM read_csv_auto('{txn}', header=true) AS t
    LEFT JOIN read_csv_auto('{product}', header=true) AS p
        ON t.PRODUCT_ID = p.PRODUCT_ID
),
basket_stats AS (
    SELECT
        household_key,
        BASKET_ID,
        COUNT(*) AS items_in_basket
    FROM txn_enriched
    GROUP BY household_key, BASKET_ID
),
household_agg AS (
    SELECT
        t.household_key,
        SUM(t.SALES_VALUE) AS total_spend,
        COUNT(DISTINCT t.BASKET_ID) AS n_baskets,
        COUNT(DISTINCT t.DAY) AS n_trips,
        COUNT(DISTINCT t.PRODUCT_ID) AS n_unique_products,
        MAX(t.WEEK_NO) - MIN(t.WEEK_NO) + 1 AS tenure_weeks
    FROM txn_enriched AS t
    GROUP BY t.household_key
)
SELECT
    h.household_key,
    h.total_spend,
    h.n_baskets,
    h.n_trips,
    bs.avg_basket_size,
    h.n_unique_products,
    h.tenure_weeks
FROM household_agg AS h
LEFT JOIN (
    SELECT
        household_key,
        AVG(items_in_basket) AS avg_basket_size
    FROM basket_stats
    GROUP BY household_key
) AS bs ON h.household_key = bs.household_key
LEFT JOIN read_csv_auto('{demo}', header=true) AS d
    ON h.household_key = d.household_key
ORDER BY h.household_key
"""


def run_etl(
    output_path: Path = DEFAULT_OUTPUT,
    data_dir: Path = DATA_DIR,
) -> str:
    """Execute the ETL pipeline and write household features to parquet.

    Args:
        output_path: Destination parquet file path.
        data_dir: Directory containing source CSV files.

    Returns:
        The SQL query that was executed.
    """
    paths = get_data_paths(data_dir)
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required file for {name}: {path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    query = build_etl_query(paths)

    conn = duckdb.connect()
    try:
        conn.execute(
            f"COPY ({query}) TO '{output_path.as_posix()}' (FORMAT PARQUET)"
        )
    finally:
        conn.close()

    return query


def load_and_summarize(output_path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Load the output parquet and return as a DataFrame.

    Args:
        output_path: Path to the parquet file.

    Returns:
        DataFrame with household feature rows.
    """
    return pd.read_parquet(output_path)


def main() -> None:
    """Run ETL, print the query, sample rows, and summary statistics."""
    print("=" * 72)
    print("Module 1: ETL — Load & Join")
    print("=" * 72)

    query = run_etl()
    print("\n--- DuckDB Query ---\n")
    print(query.strip())

    df = load_and_summarize()
    print(f"\n--- Row Count: {len(df):,} ---\n")
    print("--- Sample Rows (first 10) ---\n")
    print(df.head(10).to_string(index=False))

    print("\n--- Summary Statistics (describe) ---\n")
    print(df.describe().to_string())

    print(f"\n--- Output written to: {DEFAULT_OUTPUT} ---")


if __name__ == "__main__":
    main()
