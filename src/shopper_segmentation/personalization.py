"""Personalization module: segment recommendations and campaign uplift analysis.

Computes product lift recommendations per segment and diff-in-means uplift
for campaign-exposed households vs matched segment controls.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from shopper_segmentation.etl import DATA_DIR, OUTPUT_DIR
from shopper_segmentation.logging_config import configure_logging

logger = logging.getLogger(__name__)

SEGMENTS_INPUT = OUTPUT_DIR / "household_segments.parquet"
PROFILES_INPUT = OUTPUT_DIR / "segment_profiles.json"
RECOMMENDATIONS_OUTPUT = OUTPUT_DIR / "segment_recommendations.json"
UPLIFT_OUTPUT = OUTPUT_DIR / "uplift_report.json"

TOP_N_PRODUCTS = 10
MIN_SEGMENT_SUPPORT = 10
MIN_POPULATION_RATE = 0.001
CONFIDENCE_Z = 1.96


def compute_lift(in_segment_rate: float, population_rate: float) -> float:
    """Compute product lift as segment rate divided by population rate.

    Args:
        in_segment_rate: Share of segment households purchasing the product.
        population_rate: Share of all households purchasing the product.

    Returns:
        Lift ratio, or 0.0 when population rate is zero.
    """
    if population_rate <= 0:
        return 0.0
    return in_segment_rate / population_rate


def compute_uplift_pct(treated_mean: float, control_mean: float) -> float:
    """Compute percentage incremental spend for treated vs control.

    Args:
        treated_mean: Mean spend for exposed households.
        control_mean: Mean spend for control households.

    Returns:
        Percent uplift relative to control mean.
    """
    if control_mean <= 0:
        return 0.0
    return 100.0 * (treated_mean - control_mean) / control_mean


def diff_in_means_ci(
    treated_values: np.ndarray,
    control_values: np.ndarray,
    z: float = CONFIDENCE_Z,
) -> tuple[float, float, float]:
    """Compute diff-in-means uplift and confidence interval.

    Args:
        treated_values: Spend values for treated households.
        control_values: Spend values for control households.
        z: Z-score for confidence interval width.

    Returns:
        Tuple of (absolute_diff, ci_lower, ci_upper) on the spend scale.
    """
    treated = np.asarray(treated_values, dtype=float)
    control = np.asarray(control_values, dtype=float)

    if treated.size == 0 or control.size == 0:
        return 0.0, 0.0, 0.0

    treated_mean = treated.mean()
    control_mean = control.mean()
    diff = treated_mean - control_mean

    if treated.size < 2 or control.size < 2:
        return float(diff), float(diff), float(diff)

    se = np.sqrt(treated.var(ddof=1) / treated.size + control.var(ddof=1) / control.size)
    return float(diff), float(diff - z * se), float(diff + z * se)


def load_segment_names(profiles_path: Path = PROFILES_INPUT) -> dict[int, str]:
    """Load segment id to business name mapping.

    Args:
        profiles_path: Path to segment_profiles.json.

    Returns:
        Mapping of segment_id to segment name.
    """
    with profiles_path.open(encoding="utf-8") as f:
        profiles = json.load(f)
    return {segment["id"]: segment["name"] for segment in profiles["segments"]}


def build_recommendations_query(
    data_dir: Path,
    segments_path: Path,
    min_support: int = MIN_SEGMENT_SUPPORT,
    top_n: int = TOP_N_PRODUCTS,
) -> str:
    """Build DuckDB SQL for segment product lift recommendations.

    Args:
        data_dir: Directory containing CSV source files.
        segments_path: Path to household segment assignments.
        min_support: Minimum households in segment buying a product.
        top_n: Number of products to return per segment.

    Returns:
        SQL query string.
    """
    txn = (data_dir / "transaction_data.csv").as_posix()
    product = (data_dir / "product.csv").as_posix()
    segments = segments_path.as_posix()

    return f"""
WITH segments AS (
    SELECT household_key, segment_id
    FROM read_parquet('{segments}')
),
segment_sizes AS (
    SELECT segment_id, COUNT(*) AS segment_size
    FROM segments
    GROUP BY segment_id
),
population_size AS (
    SELECT COUNT(*) AS n_households FROM segments
),
hh_product AS (
    SELECT DISTINCT
        s.segment_id,
        t.household_key,
        t.PRODUCT_ID
    FROM read_csv_auto('{txn}', header=true) AS t
    INNER JOIN segments AS s ON t.household_key = s.household_key
),
segment_buyers AS (
    SELECT
        segment_id,
        PRODUCT_ID,
        COUNT(DISTINCT household_key) AS segment_buyers
    FROM hh_product
    GROUP BY segment_id, PRODUCT_ID
),
population_buyers AS (
    SELECT
        PRODUCT_ID,
        COUNT(DISTINCT household_key) AS population_buyers
    FROM hh_product
    GROUP BY PRODUCT_ID
),
lift_calc AS (
    SELECT
        sb.segment_id,
        sb.PRODUCT_ID,
        sb.segment_buyers,
        ss.segment_size,
        pb.population_buyers,
        ps.n_households,
        sb.segment_buyers * 1.0 / ss.segment_size AS segment_rate,
        pb.population_buyers * 1.0 / ps.n_households AS population_rate,
        CASE
            WHEN pb.population_buyers > 0
            THEN (sb.segment_buyers * 1.0 / ss.segment_size)
                 / (pb.population_buyers * 1.0 / ps.n_households)
            ELSE 0
        END AS lift
    FROM segment_buyers AS sb
    JOIN segment_sizes AS ss ON sb.segment_id = ss.segment_id
    CROSS JOIN population_size AS ps
    JOIN population_buyers AS pb ON sb.PRODUCT_ID = pb.PRODUCT_ID
    WHERE sb.segment_buyers >= {min_support}
      AND pb.population_buyers * 1.0 / ps.n_households >= {MIN_POPULATION_RATE}
),
ranked AS (
    SELECT
        lc.*,
        p.DEPARTMENT,
        p.BRAND,
        p.COMMODITY_DESC,
        ROW_NUMBER() OVER (
            PARTITION BY lc.segment_id
            ORDER BY lc.lift DESC, lc.segment_buyers DESC
        ) AS product_rank
    FROM lift_calc AS lc
    LEFT JOIN read_csv_auto('{product}', header=true) AS p
        ON lc.PRODUCT_ID = p.PRODUCT_ID
)
SELECT
    segment_id,
    PRODUCT_ID,
    DEPARTMENT,
    BRAND,
    COMMODITY_DESC,
    segment_buyers,
    segment_size,
    population_buyers,
    n_households,
    segment_rate,
    population_rate,
    lift,
    product_rank
FROM ranked
WHERE product_rank <= {top_n}
ORDER BY segment_id, product_rank
"""


def build_uplift_query(data_dir: Path, segments_path: Path) -> str:
    """Build DuckDB SQL for household spend during campaign windows.

    Args:
        data_dir: Directory containing CSV source files.
        segments_path: Path to household segment assignments.

    Returns:
        SQL query returning segment, campaign, exposure, household, spend.
    """
    txn = (data_dir / "transaction_data.csv").as_posix()
    campaign_table = (data_dir / "campaign_table.csv").as_posix()
    campaign_desc = (data_dir / "campaign_desc.csv").as_posix()
    segments = segments_path.as_posix()

    return f"""
WITH segments AS (
    SELECT household_key, segment_id
    FROM read_parquet('{segments}')
),
campaign_windows AS (
    SELECT CAMPAIGN, START_DAY, END_DAY
    FROM read_csv_auto('{campaign_desc}', header=true)
),
exposures AS (
    SELECT DISTINCT household_key, CAMPAIGN
    FROM read_csv_auto('{campaign_table}', header=true)
),
segment_campaign_households AS (
    SELECT
        s.segment_id,
        s.household_key,
        cw.CAMPAIGN,
        cw.START_DAY,
        cw.END_DAY,
        CASE WHEN e.household_key IS NOT NULL THEN 1 ELSE 0 END AS exposed
    FROM segments AS s
    CROSS JOIN campaign_windows AS cw
    LEFT JOIN exposures AS e
        ON s.household_key = e.household_key
       AND cw.CAMPAIGN = e.CAMPAIGN
),
campaign_spend AS (
    SELECT
        sch.segment_id,
        sch.CAMPAIGN,
        sch.household_key,
        sch.exposed,
        COALESCE(SUM(t.SALES_VALUE), 0) AS campaign_spend
    FROM segment_campaign_households AS sch
    LEFT JOIN read_csv_auto('{txn}', header=true) AS t
        ON sch.household_key = t.household_key
       AND t.DAY BETWEEN sch.START_DAY AND sch.END_DAY
    GROUP BY sch.segment_id, sch.CAMPAIGN, sch.household_key, sch.exposed
)
SELECT *
FROM campaign_spend
"""


def recommendations_to_json(
    recommendations_df: pd.DataFrame,
    segment_names: dict[int, str],
) -> dict[str, object]:
    """Convert recommendations DataFrame to JSON-serializable structure.

    Args:
        recommendations_df: Ranked product recommendations per segment.
        segment_names: Segment id to name mapping.

    Returns:
        Recommendations payload dictionary.
    """
    segments: list[dict[str, object]] = []
    for segment_id, group in recommendations_df.groupby("segment_id"):
        segment_id = int(segment_id)
        recs: list[dict[str, object]] = []
        for _, row in group.iterrows():
            recs.append(
                {
                    "product_id": int(row["PRODUCT_ID"]),
                    "department": row.get("DEPARTMENT"),
                    "brand": row.get("BRAND"),
                    "commodity_desc": row.get("COMMODITY_DESC"),
                    "lift": round(float(row["lift"]), 4),
                    "segment_purchase_rate": round(float(row["segment_rate"]), 4),
                    "population_purchase_rate": round(float(row["population_rate"]), 4),
                    "segment_buyers": int(row["segment_buyers"]),
                    "segment_size": int(row["segment_size"]),
                }
            )
        segments.append(
            {
                "segment_id": segment_id,
                "segment_name": segment_names.get(segment_id, f"Segment {segment_id}"),
                "recommendations": recs,
            }
        )
    return {
        "metadata": {
            "top_n": TOP_N_PRODUCTS,
            "min_segment_support": MIN_SEGMENT_SUPPORT,
            "min_population_rate": MIN_POPULATION_RATE,
        },
        "segments": segments,
    }


def uplift_to_json(
    uplift_df: pd.DataFrame,
    segment_names: dict[int, str],
) -> dict[str, object]:
    """Aggregate campaign spend rows into segment uplift report.

    Args:
        uplift_df: Household-level campaign spend with exposure flag.
        segment_names: Segment id to name mapping.

    Returns:
        Uplift report dictionary with diff-in-means and confidence intervals.
    """
    segment_reports: list[dict[str, object]] = []

    for segment_id, segment_group in uplift_df.groupby("segment_id"):
        segment_id = int(segment_id)
        campaign_reports: list[dict[str, object]] = []
        treated_all: list[float] = []
        control_all: list[float] = []

        for campaign, campaign_group in segment_group.groupby("CAMPAIGN"):
            treated = campaign_group[campaign_group["exposed"] == 1]["campaign_spend"].to_numpy()
            control = campaign_group[campaign_group["exposed"] == 0]["campaign_spend"].to_numpy()

            if treated.size == 0 or control.size == 0:
                continue

            treated_mean = float(treated.mean())
            control_mean = float(control.mean())
            diff, ci_lower, ci_upper = diff_in_means_ci(treated, control)
            uplift_pct = compute_uplift_pct(treated_mean, control_mean)
            uplift_pct_lower = 100.0 * ci_lower / control_mean if control_mean > 0 else 0.0
            uplift_pct_upper = 100.0 * ci_upper / control_mean if control_mean > 0 else 0.0

            campaign_reports.append(
                {
                    "campaign": int(campaign),
                    "treated_households": int(treated.size),
                    "control_households": int(control.size),
                    "treated_mean_spend": round(treated_mean, 2),
                    "control_mean_spend": round(control_mean, 2),
                    "incremental_spend": round(diff, 2),
                    "incremental_spend_ci_lower": round(ci_lower, 2),
                    "incremental_spend_ci_upper": round(ci_upper, 2),
                    "incremental_spend_pct": round(uplift_pct, 2),
                    "incremental_spend_pct_ci_lower": round(
                        min(uplift_pct_lower, uplift_pct_upper), 2
                    ),
                    "incremental_spend_pct_ci_upper": round(
                        max(uplift_pct_lower, uplift_pct_upper), 2
                    ),
                }
            )
            treated_all.extend(treated.tolist())
            control_all.extend(control.tolist())

        if not campaign_reports:
            continue

        treated_arr = np.asarray(treated_all, dtype=float)
        control_arr = np.asarray(control_all, dtype=float)
        treated_mean = float(treated_arr.mean())
        control_mean = float(control_arr.mean())
        diff, ci_lower, ci_upper = diff_in_means_ci(treated_arr, control_arr)
        uplift_pct = compute_uplift_pct(treated_mean, control_mean)
        uplift_pct_lower = 100.0 * ci_lower / control_mean if control_mean > 0 else 0.0
        uplift_pct_upper = 100.0 * ci_upper / control_mean if control_mean > 0 else 0.0

        segment_reports.append(
            {
                "segment_id": segment_id,
                "segment_name": segment_names.get(segment_id, f"Segment {segment_id}"),
                "treated_households": int(treated_arr.size),
                "control_households": int(control_arr.size),
                "treated_mean_spend": round(treated_mean, 2),
                "control_mean_spend": round(control_mean, 2),
                "incremental_spend": round(diff, 2),
                "incremental_spend_ci_lower": round(ci_lower, 2),
                "incremental_spend_ci_upper": round(ci_upper, 2),
                "incremental_spend_pct": round(uplift_pct, 2),
                "incremental_spend_pct_ci_lower": round(
                    min(uplift_pct_lower, uplift_pct_upper), 2
                ),
                "incremental_spend_pct_ci_upper": round(
                    max(uplift_pct_lower, uplift_pct_upper), 2
                ),
                "campaigns": campaign_reports,
            }
        )

    return {
        "metadata": {
            "method": "diff_in_means",
            "confidence_level": 0.95,
            "description": (
                "Compares campaign-window spend for exposed households vs "
                "same-segment controls with no campaign exposure."
            ),
        },
        "segments": segment_reports,
    }


def run_personalization(
    data_dir: Path = DATA_DIR,
    segments_path: Path = SEGMENTS_INPUT,
    profiles_path: Path = PROFILES_INPUT,
    recommendations_output: Path = RECOMMENDATIONS_OUTPUT,
    uplift_output: Path = UPLIFT_OUTPUT,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run recommendation and uplift pipelines.

    Args:
        data_dir: Directory containing source CSV files.
        segments_path: Household segment assignments parquet.
        profiles_path: Segment profiles JSON for names.
        recommendations_output: Output path for recommendations JSON.
        uplift_output: Output path for uplift report JSON.

    Returns:
        Tuple of (recommendations dict, uplift dict).
    """
    for path in (segments_path, profiles_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    segment_names = load_segment_names(profiles_path)
    conn = duckdb.connect()
    try:
        recommendations_df = conn.execute(
            build_recommendations_query(data_dir, segments_path)
        ).fetchdf()
        uplift_df = conn.execute(build_uplift_query(data_dir, segments_path)).fetchdf()
    finally:
        conn.close()

    recommendations = recommendations_to_json(recommendations_df, segment_names)
    uplift = uplift_to_json(uplift_df, segment_names)

    recommendations_output.parent.mkdir(parents=True, exist_ok=True)
    with recommendations_output.open("w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=2)
    with uplift_output.open("w", encoding="utf-8") as f:
        json.dump(uplift, f, indent=2)

    return recommendations, uplift


def main() -> None:
    """Run personalization and log recommendation and uplift summaries."""
    configure_logging()
    logger.info("Module 4: Personalization — Recommendations & Uplift")

    recommendations, uplift = run_personalization()

    for segment in recommendations["segments"]:
        logger.info(
            "Segment %s: %s",
            segment["segment_id"],
            segment["segment_name"],
        )
        for rec in segment["recommendations"][:3]:
            logger.info(
                "  Product %s (%s) — lift %.2f, segment rate %.3f",
                rec["product_id"],
                rec["department"],
                rec["lift"],
                rec["segment_purchase_rate"],
            )

    for segment in uplift["segments"]:
        logger.info(
            "Segment %s: %s — %+.1f%% incremental spend [%+.1f%%, %+.1f%%]",
            segment["segment_id"],
            segment["segment_name"],
            segment["incremental_spend_pct"],
            segment["incremental_spend_pct_ci_lower"],
            segment["incremental_spend_pct_ci_upper"],
        )

    logger.info("Recommendations output: %s", RECOMMENDATIONS_OUTPUT)
    logger.info("Uplift report output: %s", UPLIFT_OUTPUT)


if __name__ == "__main__":
    main()
