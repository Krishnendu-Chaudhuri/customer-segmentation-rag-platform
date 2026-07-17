"""Generate tiny synthetic CSV fixtures for pipeline integration tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shopper_segmentation.features import DEMO_COLUMNS


def write_pipeline_fixtures(data_dir: Path, n_households: int = 40) -> None:
    """Write minimal valid CSV fixtures for an end-to-end pipeline smoke test.

    Args:
        data_dir: Destination directory for generated CSV files.
        n_households: Number of synthetic households to create.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    households = list(range(1, n_households + 1))
    products = list(range(101, 111))

    txn_rows: list[dict[str, object]] = []
    basket_id = 1
    for household in households:
        for week in range(1, 4):
            for product_id in products[:3]:
                txn_rows.append(
                    {
                        "household_key": household,
                        "BASKET_ID": basket_id,
                        "DAY": week,
                        "PRODUCT_ID": product_id,
                        "SALES_VALUE": float(10 + household + product_id % 5),
                        "RETAIL_DISC": float(product_id % 3),
                        "COUPON_DISC": float(household % 2),
                        "WEEK_NO": week,
                        "STORE_ID": 1,
                    }
                )
                basket_id += 1

    pd.DataFrame(txn_rows).to_csv(data_dir / "transaction_data.csv", index=False)

    pd.DataFrame(
        {
            "PRODUCT_ID": products,
            "DEPARTMENT": ["GROCERY" if p % 2 else "DRUG" for p in products],
            "BRAND": [f"Brand{p}" for p in products],
            "COMMODITY_DESC": [f"Commodity{p}" for p in products],
        }
    ).to_csv(data_dir / "product.csv", index=False)

    demo_rows = []
    for household in households:
        row = {"household_key": household}
        for idx, col in enumerate(DEMO_COLUMNS):
            row[col] = f"Value{(household + idx) % 3}"
        demo_rows.append(row)
    pd.DataFrame(demo_rows).to_csv(data_dir / "hh_demographic.csv", index=False)

    campaign_rows = [
        {"household_key": household, "CAMPAIGN": 1 if household % 2 else 2}
        for household in households
    ]
    pd.DataFrame(campaign_rows).to_csv(data_dir / "campaign_table.csv", index=False)

    pd.DataFrame(
        {
            "CAMPAIGN": [1, 2],
            "DESCRIPTION": ["Spring Promo", "Summer Promo"],
            "START_DAY": [1, 20],
            "END_DAY": [15, 35],
        }
    ).to_csv(data_dir / "campaign_desc.csv", index=False)

    pd.DataFrame(
        {"COUPON_UPC": [1001, 1002], "DESCRIPTION": ["Coupon A", "Coupon B"]}
    ).to_csv(data_dir / "coupon.csv", index=False)

    coupon_rows = [
        {"household_key": household, "COUPON_UPC": 1001, "DAY": 2}
        for household in households[:20]
    ]
    pd.DataFrame(coupon_rows).to_csv(data_dir / "coupon_redempt.csv", index=False)

    causal_rows = []
    for product_id in products:
        for week in range(1, 4):
            causal_rows.append(
                {
                    "PRODUCT_ID": product_id,
                    "STORE_ID": 1,
                    "WEEK_NO": week,
                    "DISPLAY": 1 if product_id % 2 == 0 else 0,
                    "MAILER": 1 if product_id % 3 == 0 else 0,
                }
            )
    pd.DataFrame(causal_rows).to_csv(data_dir / "causal_data.csv", index=False)
