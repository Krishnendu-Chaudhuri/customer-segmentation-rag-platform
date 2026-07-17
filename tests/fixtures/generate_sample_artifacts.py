"""Write bundled sample pipeline JSON artifacts for CI and clean checkouts."""

from __future__ import annotations

import json
from pathlib import Path

FEATURE_KEYS = [
    "monetary",
    "frequency",
    "recency",
    "price_sensitivity",
    "promo_responsiveness",
    "display_exposure_rate",
    "mailer_exposure_rate",
    "avg_basket_size",
    "total_spend",
    "dept_grocery_pct",
]

SEGMENT_NAMES = [
    "Budget Essentials Shoppers",
    "Promo-Sensitive Regulars",
    "Premium Grocery Loyalists",
    "Occasional Basket Builders",
    "Display-Driven Explorers",
    "Mailer Responders",
    "Small-Basket Niche Buyers",
    "High-Value Frequent Shoppers",
]

SEGMENT_SIZES = [312, 67, 421, 298, 25, 189, 18, 1170]


def _feature_means(segment_id: int) -> dict[str, float]:
    """Build deterministic feature means for a segment."""
    base = 0.35 + (segment_id * 0.07)
    return {key: round(base + (idx * 0.03), 4) for idx, key in enumerate(FEATURE_KEYS)}


def build_profiles() -> dict[str, object]:
    """Build sample segment profiles with eight segments."""
    segments: list[dict[str, object]] = []
    for segment_id in range(8):
        size = SEGMENT_SIZES[segment_id]
        segments.append(
            {
                "id": segment_id,
                "name": SEGMENT_NAMES[segment_id],
                "size": size,
                "low_confidence": size < 30,
                "feature_means": _feature_means(segment_id),
                "narrative": (
                    f"Sample segment {segment_id} with {size} households for "
                    "bundled artifact testing and CI."
                ),
            }
        )
    return {
        "metadata": {
            "selected_k": 8,
            "silhouette_score": 0.41,
            "gmm_ari_vs_kmeans": 0.88,
            "mean_bootstrap_ari": 0.79,
            "source": "bundled_sample_output",
        },
        "segments": segments,
    }


def build_recommendations() -> dict[str, object]:
    """Build sample recommendations for all segments."""
    segments: list[dict[str, object]] = []
    for segment_id in range(8):
        size = SEGMENT_SIZES[segment_id]
        segments.append(
            {
                "segment_id": segment_id,
                "segment_name": SEGMENT_NAMES[segment_id],
                "recommendations": [
                    {
                        "product_id": 101 + segment_id,
                        "department": "GROCERY",
                        "brand": f"Brand{segment_id}",
                        "commodity_desc": f"Commodity{segment_id}",
                        "lift": round(1.5 + segment_id * 0.25, 4),
                        "segment_purchase_rate": round(0.12 + segment_id * 0.01, 4),
                        "population_purchase_rate": round(0.05 + segment_id * 0.005, 4),
                        "segment_buyers": max(10, size // 5),
                        "segment_size": size,
                    }
                ],
            }
        )
    return {
        "metadata": {
            "top_n": 10,
            "min_segment_support": 10,
            "min_population_rate": 0.001,
            "source": "bundled_sample_output",
        },
        "segments": segments,
    }


def build_uplift() -> dict[str, object]:
    """Build sample uplift report for all segments."""
    segments: list[dict[str, object]] = []
    for segment_id in range(8):
        size = SEGMENT_SIZES[segment_id]
        treated = max(10, size // 3)
        control = max(10, size // 4)
        segments.append(
            {
                "segment_id": segment_id,
                "segment_name": SEGMENT_NAMES[segment_id],
                "treated_households": treated,
                "control_households": control,
                "treated_mean_spend": round(42.0 + segment_id, 2),
                "control_mean_spend": round(30.0 + segment_id * 0.5, 2),
                "incremental_spend": round(12.0 + segment_id * 0.4, 2),
                "incremental_spend_ci_lower": round(8.0 + segment_id * 0.2, 2),
                "incremental_spend_ci_upper": round(16.0 + segment_id * 0.6, 2),
                "incremental_spend_pct": round(10.0 + segment_id * 4.3, 2),
                "incremental_spend_pct_ci_lower": round(5.0 + segment_id * 2.0, 2),
                "incremental_spend_pct_ci_upper": round(15.0 + segment_id * 6.0, 2),
                "campaigns": [],
            }
        )
    return {
        "metadata": {
            "method": "diff_in_means",
            "confidence_level": 0.95,
            "source": "bundled_sample_output",
        },
        "segments": segments,
    }


def write_sample_artifacts(destination: Path) -> None:
    """Write bundled sample JSON artifacts to destination."""
    destination.mkdir(parents=True, exist_ok=True)
    payloads = {
        "segment_profiles.json": build_profiles(),
        "segment_recommendations.json": build_recommendations(),
        "uplift_report.json": build_uplift(),
    }
    for filename, payload in payloads.items():
        path = destination / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote {path}")


def main() -> None:
    """Generate bundled sample artifacts under src/shopper_segmentation/resources."""
    repo_root = Path(__file__).resolve().parents[2]
    destination = repo_root / "src" / "shopper_segmentation" / "resources" / "sample_output"
    write_sample_artifacts(destination)


if __name__ == "__main__":
    main()
