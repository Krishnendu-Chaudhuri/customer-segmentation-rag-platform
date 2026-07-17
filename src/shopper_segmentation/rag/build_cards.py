"""Build markdown segment cards for RAG retrieval."""

from __future__ import annotations

from pathlib import Path

from shopper_segmentation import artifacts
from shopper_segmentation.paths import (
    profiles_path as default_profiles_path,
)
from shopper_segmentation.paths import (
    recommendations_path as default_recommendations_path,
)
from shopper_segmentation.paths import (
    uplift_path as default_uplift_path,
)

TOP_FEATURE_KEYS = [
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


def _index_by_segment_id(items: list[dict[str, object]], key: str = "segment_id") -> dict[int, dict]:
    """Index a list of segment records by segment id."""
    return {int(item[key]): item for item in items}


def format_top_features(feature_means: dict[str, float]) -> str:
    """Format key feature means as markdown bullets.

    Args:
        feature_means: Segment-level feature averages.

    Returns:
        Markdown bullet list.
    """
    lines: list[str] = []
    for feature in TOP_FEATURE_KEYS:
        if feature in feature_means:
            lines.append(f"- **{feature}**: {feature_means[feature]:.4f}")
    return "\n".join(lines)


def format_recommendations(recommendations: list[dict[str, object]], limit: int = 5) -> str:
    """Format product recommendations as markdown list.

    Args:
        recommendations: Ranked recommendation records.
        limit: Maximum products to include.

    Returns:
        Markdown numbered list.
    """
    if not recommendations:
        return "- No recommendations met the support threshold."

    lines: list[str] = []
    for idx, rec in enumerate(recommendations[:limit], start=1):
        lines.append(
            f"{idx}. Product **{rec['product_id']}** ({rec.get('department', 'N/A')}) — "
            f"lift **{rec['lift']}**, segment purchase rate **{rec['segment_purchase_rate']}**, "
            f"population rate **{rec['population_purchase_rate']}**, "
            f"buyers **{rec['segment_buyers']}** / segment size **{rec['segment_size']}**"
        )
    return "\n".join(lines)


def format_uplift(uplift: dict[str, object] | None) -> str:
    """Format uplift metrics as markdown bullets.

    Args:
        uplift: Segment uplift record, if available.

    Returns:
        Markdown bullet list.
    """
    if uplift is None:
        return "- No uplift data available for this segment."

    return (
        f"- **Incremental spend**: {uplift['incremental_spend_pct']}% "
        f"[{uplift['incremental_spend_pct_ci_lower']}%, {uplift['incremental_spend_pct_ci_upper']}%]\n"
        f"- **Treated mean spend**: {uplift['treated_mean_spend']}\n"
        f"- **Control mean spend**: {uplift['control_mean_spend']}\n"
        f"- **Treated households**: {uplift['treated_households']}\n"
        f"- **Control households**: {uplift['control_households']}"
    )


def build_segment_card(
    profile: dict[str, object],
    recommendations: dict[str, object] | None,
    uplift: dict[str, object] | None,
) -> str:
    """Build a markdown card for one segment.

    Args:
        profile: Segment profile from segment_profiles.json.
        recommendations: Recommendation record for the segment.
        uplift: Uplift record for the segment.

    Returns:
        Markdown document string.
    """
    segment_id = int(profile["id"])
    name = profile["name"]
    size = profile["size"]
    narrative = profile["narrative"]
    feature_means = profile["feature_means"]

    pct = 100.0 * size / 2500
    recs = recommendations.get("recommendations", []) if recommendations else []

    return f"""# Segment {segment_id}: {name}

## Definition
- **Segment ID**: {segment_id}
- **Segment name**: {name}
- **Households**: {size} ({pct:.1f}% of 2500 household base)
- **Summary**: {narrative}

## Top Features
{format_top_features(feature_means)}

## Top Recommended Products
{format_recommendations(recs)}

## Promo Performance
{format_uplift(uplift)}
"""


def build_all_cards(
    profiles_file: Path | None = None,
    recommendations_file: Path | None = None,
    uplift_file: Path | None = None,
) -> list[dict[str, object]]:
    """Build markdown cards for all segments.

    Args:
        profiles_file: Optional override path to segment_profiles.json.
        recommendations_file: Optional override path to segment_recommendations.json.
        uplift_file: Optional override path to uplift_report.json.

    Returns:
        List of card dicts with segment_id, segment_name, and content.

    Raises:
        ArtifactError: When required artifacts cannot be loaded or generated.
    """
    if profiles_file is None and recommendations_file is None and uplift_file is None:
        artifacts.ensure_artifacts()
        profiles = artifacts.load_profiles()
        recommendations = artifacts.load_recommendations()
        uplift = artifacts.load_uplift()
    else:
        artifacts.ensure_artifacts()
        import json

        def _load(path: Path) -> dict[str, object]:
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)

        profiles = _load(profiles_file or default_profiles_path())
        recommendations = _load(recommendations_file or default_recommendations_path())
        uplift = _load(uplift_file or default_uplift_path())

    rec_by_id = _index_by_segment_id(recommendations["segments"])
    uplift_by_id = _index_by_segment_id(uplift["segments"])

    cards: list[dict[str, object]] = []
    for profile in profiles["segments"]:
        segment_id = int(profile["id"])
        cards.append(
            {
                "segment_id": segment_id,
                "segment_name": profile["name"],
                "content": build_segment_card(
                    profile,
                    rec_by_id.get(segment_id),
                    uplift_by_id.get(segment_id),
                ),
            }
        )
    return cards
