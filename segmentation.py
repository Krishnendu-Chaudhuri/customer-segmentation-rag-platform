"""Household segmentation module using clustering and stability analysis.

Fits KMeans on scaled features, selects k via silhouette/elbow, cross-checks
with GaussianMixture, evaluates bootstrap stability, and writes segment profiles.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from etl import OUTPUT_DIR

FEATURES_INPUT = OUTPUT_DIR / "household_features.parquet"
SEGMENTS_OUTPUT = OUTPUT_DIR / "household_segments.parquet"
PROFILES_OUTPUT = OUTPUT_DIR / "segment_profiles.json"
PROFILE_TABLE_OUTPUT = OUTPUT_DIR / "segment_profiles.md"

K_RANGE = range(4, 9)
BOOTSTRAP_RUNS = 10
BOOTSTRAP_FRACTION = 0.8
RANDOM_STATE = 42

NAMING_FEATURES = [
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


def load_feature_matrix(features_path: Path = FEATURES_INPUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load household features and separate identifiers from model inputs.

    Args:
        features_path: Path to engineered household features parquet.

    Returns:
        Tuple of (full DataFrame, feature matrix without household_key).
    """
    df = pd.read_parquet(features_path)
    feature_cols = [c for c in df.columns if c != "household_key"]
    return df, df[feature_cols]


def scale_features(feature_matrix: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Standardize features for clustering.

    Args:
        feature_matrix: Raw feature DataFrame.

    Returns:
        Tuple of scaled numpy array and fitted scaler.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)
    return scaled, scaler


def evaluate_kmeans_k(
    scaled_features: np.ndarray,
    k_range: range = K_RANGE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Evaluate KMeans models across a range of k values.

    Args:
        scaled_features: Standardized feature matrix.
        k_range: Candidate cluster counts.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with k, inertia, and silhouette score columns.
    """
    rows: list[dict[str, float | int]] = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(scaled_features)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(scaled_features, labels),
            }
        )
    return pd.DataFrame(rows)


def select_best_k(evaluation: pd.DataFrame) -> int:
    """Select the best k using silhouette score with elbow as tie-breaker.

    Args:
        evaluation: Output from evaluate_kmeans_k.

    Returns:
        Selected number of clusters.
    """
    best_silhouette = evaluation.loc[evaluation["silhouette"].idxmax(), "k"]
    return int(best_silhouette)


def fit_kmeans(
    scaled_features: np.ndarray,
    k: int,
    random_state: int = RANDOM_STATE,
) -> tuple[KMeans, np.ndarray]:
    """Fit KMeans and return model plus cluster labels.

    Args:
        scaled_features: Standardized feature matrix.
        k: Number of clusters.
        random_state: Random seed.

    Returns:
        Tuple of fitted KMeans model and label array.
    """
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(scaled_features)
    return model, labels


def cross_check_gmm(
    scaled_features: np.ndarray,
    k: int,
    kmeans_labels: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, float]:
    """Cross-check KMeans labels with a Gaussian Mixture model.

    Args:
        scaled_features: Standardized feature matrix.
        k: Number of components.
        kmeans_labels: Labels from KMeans for comparison.
        random_state: Random seed.

    Returns:
        Tuple of GMM labels and ARI vs KMeans labels.
    """
    gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=5)
    gmm_labels = gmm.fit_predict(scaled_features)
    ari = adjusted_rand_score(kmeans_labels, gmm_labels)
    return gmm_labels, ari


def bootstrap_stability(
    scaled_features: np.ndarray,
    full_labels: np.ndarray,
    k: int,
    n_runs: int = BOOTSTRAP_RUNS,
    fraction: float = BOOTSTRAP_FRACTION,
    random_state: int = RANDOM_STATE,
) -> float:
    """Estimate clustering stability via bootstrap resampling.

    Args:
        scaled_features: Standardized feature matrix.
        full_labels: Labels fitted on the full dataset.
        k: Number of clusters.
        n_runs: Number of bootstrap iterations.
        fraction: Fraction of rows to sample each run.
        random_state: Random seed.

    Returns:
        Mean Adjusted Rand Index between bootstrap and full-data labels.
    """
    rng = np.random.default_rng(random_state)
    n_rows = scaled_features.shape[0]
    sample_size = int(n_rows * fraction)
    aris: list[float] = []

    for _ in range(n_runs):
        indices = rng.choice(n_rows, size=sample_size, replace=False)
        sample_x = scaled_features[indices]
        sample_labels = full_labels[indices]

        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        boot_labels = model.fit_predict(sample_x)
        aris.append(adjusted_rand_score(sample_labels, boot_labels))

    return float(np.mean(aris))


def compute_pca_coordinates(
    scaled_features: np.ndarray,
    n_components: int = 2,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Compute PCA coordinates for visualization only.

    Args:
        scaled_features: Standardized feature matrix.
        n_components: Number of PCA dimensions.
        random_state: Random seed.

    Returns:
        Array of shape (n_samples, n_components).
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(scaled_features)


def profile_segments(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compute cluster-level feature means and deltas vs population.

    Args:
        df: Full household feature DataFrame.
        labels: Cluster assignment array.
        feature_cols: Feature column names.

    Returns:
        Long profile DataFrame with segment_id, feature, segment_mean,
        population_mean, and delta columns.
    """
    working = df.copy()
    working["segment_id"] = labels
    population_means = working[feature_cols].mean()

    profiles: list[dict[str, float | int | str]] = []
    for segment_id in sorted(working["segment_id"].unique()):
        segment_df = working[working["segment_id"] == segment_id]
        segment_means = segment_df[feature_cols].mean()
        for feature in feature_cols:
            profiles.append(
                {
                    "segment_id": int(segment_id),
                    "feature": feature,
                    "segment_mean": float(segment_means[feature]),
                    "population_mean": float(population_means[feature]),
                    "delta": float(segment_means[feature] - population_means[feature]),
                }
            )
    return pd.DataFrame(profiles)


def profile_markdown_table(profile_df: pd.DataFrame) -> str:
    """Render segment profiles as a markdown table of mean feature deltas.

    Args:
        profile_df: Long-form profile DataFrame.

    Returns:
        Markdown string with one row per segment/feature combination.
    """
    pivot = profile_df.pivot(
        index="feature",
        columns="segment_id",
        values="segment_mean",
    ).round(4)
    pop = profile_df.drop_duplicates("feature").set_index("feature")["population_mean"].round(4)
    pivot["population"] = pop
    pivot = pivot.sort_index()
    return pivot.reset_index().to_markdown(index=False)


def _feature_level(value: float, population: float, spread: float) -> str:
    """Classify a feature value as high, low, or typical vs population."""
    if spread <= 0:
        return "typical"
    z = (value - population) / spread
    if z >= 0.5:
        return "high"
    if z <= -0.5:
        return "low"
    return "typical"


def generate_segment_name(
    segment_means: pd.Series,
    population_means: pd.Series,
    feature_spreads: pd.Series,
) -> str:
    """Generate a business-friendly segment name from distinguishing features.

    Args:
        segment_means: Mean feature values for the segment.
        population_means: Population mean feature values.
        feature_spreads: Population standard deviations for naming features.

    Returns:
        Rule-based segment label.
    """
    levels = {
        feature: _feature_level(
            segment_means.get(feature, population_means.get(feature, 0.0)),
            population_means.get(feature, 0.0),
            feature_spreads.get(feature, 1.0),
        )
        for feature in NAMING_FEATURES
    }

    monetary = levels.get("monetary", "typical")
    frequency = levels.get("frequency", "typical")
    price = levels.get("price_sensitivity", "typical")
    promo = levels.get("promo_responsiveness", "typical")
    basket = levels.get("avg_basket_size", "typical")
    recency = levels.get("recency", "typical")

    if monetary == "high" and price == "low":
        return "Premium Loyal Shoppers"
    if price == "high" and frequency == "high":
        return "Budget-Conscious Bulk Buyers"
    if promo == "high" and price == "high":
        return "Promo-Driven Deal Hunters"
    if promo == "high":
        return "Campaign-Responsive Shoppers"
    if frequency == "high" and basket == "high":
        return "High-Volume Stock-Up Shoppers"
    if monetary == "high" and frequency == "high":
        return "High-Value Frequent Shoppers"
    if recency == "low" and frequency == "high":
        return "Active Routine Shoppers"
    if monetary == "low" and frequency == "low":
        return "Occasional Value Seekers"
    if basket == "high":
        return "Large-Basket Shoppers"
    if price == "high":
        return "Discount-Sensitive Shoppers"
    return "Balanced Mainstream Shoppers"


def build_segment_narrative(
    segment_id: int,
    size: int,
    total_households: int,
    name: str,
    profile_df: pd.DataFrame,
) -> str:
    """Create a short narrative highlighting top distinguishing features.

    Args:
        segment_id: Cluster identifier.
        size: Number of households in the segment.
        total_households: Total households in the dataset.
        name: Generated segment name.
        profile_df: Long-form profile DataFrame.

    Returns:
        Human-readable segment narrative.
    """
    segment_profile = profile_df[profile_df["segment_id"] == segment_id].copy()
    segment_profile["abs_delta"] = segment_profile["delta"].abs()
    top_features = segment_profile.nlargest(3, "abs_delta")

    highlights: list[str] = []
    for _, row in top_features.iterrows():
        direction = "above" if row["delta"] >= 0 else "below"
        highlights.append(
            f"{row['feature']} is {direction} average "
            f"({row['segment_mean']:.3f} vs {row['population_mean']:.3f})"
        )

    pct = 100.0 * size / total_households
    highlight_text = "; ".join(highlights)
    return (
        f"{name} includes {size:,} households ({pct:.1f}% of base). "
        f"Key traits: {highlight_text}."
    )


def build_segment_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
    evaluation: pd.DataFrame,
    selected_k: int,
    gmm_ari: float,
    bootstrap_ari: float,
) -> dict[str, object]:
    """Build segment profile JSON payload.

    Args:
        df: Full household feature DataFrame.
        labels: Cluster labels.
        feature_cols: Feature column names.
        evaluation: K evaluation metrics.
        selected_k: Chosen cluster count.
        gmm_ari: ARI between KMeans and GMM.
        bootstrap_ari: Mean bootstrap ARI.

    Returns:
        JSON-serializable segment profile dictionary.
    """
    profile_df = profile_segments(df, labels, feature_cols)
    population_means = df[feature_cols].mean()
    feature_spreads = df[NAMING_FEATURES].std().replace(0, 1.0)

    segments: list[dict[str, object]] = []
    for segment_id in sorted(np.unique(labels)):
        segment_id = int(segment_id)
        segment_rows = df[labels == segment_id]
        segment_means = segment_rows[feature_cols].mean()
        name = generate_segment_name(segment_means, population_means, feature_spreads)
        size = int(len(segment_rows))
        feature_means = {col: float(segment_means[col]) for col in feature_cols}
        narrative = build_segment_narrative(
            segment_id, size, len(df), name, profile_df
        )

        segments.append(
            {
                "id": segment_id,
                "name": name,
                "size": size,
                "feature_means": feature_means,
                "narrative": narrative,
            }
        )

    return {
        "metadata": {
            "selected_k": selected_k,
            "k_evaluation": evaluation.to_dict(orient="records"),
            "silhouette_score": float(
                evaluation.loc[evaluation["k"] == selected_k, "silhouette"].iloc[0]
            ),
            "gmm_ari_vs_kmeans": float(gmm_ari),
            "mean_bootstrap_ari": float(bootstrap_ari),
        },
        "segments": segments,
    }


def run_segmentation(
    features_path: Path = FEATURES_INPUT,
    segments_output: Path = SEGMENTS_OUTPUT,
    profiles_output: Path = PROFILES_OUTPUT,
    profile_table_output: Path = PROFILE_TABLE_OUTPUT,
) -> dict[str, object]:
    """Run the full segmentation pipeline.

    Args:
        features_path: Input features parquet path.
        segments_output: Output parquet for household segment assignments.
        profiles_output: Output JSON for segment profiles.
        profile_table_output: Output markdown profile table.

    Returns:
        Segment profiles dictionary.
    """
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")

    df, feature_matrix = load_feature_matrix(features_path)
    feature_cols = list(feature_matrix.columns)
    scaled, _ = scale_features(feature_matrix)

    evaluation = evaluate_kmeans_k(scaled)
    selected_k = select_best_k(evaluation)
    _, labels = fit_kmeans(scaled, selected_k)
    _, gmm_ari = cross_check_gmm(scaled, selected_k, labels)
    bootstrap_ari = bootstrap_stability(scaled, labels, selected_k)
    pca_coords = compute_pca_coordinates(scaled)

    profiles = build_segment_profiles(
        df,
        labels,
        feature_cols,
        evaluation,
        selected_k,
        gmm_ari,
        bootstrap_ari,
    )

    segments_output.parent.mkdir(parents=True, exist_ok=True)
    segments_df = pd.DataFrame(
        {
            "household_key": df["household_key"],
            "segment_id": labels,
            "pca_1": pca_coords[:, 0],
            "pca_2": pca_coords[:, 1],
        }
    )
    segments_df.to_parquet(segments_output, index=False)

    profile_df = profile_segments(df, labels, feature_cols)
    markdown_table = profile_markdown_table(profile_df)

    with profiles_output.open("w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

    profile_table_output.write_text(markdown_table, encoding="utf-8")
    profiles["profile_markdown"] = markdown_table
    return profiles


def main() -> None:
    """Run segmentation and print evaluation metrics and segment summaries."""
    print("=" * 72)
    print("Module 3: Segmentation")
    print("=" * 72)

    profiles = run_segmentation()
    metadata = profiles["metadata"]

    print("\n--- K Evaluation (k=4..8) ---\n")
    print(pd.DataFrame(metadata["k_evaluation"]).to_string(index=False))

    print(f"\n--- Selected k: {metadata['selected_k']} ---")
    print(f"Silhouette score: {metadata['silhouette_score']:.4f}")
    print(f"GMM ARI vs KMeans: {metadata['gmm_ari_vs_kmeans']:.4f}")
    print(f"Mean bootstrap ARI (10 runs @ 80%): {metadata['mean_bootstrap_ari']:.4f}")

    print("\n--- Segment Summary ---\n")
    for segment in profiles["segments"]:
        print(f"Segment {segment['id']}: {segment['name']} ({segment['size']:,} households)")
        print(f"  {segment['narrative']}\n")

    print("--- Profile Markdown Table (preview) ---\n")
    print(str(profiles["profile_markdown"])[:2000])
    print("\n...")

    print(f"\n--- Outputs ---")
    print(f"Segments: {SEGMENTS_OUTPUT}")
    print(f"Profiles: {PROFILES_OUTPUT}")
    print(f"Profile table: {PROFILE_TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
