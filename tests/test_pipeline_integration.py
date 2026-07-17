"""End-to-end pipeline integration smoke test."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shopper_segmentation import etl, features, personalization, segmentation
from tests.fixtures.generate_fixtures import write_pipeline_fixtures


@pytest.fixture
def pipeline_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Prepare isolated data and output directories for the pipeline test."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    write_pipeline_fixtures(data_dir)

    monkeypatch.setattr(etl, "DATA_DIR", data_dir)
    monkeypatch.setattr(etl, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(etl, "DEFAULT_OUTPUT", output_dir / "household_features_raw.parquet")
    monkeypatch.setattr(features, "DATA_DIR", data_dir)
    monkeypatch.setattr(features, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(features, "RAW_INPUT", output_dir / "household_features_raw.parquet")
    monkeypatch.setattr(features, "DEFAULT_OUTPUT", output_dir / "household_features.parquet")
    monkeypatch.setattr(segmentation, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(segmentation, "FEATURES_INPUT", output_dir / "household_features.parquet")
    monkeypatch.setattr(segmentation, "SEGMENTS_OUTPUT", output_dir / "household_segments.parquet")
    monkeypatch.setattr(segmentation, "PROFILES_OUTPUT", output_dir / "segment_profiles.json")
    monkeypatch.setattr(
        segmentation, "PROFILE_TABLE_OUTPUT", output_dir / "segment_profiles.md"
    )
    monkeypatch.setattr(personalization, "DATA_DIR", data_dir)
    monkeypatch.setattr(personalization, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(personalization, "SEGMENTS_INPUT", output_dir / "household_segments.parquet")
    monkeypatch.setattr(personalization, "PROFILES_INPUT", output_dir / "segment_profiles.json")
    monkeypatch.setattr(
        personalization,
        "RECOMMENDATIONS_OUTPUT",
        output_dir / "segment_recommendations.json",
    )
    monkeypatch.setattr(
        personalization, "UPLIFT_OUTPUT", output_dir / "uplift_report.json"
    )

    return {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "raw_features": output_dir / "household_features_raw.parquet",
        "features": output_dir / "household_features.parquet",
        "segments": output_dir / "household_segments.parquet",
        "profiles": output_dir / "segment_profiles.json",
        "recommendations": output_dir / "segment_recommendations.json",
        "uplift": output_dir / "uplift_report.json",
    }


def test_full_pipeline_smoke(pipeline_paths: dict[str, Path]) -> None:
    """Run etl -> features -> segmentation -> personalization without errors."""
    etl.run_etl(output_path=pipeline_paths["raw_features"], data_dir=pipeline_paths["data_dir"])
    assert pipeline_paths["raw_features"].exists()
    raw_df = pd.read_parquet(pipeline_paths["raw_features"])
    assert "household_key" in raw_df.columns
    assert len(raw_df) > 0

    features.run_features(
        output_path=pipeline_paths["features"],
        data_dir=pipeline_paths["data_dir"],
        raw_input=pipeline_paths["raw_features"],
    )
    assert pipeline_paths["features"].exists()
    feat_df = pd.read_parquet(pipeline_paths["features"])
    assert "segment_id" not in feat_df.columns
    assert len(feat_df) > 0

    profiles = segmentation.run_segmentation(
        features_path=pipeline_paths["features"],
        segments_output=pipeline_paths["segments"],
        profiles_output=pipeline_paths["profiles"],
        profile_table_output=pipeline_paths["output_dir"] / "segment_profiles.md",
    )
    assert pipeline_paths["segments"].exists()
    assert pipeline_paths["profiles"].exists()
    seg_df = pd.read_parquet(pipeline_paths["segments"])
    assert {"household_key", "segment_id", "pca_1", "pca_2"}.issubset(seg_df.columns)
    assert len(profiles["segments"]) > 0
    assert all("low_confidence" in segment for segment in profiles["segments"])

    recommendations, uplift = personalization.run_personalization(
        segments_path=pipeline_paths["segments"],
        profiles_path=pipeline_paths["profiles"],
        recommendations_output=pipeline_paths["recommendations"],
        uplift_output=pipeline_paths["uplift"],
        data_dir=pipeline_paths["data_dir"],
    )
    assert pipeline_paths["recommendations"].exists()
    assert pipeline_paths["uplift"].exists()
    assert "segments" in recommendations
    assert "segments" in uplift
