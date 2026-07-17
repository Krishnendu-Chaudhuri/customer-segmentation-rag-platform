"""Regression tests for resilient artifact loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from shopper_segmentation import artifacts, paths
from shopper_segmentation.artifacts import ArtifactError, ensure_artifacts, reset_artifact_cache
from shopper_segmentation.rag.build_cards import build_all_cards
from tests.fixtures.generate_fixtures import write_pipeline_fixtures


@pytest.fixture(autouse=True)
def reset_artifact_state() -> None:
    """Reset artifact caches after each test in this module."""
    yield
    reset_artifact_cache()
    artifacts._artifacts_ready = False


@pytest.fixture
def isolated_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use an isolated output directory without source CSV data."""
    empty_data_dir = tmp_path / "empty_data"
    empty_data_dir.mkdir()
    output_dir = tmp_path / "output"
    monkeypatch.setattr(paths, "DATA_DIR", empty_data_dir)
    monkeypatch.setattr(paths, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(artifacts, "_artifacts_ready", False)
    reset_artifact_cache()
    return output_dir


def test_ensure_creates_output_directory(isolated_output: Path) -> None:
    """Missing output directory should be created during ensure_artifacts()."""
    assert not isolated_output.exists()
    ensure_artifacts()
    assert isolated_output.exists()
    assert (isolated_output / "segment_profiles.json").exists()


def test_seed_from_bundled_when_no_source_data(isolated_output: Path) -> None:
    """Artifacts should seed from bundled samples when source CSVs are absent."""
    ensure_artifacts()
    profiles = (isolated_output / "segment_profiles.json").read_text(encoding="utf-8")
    assert "Budget Essentials Shoppers" in profiles
    assert len(list(isolated_output.glob("*.json"))) == 3


def test_pipeline_regenerates_from_synthetic_csvs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline should regenerate artifacts when synthetic source CSVs exist."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    write_pipeline_fixtures(data_dir)

    monkeypatch.setattr(paths, "DATA_DIR", data_dir)
    monkeypatch.setattr(paths, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(artifacts, "_artifacts_ready", False)
    reset_artifact_cache()

    ensure_artifacts(force=True)

    assert (output_dir / "segment_profiles.json").exists()
    assert (output_dir / "segment_recommendations.json").exists()
    assert (output_dir / "uplift_report.json").exists()
    assert (output_dir / "household_segments.parquet").exists()


def test_build_all_cards_after_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_all_cards() should succeed after artifacts are regenerated."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    write_pipeline_fixtures(data_dir)

    monkeypatch.setattr(paths, "DATA_DIR", data_dir)
    monkeypatch.setattr(paths, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(artifacts, "_artifacts_ready", False)
    reset_artifact_cache()

    cards = build_all_cards()
    assert len(cards) > 0
    assert all("content" in card for card in cards)


def test_api_returns_200_not_500_with_artifacts(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """Segment endpoints should return 200 when artifacts are available."""
    response = client.get("/segments", headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json()) == 8

    detail_response = client.get("/segments/2", headers=auth_headers)
    assert detail_response.status_code == 200
    assert "monetary" in detail_response.json()["feature_means"]

    recommendations_response = client.get("/segments/2/recommendations", headers=auth_headers)
    assert recommendations_response.status_code == 200
    assert recommendations_response.json()["segment_id"] == 2
    assert recommendations_response.json()["recommendations"]


def test_api_unknown_segment_returns_404(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """Unknown segment ids should return 404 instead of 500."""
    response = client.get("/segments/999", headers=auth_headers)
    assert response.status_code == 404


def test_missing_bundled_samples_raises_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Artifact initialization should raise ArtifactError when seeding is impossible."""
    empty_data_dir = tmp_path / "empty_data"
    empty_data_dir.mkdir()
    output_dir = tmp_path / "output"
    missing_samples = tmp_path / "missing_samples"

    monkeypatch.setattr(paths, "DATA_DIR", empty_data_dir)
    monkeypatch.setattr(paths, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(paths, "SAMPLE_OUTPUT_DIR", missing_samples)
    monkeypatch.setattr(artifacts, "_artifacts_ready", False)
    reset_artifact_cache()

    with pytest.raises(ArtifactError, match="Bundled sample artifacts directory not found"):
        ensure_artifacts(force=True)
