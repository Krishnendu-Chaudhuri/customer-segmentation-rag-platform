"""Ensure pipeline artifacts exist and provide cached JSON loaders."""

from __future__ import annotations

import json
import logging
import shutil
import threading
from functools import lru_cache
from pathlib import Path

from shopper_segmentation import etl, features, paths, personalization, segmentation
from shopper_segmentation.paths import (
    PIPELINE_SOURCE_FILES,
    REQUIRED_JSON_ARTIFACTS,
    features_path,
    profiles_path,
    raw_features_path,
    recommendations_path,
    segments_path,
    uplift_path,
)

logger = logging.getLogger(__name__)

_init_lock = threading.Lock()
_artifacts_ready = False


class ArtifactError(Exception):
    """Raised when required pipeline artifacts cannot be generated or loaded."""


def missing_artifacts(output_dir: Path | None = None) -> list[Path]:
    """Return paths to required JSON artifacts that do not exist."""
    return [
        path
        for name in REQUIRED_JSON_ARTIFACTS
        if not (path := _artifact_path(name, output_dir)).exists()
    ]


def _artifact_path(name: str, output_dir: Path | None = None) -> Path:
    """Map artifact filename to its full path."""
    mapping = {
        "segment_profiles.json": profiles_path(output_dir),
        "segment_recommendations.json": recommendations_path(output_dir),
        "uplift_report.json": uplift_path(output_dir),
    }
    return mapping[name]


def _has_pipeline_source_data(data_dir: Path) -> bool:
    """Return True when all pipeline source CSV files are present."""
    return all((data_dir / filename).exists() for filename in PIPELINE_SOURCE_FILES)


def run_full_pipeline(data_dir: Path, output_dir: Path) -> None:
    """Run etl -> features -> segmentation -> personalization into output_dir.

    Args:
        data_dir: Directory containing source CSV files.
        output_dir: Directory for generated artifacts.

    Raises:
        FileNotFoundError: When required source files are missing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output = raw_features_path(output_dir)
    features_output = features_path(output_dir)
    segments_output = segments_path(output_dir)
    profiles_output = profiles_path(output_dir)
    profile_table_output = output_dir / "segment_profiles.md"
    recommendations_output = recommendations_path(output_dir)
    uplift_output = uplift_path(output_dir)

    logger.info("Running full pipeline from %s into %s", data_dir, output_dir)
    etl.run_etl(output_path=raw_output, data_dir=data_dir)
    features.run_features(
        output_path=features_output,
        data_dir=data_dir,
        raw_input=raw_output,
    )
    segmentation.run_segmentation(
        features_path=features_output,
        segments_output=segments_output,
        profiles_output=profiles_output,
        profile_table_output=profile_table_output,
    )
    personalization.run_personalization(
        data_dir=data_dir,
        segments_path=segments_output,
        profiles_path=profiles_output,
        recommendations_output=recommendations_output,
        uplift_output=uplift_output,
    )


def _seed_from_bundled(output_dir: Path) -> None:
    """Copy bundled sample JSON artifacts into output_dir."""
    sample_dir = paths.SAMPLE_OUTPUT_DIR
    if not sample_dir.is_dir():
        raise ArtifactError(f"Bundled sample artifacts directory not found: {sample_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_JSON_ARTIFACTS:
        source = sample_dir / name
        if not source.exists():
            raise ArtifactError(f"Missing bundled sample artifact: {source}")
        destination = _artifact_path(name, output_dir)
        shutil.copy2(source, destination)
        logger.info("Seeded artifact from bundled sample: %s", destination)


def ensure_artifacts(*, force: bool = False) -> None:
    """Ensure required JSON artifacts exist in OUTPUT_DIR.

    Idempotent and thread-safe. Runs the full pipeline when source CSVs are
    available; otherwise seeds from bundled sample output files.

    Args:
        force: When True, regenerate or re-seed even if artifacts already exist.

    Raises:
        ArtifactError: When artifacts cannot be created.
    """
    global _artifacts_ready

    if not force and not missing_artifacts() and _artifacts_ready:
        return

    with _init_lock:
        if not force and not missing_artifacts() and _artifacts_ready:
            return

        logger.info("Ensuring pipeline artifacts in %s", paths.OUTPUT_DIR)
        missing = missing_artifacts()
        for path in missing:
            logger.warning("Missing artifact: %s", path)

        paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if force or missing_artifacts():
            if _has_pipeline_source_data(paths.DATA_DIR):
                run_full_pipeline(paths.DATA_DIR, paths.OUTPUT_DIR)
            elif force or missing_artifacts():
                _seed_from_bundled(paths.OUTPUT_DIR)

        remaining = missing_artifacts()
        if remaining:
            missing_names = ", ".join(path.name for path in remaining)
            raise ArtifactError(
                "Required pipeline artifacts are missing after initialization: "
                f"{missing_names}. Set DATA_DIR to a directory with source CSV files "
                "or run scripts/run_pipeline.sh to generate output artifacts."
            )

        reset_artifact_cache()
        _artifacts_ready = True
        logger.info("Pipeline artifacts ready in %s", paths.OUTPUT_DIR)


def reset_artifact_cache() -> None:
    """Clear in-memory artifact caches."""
    _load_profiles_cached.cache_clear()
    _load_recommendations_cached.cache_clear()
    _load_uplift_cached.cache_clear()


def _output_dir_key() -> str:
    """Return a stable cache key for the active output directory."""
    return str(paths.OUTPUT_DIR.resolve())


def _read_json(path: Path) -> dict[str, object]:
    """Read and parse a JSON file."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache
def _load_profiles_cached(output_dir_key: str) -> dict[str, object]:
    """Load segment profiles for a specific output directory."""
    ensure_artifacts()
    return _read_json(profiles_path())


@lru_cache
def _load_recommendations_cached(output_dir_key: str) -> dict[str, object]:
    """Load segment recommendations for a specific output directory."""
    ensure_artifacts()
    return _read_json(recommendations_path())


@lru_cache
def _load_uplift_cached(output_dir_key: str) -> dict[str, object]:
    """Load uplift report for a specific output directory."""
    ensure_artifacts()
    return _read_json(uplift_path())


def load_profiles() -> dict[str, object]:
    """Load segment profiles, ensuring artifacts exist first."""
    return _load_profiles_cached(_output_dir_key())


def load_recommendations() -> dict[str, object]:
    """Load segment recommendations, ensuring artifacts exist first."""
    return _load_recommendations_cached(_output_dir_key())


def load_uplift() -> dict[str, object]:
    """Load uplift report, ensuring artifacts exist first."""
    return _load_uplift_cached(_output_dir_key())
