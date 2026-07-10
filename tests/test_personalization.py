"""Unit tests for personalization lift and uplift helpers."""

from __future__ import annotations

import numpy as np
import pytest

from personalization import (
    compute_lift,
    compute_uplift_pct,
    diff_in_means_ci,
)


def test_compute_lift_basic() -> None:
    """Lift should be the ratio of segment to population purchase rates."""
    assert compute_lift(in_segment_rate=0.20, population_rate=0.10) == pytest.approx(2.0)
    assert compute_lift(in_segment_rate=0.05, population_rate=0.0) == 0.0


def test_compute_uplift_pct() -> None:
    """Uplift percent should measure treated lift over control."""
    assert compute_uplift_pct(treated_mean=120.0, control_mean=100.0) == pytest.approx(20.0)
    assert compute_uplift_pct(treated_mean=50.0, control_mean=0.0) == 0.0


def test_diff_in_means_ci_contains_difference() -> None:
    """Confidence interval should bracket the observed mean difference."""
    treated = np.array([120.0, 130.0, 110.0, 125.0])
    control = np.array([90.0, 95.0, 100.0, 92.0])
    diff, ci_lower, ci_upper = diff_in_means_ci(treated, control)

    assert diff == pytest.approx(treated.mean() - control.mean())
    assert ci_lower < diff < ci_upper


def test_diff_in_means_ci_empty_groups() -> None:
    """Empty treated or control groups should return zeroed intervals."""
    diff, ci_lower, ci_upper = diff_in_means_ci(np.array([]), np.array([1.0, 2.0]))
    assert diff == 0.0
    assert ci_lower == 0.0
    assert ci_upper == 0.0
