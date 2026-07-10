"""Ensure project root is on sys.path for Streamlit page imports."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_project_root() -> Path:
    """Add project root to sys.path if needed.

    Returns:
        Project root directory path.
    """
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT
