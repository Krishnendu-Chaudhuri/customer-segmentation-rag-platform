"""Ensure project root is on sys.path for Streamlit page imports."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]



def ensure_project_root():
    current = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(current)

    if root not in sys.path:
        sys.path.insert(0, root)
