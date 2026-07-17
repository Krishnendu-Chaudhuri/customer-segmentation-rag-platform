"""Streamlit dashboard entry point for shopper segmentation."""

from __future__ import annotations

import os
import sys

import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from path_setup import ensure_project_root

ensure_project_root()

from frontend.api_client import check_health, get_api_base_url

st.set_page_config(
    page_title="Shopper Segmentation Dashboard",
    page_icon="🛒",
    layout="wide",
)

st.title("Shopper Segmentation & Personalization Engine")
st.markdown(
    """
Welcome to the **dunnhumby Complete Journey** segmentation dashboard.

Use the sidebar pages to explore:
- **Segment Overview** — PCA cluster visualization and radar profiles
- **Recommendations** — top product targets by segment lift
- **Analyst Chat** — ask questions via the RAG-powered `/chat` API
"""
)

api_url = get_api_base_url()
if check_health():
    st.success(f"API connected at `{api_url}`")
else:
    st.error(
        f"API not reachable at `{api_url}`. "
        "Start the backend: `uvicorn app:app --host 127.0.0.1 --port 8000`"
    )

st.info(
    "Pipeline outputs: 2,500 households · 8 segments · "
    "RFM + promo features · Groq Llama 3 70B analyst chat"
)
