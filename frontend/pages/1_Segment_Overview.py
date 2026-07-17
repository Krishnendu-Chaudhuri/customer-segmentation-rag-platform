"""Segment overview page with PCA scatter and radar charts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.path_setup import ensure_project_root

ensure_project_root()

from frontend.api_client import get_segment, get_segments
from frontend.charts import (
    build_pca_scatter,
    build_radar_chart,
    load_population_feature_means,
    load_segment_scatter_data,
)

st.set_page_config(page_title="Segment Overview", layout="wide")
st.title("Segment Overview")

try:
    segments = get_segments()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

segment_names = {int(s["id"]): str(s["name"]) for s in segments}
low_confidence_ids = {int(s["id"]) for s in segments if s.get("low_confidence")}
scatter_df = load_segment_scatter_data()
population_means = load_population_feature_means()

col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(build_pca_scatter(scatter_df, segment_names), use_container_width=True)

with col2:
    summary_df = pd.DataFrame(segments)[["id", "name", "size"]]
    summary_df.columns = ["Segment ID", "Name", "Households"]
    fig_bar = px.bar(
        summary_df,
        x="Segment ID",
        y="Households",
        color="Name",
        title="Segment Sizes",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

if low_confidence_ids:
    st.warning(
        "Low-confidence segments (fewer than 30 households): "
        + ", ".join(f"Segment {sid}" for sid in sorted(low_confidence_ids))
    )

st.subheader("Segment Radar Profile")
selected_id = st.selectbox(
    "Select segment for radar comparison vs population",
    options=[int(s["id"]) for s in segments],
    format_func=lambda sid: f"{sid}: {segment_names[sid]}",
)

try:
    detail = get_segment(selected_id)
    if detail.get("low_confidence"):
        st.warning(
            f"Segment {selected_id} has only {detail['size']} households — "
            "interpret profiles and comparisons with caution."
        )
    st.plotly_chart(
        build_radar_chart(
            segment_name=str(detail["name"]),
            segment_means=detail["feature_means"],
            population_means=population_means,
        ),
        use_container_width=True,
    )
    st.caption(detail["narrative"])
except RuntimeError as exc:
    st.error(str(exc))
