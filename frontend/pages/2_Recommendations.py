"""Product recommendations page by segment."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.path_setup import ensure_project_root

ensure_project_root()

from frontend.api_client import get_recommendations, get_segments

st.set_page_config(page_title="Recommendations", layout="wide")
st.title("Segment Product Recommendations")

try:
    segments = get_segments()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

segment_names = {int(s["id"]): str(s["name"]) for s in segments}
selected_id = st.selectbox(
    "Select segment",
    options=[int(s["id"]) for s in segments],
    format_func=lambda sid: f"{sid}: {segment_names[sid]}",
)

try:
    payload = get_recommendations(selected_id)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

recs = payload.get("recommendations", [])
if not recs:
    st.warning(f"No recommendations available for segment {selected_id}.")
    st.stop()

df = pd.DataFrame(recs)
display_cols = [
    "product_id",
    "department",
    "brand",
    "commodity_desc",
    "lift",
    "segment_purchase_rate",
    "population_purchase_rate",
    "segment_buyers",
]
df = df[display_cols]
df.columns = [
    "Product ID",
    "Department",
    "Brand",
    "Commodity",
    "Lift",
    "Segment Rate",
    "Population Rate",
    "Segment Buyers",
]

st.subheader(f"Top products for {payload['segment_name']}")
st.dataframe(df, use_container_width=True, hide_index=True)

st.caption(
    "Lift = (in-segment purchase rate) / (population purchase rate). "
    "Products filtered by minimum support threshold."
)
