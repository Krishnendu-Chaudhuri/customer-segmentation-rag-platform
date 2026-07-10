"""Analyst chat page calling the FastAPI /chat endpoint."""

from __future__ import annotations

import streamlit as st

from frontend.path_setup import ensure_project_root

ensure_project_root()

from frontend.api_client import get_api_base_url, post_chat

st.set_page_config(page_title="Analyst Chat", layout="wide")
st.title("Analyst Chat")

st.markdown(
    f"Ask questions about shopper segments. Responses are powered by "
    f"**Groq Llama 3 70B** via `{get_api_base_url()}/chat`."
)

example_queries = [
    "Who are our high-value promo-sensitive shoppers?",
    "What products should we target to segment 2 this week?",
    "Which segment shows the strongest campaign uplift?",
]

query = st.text_area(
    "Your question",
    placeholder="Ask about segments, promotions, or product targeting...",
    height=100,
)

cols = st.columns(len(example_queries))
for col, example in zip(cols, example_queries, strict=True):
    if col.button(example, use_container_width=True):
        query = example

if st.button("Ask", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Querying analyst chatbot..."):
        try:
            result = post_chat(query.strip())
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

    st.subheader("Answer")
    st.markdown(result["answer"])

    st.subheader("Retrieved Segments")
    for segment in result.get("retrieved_segments", []):
        st.write(
            f"- Segment **{segment['segment_id']}**: {segment['segment_name']} "
            f"(distance: {segment['distance']:.4f})"
        )

    validation = result.get("validation", {})
    if validation.get("validated"):
        st.success("Validation: all cited numbers found in retrieved context.")
    else:
        unsupported = validation.get("unsupported_numbers", [])
        st.warning(f"Validation flags — unsupported numbers: {unsupported}")
