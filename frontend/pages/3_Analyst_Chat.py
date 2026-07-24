"""Analyst chat page calling the FastAPI /chat endpoint."""

from __future__ import annotations

import streamlit as st

from frontend.path_setup import ensure_project_root

ensure_project_root()

from frontend.api_client import ChatStreamError, get_api_base_url, post_chat_stream

st.set_page_config(page_title="Analyst Chat", layout="wide")
st.title("Analyst Chat")

if "chat_thread_id" not in st.session_state:
    st.session_state.chat_thread_id = None

st.markdown(
    f"Ask questions about shopper segments. Responses stream from "
    f"**Groq Llama 3 70B** via `{get_api_base_url()}/chat`."
)

if st.session_state.chat_thread_id:
    st.caption(f"Session thread: `{st.session_state.chat_thread_id}`")

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

    answer_placeholder = st.empty()
    stream_buffer: list[str] = [""]

    def append_token(token: str) -> None:
        stream_buffer[0] += token
        answer_placeholder.markdown(stream_buffer[0])

    with st.spinner("Streaming analyst response..."):
        try:
            result = post_chat_stream(
                query.strip(),
                thread_id=st.session_state.chat_thread_id,
                on_token=append_token,
            )
        except ChatStreamError as exc:
            if stream_buffer[0]:
                st.caption("Partial response")
                st.markdown(stream_buffer[0])
            st.error(f"Chat failed ({exc.code}): {exc.message}")
            st.stop()
        except Exception as exc:
            if stream_buffer[0]:
                st.caption("Partial response")
                st.markdown(stream_buffer[0])
            st.error(f"Unexpected chat error: {exc}")
            st.stop()

    if result.get("thread_id"):
        st.session_state.chat_thread_id = result["thread_id"]

    st.subheader("Answer")
    st.markdown(result.get("answer", stream_buffer[0]))

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

if st.button("New conversation"):
    st.session_state.chat_thread_id = None
    st.rerun()
