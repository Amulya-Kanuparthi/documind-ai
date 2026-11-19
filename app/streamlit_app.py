
from pathlib import Path

import streamlit as st
from documind.rag import ask_documind

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "artifacts" / "benchmark-index"

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize question history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🧠 DocuMind AI")
st.caption("Enterprise Document Intelligence Platform")
st.divider()

# Sidebar
st.sidebar.header("Settings")

retrieval_method = st.sidebar.selectbox(
    "Retrieval Method",
    ["FAISS", "Hybrid"],
    index=1,
)

top_k = st.sidebar.slider(
    "Number of Retrieved Chunks",
    min_value=1,
    max_value=10,
    value=3,
)

st.sidebar.info(
    "Powered by Groq LLM and Retrieval-Augmented Generation"
)

st.sidebar.divider()

st.sidebar.subheader("Session Statistics")
st.sidebar.metric(
    "Questions Asked",
    len(st.session_state.chat_history),
)

if st.sidebar.button("Clear History"):
    st.session_state.chat_history = []
    st.rerun()


def display_result(result):
    """Display the answer, sources and citation validation."""

    st.subheader("AI Answer")
    st.markdown(result["answer"])

    st.subheader("Retrieved Sources")

    if not result["sources"]:
        st.info("No supporting document chunks were retrieved.")
    else:
        for source in result["sources"]:
            citation = source["citation"]
            filename = source["source"]
            page = source["page"]
            chunk_id = source["chunk_id"]
            evidence = source["evidence"]

            with st.expander(
                f"[{citation}] {filename} (Page {page})"
            ):
                st.markdown("**Retrieved Evidence**")
                st.write(evidence)
                st.caption(f"Chunk ID: {chunk_id}")

        if result["citation_valid"]:
            st.success("Citation format validation passed.")
        else:
            st.warning(
                "Answer contains missing or invalid citations."
            )


# Question input
st.subheader("Ask Your Documents")

with st.form("question_form"):
    question = st.text_input(
        "Enter your question",
        placeholder=(
            "When must terminated employee accounts be disabled?"
        ),
    )

    submitted = st.form_submit_button(
        "Ask DocuMind",
        type="primary",
    )

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")

    elif not INDEX_DIR.is_dir():
        st.error(f"Document index not found: {INDEX_DIR}")

    else:
        with st.spinner(
            "Searching documents and generating answer..."
        ):
            try:
                result = ask_documind(
                    question=question.strip(),
                    index_dir=INDEX_DIR,
                    method=retrieval_method.lower(),
                    top_k=top_k,
                )

            except Exception as exc:
                st.error(f"Unable to generate answer: {exc}")

            else:
                # Save question and result in session history
                st.session_state.chat_history.append(
                    {
                        "question": question.strip(),
                        "method": retrieval_method,
                        "top_k": top_k,
                        "result": result,
                    }
                )

# Display latest answer
if st.session_state.chat_history:
    latest = st.session_state.chat_history[-1]

    st.divider()
    display_result(latest["result"])


# Question history
st.divider()
st.subheader("Question & Answer History")

if not st.session_state.chat_history:
    st.info("No questions asked yet.")

else:
    # Show newest questions first
    for i, entry in enumerate(
        reversed(st.session_state.chat_history),
        start=1,
    ):
        with st.expander(
            f"{i}. {entry['question']}"
        ):
            st.caption(
                f"Retrieval: {entry['method']} | "
                f"Top K: {entry['top_k']}"
            )

            st.markdown("**Answer**")
            st.markdown(entry["result"]["answer"])

            st.markdown("**Sources**")

            for source in entry["result"]["sources"]:
                st.write(
                    f"[{source['citation']}] "
                    f"{source['source']} "
                    f"(Page {source['page']})"
                )
