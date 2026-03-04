"""
Streamlit GUI for the Multi-Agent Peer Review Engine.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from review_engine.config import ensure_types_loaded, get_config, REVIEW_TYPES
from review_engine.backends.claude import (
    MODEL_FALLBACK_CHAIN as CLAUDE_MODELS,
    DEFAULT_MODEL as CLAUDE_DEFAULT,
)
from review_engine.backends.gemini import (
    DEFAULT_FALLBACK_CHAIN as GEMINI_MODELS,
    DEFAULT_MODEL as GEMINI_DEFAULT,
)

# ── Load review type configs ─────────────────────────────────────────────────
ensure_types_loaded()


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multi-Agent Peer Review",
    page_icon="📝",
    layout="wide",
)


# ── Session state defaults ───────────────────────────────────────────────────

if "review_running" not in st.session_state:
    st.session_state.review_running = False
if "output_buffer" not in st.session_state:
    st.session_state.output_buffer = ""
if "result_md" not in st.session_state:
    st.session_state.result_md = None
if "result_pdf_bytes" not in st.session_state:
    st.session_state.result_pdf_bytes = None
if "current_step" not in st.session_state:
    st.session_state.current_step = ""


# ── Sidebar — Configuration ─────────────────────────────────────────────────

with st.sidebar:
    st.title("Multi-Agent Peer Review")
    st.markdown("---")

    # Review type
    type_options = {cfg.display_name: name for name, cfg in REVIEW_TYPES.items()}
    selected_display = st.selectbox(
        "Review Type",
        options=list(type_options.keys()),
        help="Choose the type of peer review to run.",
    )
    review_type = type_options[selected_display]
    config = get_config(review_type)

    # File upload
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help=f"Upload a {config.document_noun} as PDF.",
    )

    st.markdown("---")

    # Backend
    backend = st.radio(
        "AI Backend",
        options=["Claude", "Gemini"],
        horizontal=True,
    ).lower()

    # Model
    if backend == "claude":
        model_options = CLAUDE_MODELS
        default_model = config.claude_default_model
    else:
        model_options = config.gemini_fallback_chain
        default_model = config.gemini_default_model

    default_idx = model_options.index(default_model) if default_model in model_options else 0
    model = st.selectbox(
        "Model",
        options=model_options,
        index=default_idx,
        help="Select the AI model to use.",
    )

    # Max rounds
    if config.iteration.mode == "single_pass":
        max_rounds = 1
        st.info("Foundation reviews run a single pass (no iteration).")
    else:
        max_rounds = st.number_input(
            "Max Rounds",
            min_value=1,
            max_value=10,
            value=config.iteration.default_max_rounds,
            help="Maximum number of review rounds.",
        )

    st.markdown("---")

    # API key status
    if backend == "claude":
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        key_name = "ANTHROPIC_API_KEY"
    else:
        has_key = bool(os.environ.get("GEMINI_API_KEY"))
        key_name = "GEMINI_API_KEY"

    if has_key:
        st.success(f"{key_name} is set")
    else:
        st.error(f"{key_name} not set")
        st.caption(f"Export it in your terminal before running:\n`export {key_name}=...`")

    # Start button
    can_start = uploaded_file is not None and has_key and not st.session_state.review_running
    start_clicked = st.button(
        "Start Review",
        type="primary",
        disabled=not can_start,
        use_container_width=True,
    )


# ── Main area ────────────────────────────────────────────────────────────────

if not uploaded_file and not st.session_state.result_md:
    st.markdown(
        """
        ## Welcome

        Upload a PDF in the sidebar to get started.

        This tool runs a multi-agent peer review using AI models.
        Each review includes:

        1. **Review Panel** — Multiple specialized reviewers critique the document
        2. **Independence Audit** — A challenge pass detects biases
        3. **Synthesizer** — An SRO, Editor, or Panel Chair issues a decision
        4. **Author Revision** — The PI/Author revises based on feedback
        """
    )

elif start_clicked:
    # ── Run the review ───────────────────────────────────────────────────
    st.session_state.review_running = True
    st.session_state.output_buffer = ""
    st.session_state.result_md = None
    st.session_state.result_pdf_bytes = None

    # Write uploaded PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_pdf_path = tmp.name

    # Output path
    pdf_stem = Path(uploaded_file.name).stem
    tmp_output = tempfile.mktemp(suffix=".md", prefix=f"{pdf_stem}_review_")

    # Streaming output area
    output_placeholder = st.empty()
    status_placeholder = st.empty()

    def on_chunk(text: str) -> None:
        st.session_state.output_buffer += text
        output_placeholder.code(st.session_state.output_buffer[-5000:], language=None)

    def on_status(msg: str) -> None:
        status_placeholder.info(msg)

    try:
        from review_engine.engine import run_review

        result_md = run_review(
            config=config,
            backend=backend,
            pdf_path=tmp_pdf_path,
            model=model,
            max_rounds=max_rounds,
            output_path=tmp_output,
            on_chunk=on_chunk,
            on_status=on_status,
        )

        st.session_state.result_md = result_md

        # Read generated PDF if it exists
        pdf_output = Path(tmp_output).with_suffix(".pdf")
        if pdf_output.exists():
            st.session_state.result_pdf_bytes = pdf_output.read_bytes()

        st.session_state.review_running = False
        status_placeholder.success("Review complete!")

    except Exception as e:
        st.session_state.review_running = False
        st.error(f"Review failed: {e}")

    finally:
        # Cleanup temp PDF
        try:
            os.unlink(tmp_pdf_path)
        except OSError:
            pass

    st.rerun()

elif st.session_state.result_md:
    # ── Show results ─────────────────────────────────────────────────────
    st.success("Review Complete")

    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Markdown",
            data=st.session_state.result_md,
            file_name="review.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        if st.session_state.result_pdf_bytes:
            st.download_button(
                label="Download PDF",
                data=st.session_state.result_pdf_bytes,
                file_name="review.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.button("PDF not available", disabled=True, use_container_width=True)

    # Show the review content
    with st.expander("Full Review Output", expanded=True):
        st.markdown(st.session_state.result_md)

    # Reset button
    if st.button("Start New Review"):
        st.session_state.result_md = None
        st.session_state.result_pdf_bytes = None
        st.session_state.output_buffer = ""
        st.rerun()

elif st.session_state.review_running:
    st.info("Review in progress... Please wait.")
    st.code(st.session_state.output_buffer[-5000:] if st.session_state.output_buffer else "Starting...", language=None)
