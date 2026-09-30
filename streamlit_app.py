"""
Streamlit GUI for the Multi-Agent Peer Review Engine.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# ── Page config (set first to prevent model loading during imports) ──────────

st.set_page_config(
    page_title="Multi-Agent Peer Review",
    page_icon="📝",
    layout="wide",
)

from review_engine.config import ensure_types_loaded, get_config, REVIEW_TYPES
from review_engine.backends.claude import (
    MODEL_FALLBACK_CHAIN as CLAUDE_MODELS,
    DEFAULT_MODEL as CLAUDE_DEFAULT,
)
from review_engine.backends.gemini import (
    DEFAULT_FALLBACK_CHAIN as GEMINI_MODELS,
    DEFAULT_MODEL as GEMINI_DEFAULT,
)
from review_engine.backends.local import (
    DEFAULT_MODEL as LOCAL_DEFAULT,
    _MARKITDOWN_OK,
    _LLAMA_CPP_OK,
    _TRANSFORMERS_OK,
)

# ── Load review type configs ─────────────────────────────────────────────────
ensure_types_loaded()


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
        key="review_type_select",
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
        options=["Claude", "Gemini", "Local"],
        horizontal=True,
        key="backend_radio",
    ).lower()

    # Model
    if backend == "claude":
        model_options = CLAUDE_MODELS
        default_model = config.claude_default_model
        default_idx = (
            model_options.index(default_model) if default_model in model_options else 0
        )
        model = st.selectbox(
            "Model",
            options=model_options,
            index=default_idx,
            key="claude_model_select",
            help="Select the Claude model to use.",
        )
    elif backend == "gemini":
        model_options = config.gemini_fallback_chain
        default_model = config.gemini_default_model
        default_idx = (
            model_options.index(default_model) if default_model in model_options else 0
        )
        model = st.selectbox(
            "Model",
            options=model_options,
            index=default_idx,
            key="gemini_model_select",
            help="Select the Gemini model to use.",
        )
    else:  # local
        # Preset local models
        preset_models = [
            "google/gemma-4-E4B-it",
            "google/gemma-4-31B",
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen2.5-7B",
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-2-7b",
            "gpt2",
            "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF",
            "google/gemma-3-27b-it",
            "google/gemma-3-12b-it",
            "google/gemma-3-4b-it",
            "Custom model...",
        ]
        
        model_choice = st.selectbox(
            "Model",
            options=preset_models,
            index=0,
            key="local_model_select",
            help="Select a preset model or choose 'Custom model...' to enter your own.",
        )
        
        if model_choice == "Custom model...":
            model = st.text_input(
                "Custom Model ID",
                value="",
                placeholder="Enter HuggingFace repo ID or local path",
                help=(
                    "HuggingFace repo ID or local file path.\n\n"
                    "GGUF models use llama-cpp-python; "
                    "standard HF models use transformers."
                ),
                key="local_custom_model_input",
            )
        else:
            model = model_choice

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

    # API key / dependency status
    if backend == "claude":
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        key_name = "ANTHROPIC_API_KEY"
        if has_key:
            st.success(f"{key_name} is set")
        else:
            st.error(f"{key_name} not set")
            st.caption(f"Export it in your terminal before running:\n`export {key_name}=...`")
        ready = has_key

    elif backend == "gemini":
        has_key = bool(os.environ.get("GEMINI_API_KEY"))
        key_name = "GEMINI_API_KEY"
        if has_key:
            st.success(f"{key_name} is set")
        else:
            st.error(f"{key_name} not set")
            st.caption(f"Export it in your terminal before running:\n`export {key_name}=...`")
        ready = has_key

    else:  # local
        has_converter = _MARKITDOWN_OK
        has_inference = _LLAMA_CPP_OK or _TRANSFORMERS_OK

        if has_converter and has_inference:
            st.success("Local dependencies ready")
        else:
            if not has_converter:
                st.error("markitdown not installed")
                st.caption("`pip install markitdown`")
            if not has_inference:
                st.error("No inference backend")
                st.caption(
                    "`pip install llama-cpp-python` (GGUF)\n\n"
                    "`pip install transformers torch` (HF)"
                )
        ready = has_converter and has_inference

        with st.expander("Dependency status"):
            st.markdown(
                f"- **markitdown**: {'✅' if _MARKITDOWN_OK else '❌'}\n"
                f"- **llama-cpp-python**: {'✅' if _LLAMA_CPP_OK else '❌'}\n"
                f"- **transformers**: {'✅' if _TRANSFORMERS_OK else '❌'}"
            )

    # Start button
    can_start = uploaded_file is not None and ready and not st.session_state.review_running
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
    # ── Prepare the review — save params and rerun so the UI disables the button
    st.session_state.review_running = True
    st.session_state.output_buffer = ""
    st.session_state.result_md = None
    st.session_state.result_pdf_bytes = None

    # Write uploaded PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        st.session_state._tmp_pdf_path = tmp.name

    pdf_stem = Path(uploaded_file.name).stem
    st.session_state._tmp_output = tempfile.mktemp(suffix=".md", prefix=f"{pdf_stem}_review_")
    st.session_state._run_config = config
    st.session_state._run_backend = backend
    st.session_state._run_model = model
    st.session_state._run_max_rounds = max_rounds

    st.rerun()

elif st.session_state.review_running:
    # ── Execute the review (button is already disabled on this rerun) ────
    tmp_pdf_path = st.session_state.get("_tmp_pdf_path")
    tmp_output = st.session_state.get("_tmp_output")
    run_config = st.session_state.get("_run_config", config)
    run_backend = st.session_state.get("_run_backend", backend)
    run_model = st.session_state.get("_run_model", model)
    run_max_rounds = st.session_state.get("_run_max_rounds", max_rounds)

    if tmp_pdf_path:
        # Streaming output area
        progress_bar = st.progress(0, text="Starting review...")
        status_placeholder = st.empty()
        output_placeholder = st.empty()

        steps_seen: set[str] = set()
        step_labels = [
            "Review Panel", "Independence Auditor", "Panel Chair",
            "Project Director", "SRO", "Editor", "Recommendation",
            "Revision", "Uploading", "Loading",
        ]

        def on_chunk(text: str) -> None:
            st.session_state.output_buffer += text
            output_placeholder.code(st.session_state.output_buffer[-5000:], language=None)

        def on_status(msg: str) -> None:
            # Update progress bar based on steps detected
            for i, label in enumerate(step_labels):
                if label.lower() in msg.lower():
                    steps_seen.add(label)
            pct = min(0.95, len(steps_seen) / max(len(step_labels), 1))
            progress_bar.progress(pct, text=msg)
            status_placeholder.info(msg)

        try:
            from review_engine.engine import run_review

            result_md = run_review(
                config=run_config,
                backend=run_backend,
                pdf_path=tmp_pdf_path,
                model=run_model,
                max_rounds=run_max_rounds,
                output_path=tmp_output,
                on_chunk=on_chunk,
                on_status=on_status,
            )

            st.session_state.result_md = result_md

            # Read generated PDF if it exists
            pdf_output = Path(tmp_output).with_suffix(".pdf")
            if pdf_output.exists():
                st.session_state.result_pdf_bytes = pdf_output.read_bytes()

            progress_bar.progress(1.0, text="Review complete!")
            status_placeholder.success("Review complete!")

        except Exception as e:
            st.session_state.review_running = False
            # Cleanup temp PDF
            try:
                os.unlink(tmp_pdf_path)
            except OSError:
                pass
            
            # Display error with full traceback and keep it visible
            st.error("❌ Review failed")
            st.exception(e)
            st.info("Please check the error above and try again with:")
            st.markdown("- A different model\n- More GPU memory (if available)\n- A smaller model like 'gpt2' for testing")
            
            if st.button("🔄 Try Again", key="retry_button"):
                st.session_state.result_md = None
                st.session_state.result_pdf_bytes = None
                st.session_state.output_buffer = ""
                st.rerun()
        else:
            # Only rerun on success
            st.rerun()
    else:
        st.session_state.review_running = False
        st.error("No PDF found to review. Please try again.")
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
