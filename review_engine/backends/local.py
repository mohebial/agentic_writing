"""
Local open-source LLM backend using HuggingFace models.

Supports two inference paths:
  - GGUF models via llama-cpp-python  (recommended for quantised models)
  - Standard HF models via transformers (AutoModelForCausalLM)

PDF handling uses Microsoft's markitdown for PDF-to-markdown conversion,
since local LLMs cannot process binary PDF files natively.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from review_engine.helpers import banner

# ── Dependency checks ────────────────────────────────────────────────────────

try:
    import torch
    _TORCH_OK = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_OK = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    _TRANSFORMERS_OK = True
except ImportError:
    AutoModelForCausalLM = AutoTokenizer = TextIteratorStreamer = None
    _TRANSFORMERS_OK = False

try:
    from llama_cpp import Llama
    _LLAMA_CPP_OK = True
except ImportError:
    Llama = None  # type: ignore[assignment]
    _LLAMA_CPP_OK = False

try:
    from markitdown import MarkItDown
    _MARKITDOWN_OK = True
except ImportError:
    MarkItDown = None  # type: ignore[assignment]
    _MARKITDOWN_OK = False

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
MAX_NEW_TOKENS = 16384
DEFAULT_N_GPU_LAYERS = -1   # offload all layers to GPU
DEFAULT_N_CTX = 8192        # context window size


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class LocalModel:
    """Container for a loaded local model and its metadata."""

    model: Any                     # Llama instance or HF model
    tokenizer: Any | None          # AutoTokenizer (HF path only)
    backend_type: str              # "llama_cpp" or "transformers"
    model_id: str                  # HF repo ID or local path
    device: str                    # "cuda", "mps", "cpu"


# ── Public API ───────────────────────────────────────────────────────────────

class LocalSetupError(RuntimeError):
    """Raised when the local backend cannot be initialised."""


def validate_startup() -> None:
    """Check that required dependencies are installed.

    Unlike Claude/Gemini, no API key is needed — only local libraries.
    """
    if not _MARKITDOWN_OK:
        raise LocalSetupError(
            "Missing dependency 'markitdown'.\n"
            "Install it with: pip install markitdown"
        )
    if not _LLAMA_CPP_OK and not _TRANSFORMERS_OK:
        raise LocalSetupError(
            "No inference backend found.\n"
            "Install one of:\n"
            "  pip install llama-cpp-python    (for GGUF models)\n"
            "  pip install transformers torch  (for safetensors models)"
        )


# ── Device / Memory helpers ──────────────────────────────────────────────────

def _detect_device() -> str:
    """Detect the best available compute device."""
    if not _TORCH_OK:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _check_memory(device: str, model_id: str) -> None:
    """Warn if GPU memory looks insufficient."""
    if device == "cuda" and _TORCH_OK:
        free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        total_gb = torch.cuda.mem_get_info()[1] / (1024 ** 3)
        if free_gb < 8:
            warnings.warn(
                f"Only {free_gb:.1f}/{total_gb:.1f} GB GPU memory free. "
                f"Loading '{model_id}' may fail. "
                "Consider a smaller quantisation or fewer GPU layers."
            )


# ── GGUF detection ───────────────────────────────────────────────────────────

def _is_gguf_model(model_id: str) -> bool:
    """Determine if *model_id* points to a GGUF model."""
    lower = model_id.lower()
    if lower.endswith(".gguf"):
        return True
    if "gguf" in lower:
        return True
    p = Path(model_id)
    if p.is_dir():
        return any(p.glob("*.gguf"))
    return False


# ── Model loading ────────────────────────────────────────────────────────────

def make_client(
    model_id: str = DEFAULT_MODEL,
    *,
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
    n_ctx: int = DEFAULT_N_CTX,
    gguf_filename: str | None = None,
) -> LocalModel:
    """Load a local model and return a :class:`LocalModel` container.

    Args:
        model_id:      HuggingFace repo ID or local filesystem path.
        n_gpu_layers:  Layers to offload to GPU (-1 = all, 0 = CPU only).
        n_ctx:         Context window size (llama-cpp models only).
        gguf_filename: Specific ``.gguf`` filename inside a HF repo.
                       ``None`` auto-selects the first match.
    """
    device = _detect_device()
    _check_memory(device, model_id)

    if _is_gguf_model(model_id):
        if not _LLAMA_CPP_OK:
            raise LocalSetupError(
                f"Model '{model_id}' appears to be GGUF but "
                "llama-cpp-python is not installed.\n"
                "Install with: pip install llama-cpp-python"
            )
        return _load_llama_cpp(model_id, n_gpu_layers, n_ctx, gguf_filename)

    if not _TRANSFORMERS_OK or not _TORCH_OK:
        raise LocalSetupError(
            f"Model '{model_id}' requires transformers and torch.\n"
            "Install with: pip install transformers torch"
        )
    return _load_transformers(model_id, device)


def _load_llama_cpp(
    model_id: str,
    n_gpu_layers: int,
    n_ctx: int,
    gguf_filename: str | None,
) -> LocalModel:
    """Load a GGUF model via llama-cpp-python."""
    local_path = Path(model_id)
    if local_path.is_file() and model_id.lower().endswith(".gguf"):
        model = Llama(
            model_path=str(local_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )
    else:
        # HuggingFace repo — llama-cpp-python can download from HF Hub
        model = Llama.from_pretrained(
            repo_id=model_id,
            filename=gguf_filename or "*.gguf",
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )

    device = (
        "cuda"
        if n_gpu_layers != 0 and _TORCH_OK and torch.cuda.is_available()
        else "cpu"
    )
    return LocalModel(
        model=model,
        tokenizer=None,
        backend_type="llama_cpp",
        model_id=model_id,
        device=device,
    )


def _load_transformers(model_id: str, device: str) -> LocalModel:
    """Load a standard HF model via transformers."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
    )
    return LocalModel(
        model=model,
        tokenizer=tokenizer,
        backend_type="transformers",
        model_id=model_id,
        device=device,
    )


# ── PDF conversion ───────────────────────────────────────────────────────────

def convert_pdf(pdf_path: str) -> str:
    """Convert a PDF to markdown text using markitdown.

    Returns the markdown text content of the PDF.
    """
    converter = MarkItDown()
    result = converter.convert(pdf_path)
    return result.text_content


# ── Content building ─────────────────────────────────────────────────────────

def build_content(
    prompt: str,
    manuscript_text: str | None = None,
    revised_heading: str = "Document (Revised)",
) -> list[dict]:
    """Build the message list for a local model.

    Always text-based — no binary PDF attachments.  The PDF has already
    been converted to markdown via :func:`convert_pdf`.

    Round 1 (manuscript_text from PDF conversion): document + prompt
    Round 2+ (manuscript_text from revision):      same pattern
    """
    if manuscript_text:
        combined = f"## {revised_heading}\n\n{manuscript_text}\n\n{prompt}"
    else:
        combined = prompt
    return [{"role": "user", "content": combined}]


# ── Streaming inference ──────────────────────────────────────────────────────

def stream_agent(
    client: LocalModel,
    model: str,
    system: str,
    content: list[dict],
    label: str,
    on_chunk: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """Stream a response from the local model.

    Returns ``(full_text, model_used)``.

    No fallback chain — the local model either works or raises.
    No retry logic — local inference has no rate limits.
    """
    if on_chunk is None:
        on_chunk = lambda text: print(text, end="", flush=True)

    header = banner(f"{label}  [model: {client.model_id}]")
    on_chunk(header)

    messages = [{"role": "system", "content": system}] + content

    if client.backend_type == "llama_cpp":
        collected = _stream_llama_cpp(client.model, messages, on_chunk)
    else:
        collected = _stream_transformers(
            client.model, client.tokenizer, messages, on_chunk,
        )

    on_chunk("\n")
    return "".join(collected), model


def _stream_llama_cpp(
    model: Any,
    messages: list[dict],
    on_chunk: Callable[[str], None],
) -> list[str]:
    """Stream tokens from a llama-cpp model."""
    collected: list[str] = []
    response = model.create_chat_completion(
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
        stream=True,
    )
    for chunk in response:
        delta = chunk["choices"][0].get("delta", {})
        text = delta.get("content", "")
        if text:
            on_chunk(text)
            collected.append(text)
    return collected


def _stream_transformers(
    model: Any,
    tokenizer: Any,
    messages: list[dict],
    on_chunk: Callable[[str], None],
) -> list[str]:
    """Stream tokens from a transformers model via TextIteratorStreamer."""
    import threading

    # Apply chat template if the tokenizer supports it, otherwise
    # concatenate with simple role tags.
    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        input_text = "\n\n".join(
            f"[{m['role']}]\n{m['content']}" for m in messages
        )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 1.0,
        "do_sample": True,
        "streamer": streamer,
    }

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    collected: list[str] = []
    for text in streamer:
        if text:
            on_chunk(text)
            collected.append(text)

    thread.join()
    return collected


# ── Cleanup ──────────────────────────────────────────────────────────────────

def cleanup(client: LocalModel) -> None:
    """Release model resources and free VRAM."""
    if client.backend_type == "transformers" and _TORCH_OK:
        del client.model
        del client.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
