"""
Local open-source LLM backend using HuggingFace models.

Supports three inference paths:
  - GGUF models via llama-cpp-python  (recommended for quantised models)
  - Standard HF models via transformers (AutoModelForCausalLM)
  - OpenAI-compatible API servers (e.g. LM Studio) for models that
    cannot be loaded natively on the current platform

PDF handling uses Microsoft's markitdown for PDF-to-markdown conversion,
since local LLMs cannot process binary PDF files natively.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Disable Rust-based download backends (hf_transfer, hf_xet) which can crash
# mid-download on Windows.  The Python-based fallback is slower but reliable.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Disable torch.compile / dynamo on Windows — triton's JIT compilation
# requires a C compiler that is typically not available.
if os.name == "nt":
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    # Triton needs a C compiler for JIT kernel compilation.  Point it to
    # MSVC's cl.exe if available and CC is not already set.
    if "CC" not in os.environ:
        import glob as _glob
        _cl_matches = _glob.glob(
            r"C:\Program Files*\Microsoft Visual Studio\**\cl.exe",
            recursive=True,
        )
        if _cl_matches:
            os.environ["CC"] = _cl_matches[0]

from review_engine.helpers import banner

# ── Dependency checks (lightweight — no heavy imports at module level) ────────
# We use importlib to check availability without triggering model downloads
# that some versions of transformers / llama-cpp perform on import.

import importlib.util as _ilu

_TORCH_OK = _ilu.find_spec("torch") is not None
_TRANSFORMERS_OK = _ilu.find_spec("transformers") is not None
_LLAMA_CPP_OK = _ilu.find_spec("llama_cpp") is not None
_MARKITDOWN_OK = _ilu.find_spec("markitdown") is not None
_OPENAI_OK = _ilu.find_spec("openai") is not None

del _ilu  # keep module namespace clean

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
MAX_NEW_TOKENS = 16384
DEFAULT_N_GPU_LAYERS = -1   # offload all layers to GPU
DEFAULT_N_CTX = 8192        # context window size
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"  # LM Studio default


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class LocalModel:
    """Container for a loaded local model and its metadata."""

    model: Any                     # Llama instance, HF model, or OpenAI client
    tokenizer: Any | None          # AutoTokenizer (HF path only)
    backend_type: str              # "llama_cpp", "transformers", or "openai_compat"
    model_id: str                  # HF repo ID, local path, or API model name
    device: str                    # "cuda", "mps", "cpu", or "api"


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
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _check_memory(device: str, model_id: str) -> None:
    """Warn if GPU memory looks insufficient."""
    if device == "cuda" and _TORCH_OK:
        import torch
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


def _is_qwen3_model(model_id: str) -> bool:
    """Check if *model_id* is a Qwen3 variant that supports thinking mode."""
    return "qwen3" in model_id.lower()


def _needs_fla(model_id: str) -> bool:
    """Check whether a model requires flash-linear-attention (fla) for inference.

    Models using hybrid linear-attention architectures (e.g. Qwen3.5) import
    fla triton kernels which cannot run on Windows.
    """
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id)
        return getattr(cfg, "model_type", "") in ("qwen3_5",)
    except Exception:
        return False


_GGUF_EQUIVALENTS: dict[str, tuple[str, str]] = {
    # model_type -> (gguf_repo, specific_gguf_filename)
    "qwen3_5": ("unsloth/Qwen3.5-9B-GGUF", "*Q4_K_M.gguf"),
}


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

    # On Windows, models that require fla (flash-linear-attention) cannot use
    # the transformers backend because fla's triton kernels don't compile.
    # Try the LM Studio OpenAI-compatible API first (user likely already has
    # the model loaded), then fall back to a GGUF equivalent via llama-cpp.
    if os.name == "nt" and not _is_gguf_model(model_id) and _needs_fla(model_id):
        lm = _try_lmstudio(model_id)
        if lm is not None:
            return lm

        # LM Studio not available — fall back to GGUF equivalent
        if not _LLAMA_CPP_OK:
            raise LocalSetupError(
                f"Model '{model_id}' requires flash-linear-attention (fla) which "
                "uses triton kernels that cannot compile on Windows.\n"
                "Either:\n"
                "  1. Load the model in LM Studio and leave its local server running, or\n"
                "  2. Install llama-cpp-python to use a GGUF version:\n"
                "     pip install llama-cpp-python"
            )
        try:
            from transformers import AutoConfig
            model_type = AutoConfig.from_pretrained(model_id).model_type
        except Exception:
            model_type = None
        gguf_repo, gguf_glob = _GGUF_EQUIVALENTS.get(
            model_type, (None, None)
        )
        if gguf_repo is None:
            raise LocalSetupError(
                f"Model '{model_id}' requires fla triton kernels which cannot "
                "compile on Windows, and no GGUF equivalent is known.\n"
                "Please load the model in LM Studio or specify a GGUF model directly."
            )
        banner(
            f"  '{model_id}' requires fla (triton) — not supported on Windows.\n"
            f"  LM Studio not detected. Auto-switching to GGUF: {gguf_repo}"
        )
        return _load_llama_cpp(
            gguf_repo, n_gpu_layers, n_ctx,
            gguf_filename or gguf_glob,
        )

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


def _try_lmstudio(model_id: str) -> LocalModel | None:
    """Try to connect to LM Studio's OpenAI-compatible API.

    Returns a :class:`LocalModel` if LM Studio is running and has a model
    loaded, otherwise ``None``.
    """
    if not _OPENAI_OK:
        return None
    try:
        import httpx
        from openai import OpenAI

        # Quick connectivity check (short timeout)
        client = OpenAI(
            base_url=LMSTUDIO_BASE_URL,
            api_key="lm-studio",
            timeout=httpx.Timeout(5.0, connect=2.0),
        )
        models = client.models.list()
        if not models.data:
            return None

        # Use the first loaded model (LM Studio typically has one)
        api_model = models.data[0].id

        # Re-create client with normal timeout for inference
        client = OpenAI(
            base_url=LMSTUDIO_BASE_URL,
            api_key="lm-studio",
        )

        banner(
            f"  '{model_id}' requires fla (triton) — not supported on Windows.\n"
            f"  Connected to LM Studio API -> model: {api_model}"
        )
        return LocalModel(
            model=client,
            tokenizer=None,
            backend_type="openai_compat",
            model_id=api_model,
            device="api",
        )
    except Exception:
        return None


def _load_llama_cpp(
    model_id: str,
    n_gpu_layers: int,
    n_ctx: int,
    gguf_filename: str | None,
) -> LocalModel:
    """Load a GGUF model via llama-cpp-python."""
    from llama_cpp import Llama

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
        if n_gpu_layers != 0 and _TORCH_OK and __import__("torch").cuda.is_available()
        else "cpu"
    )
    return LocalModel(
        model=model,
        tokenizer=None,
        backend_type="llama_cpp",
        model_id=model_id,
        device=device,
    )


def _patch_fla_cpu_context() -> None:
    """Patch fla's ``custom_device_ctx`` so it works when tensors are on CPU.

    ``torch.cpu`` has no ``.device()`` context manager (unlike ``torch.cuda``),
    which crashes fla on Windows when layers spill to CPU.  We replace it with
    a no-op context manager for the CPU case.
    """
    try:
        import fla.utils as _fla_utils
        _orig = _fla_utils.custom_device_ctx

        def _safe_device_ctx(index: int):
            import torch
            if _fla_utils.device_torch_lib is torch.cpu:
                from contextlib import nullcontext
                return nullcontext()
            return _orig(index)

        _fla_utils.custom_device_ctx = _safe_device_ctx
    except Exception:
        pass  # fla not installed — nothing to patch


def _load_transformers(model_id: str, device: str) -> LocalModel:
    """Load a standard HF model via transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _patch_fla_cpu_context()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    load_kwargs: dict[str, Any] = dict(
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # If the model won't fit in VRAM at full precision, use 4-bit quantisation.
    if device == "cuda":
        free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_id)
            # Rough estimate: 2 bytes per param for bf16/fp16
            param_count = getattr(cfg, "num_parameters", None)
            if param_count is None:
                # Fallback heuristic from hidden_size * num_layers
                h = getattr(cfg, "hidden_size", 4096)
                n = getattr(cfg, "num_hidden_layers", 32)
                param_count = h * h * 4 * n  # very rough
            model_gb = (param_count * 2) / (1024 ** 3)
        except Exception:
            model_gb = 0

        if model_gb > free_gb * 0.9:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                )
                banner(f"  4-bit quantisation enabled (model ~{model_gb:.0f} GB, free VRAM ~{free_gb:.0f} GB)")
            except ImportError:
                warnings.warn(
                    f"Model needs ~{model_gb:.0f} GB but only {free_gb:.0f} GB VRAM free. "
                    "Install bitsandbytes for automatic 4-bit quantisation: "
                    "pip install bitsandbytes"
                )

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except ValueError as exc:
        if "does not recognize this architecture" in str(exc) or "model type" in str(exc).lower():
            raise LocalSetupError(
                f"Your version of transformers does not support '{model_id}'.\n"
                "The model architecture is too new. Upgrade with:\n\n"
                "  pip install git+https://github.com/huggingface/transformers.git\n\n"
                f"Original error: {exc}"
            ) from exc
        raise
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
    from markitdown import MarkItDown

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


# ── Qwen3 thinking-mode helpers ──────────────────────────────────────────────

class _ThinkFilter:
    """Filter that strips ``<think>...</think>`` blocks from streamed text.

    Buffers tokens while inside a ``<think>`` block and discards them
    when the closing ``</think>`` tag is found.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._inside_think = False

    def feed(self, text: str) -> str:
        """Process incoming text and return only non-thinking content."""
        self._buffer += text
        output: list[str] = []

        while self._buffer:
            if self._inside_think:
                end_idx = self._buffer.find("</think>")
                if end_idx == -1:
                    # Still inside think block — consume everything
                    self._buffer = ""
                    break
                # Skip past closing tag
                self._buffer = self._buffer[end_idx + len("</think>"):]
                self._inside_think = False
            else:
                start_idx = self._buffer.find("<think>")
                if start_idx == -1:
                    # No think tag — hold back a possible partial match
                    safe = self._safe_emit_length()
                    if safe > 0:
                        output.append(self._buffer[:safe])
                        self._buffer = self._buffer[safe:]
                    break
                else:
                    # Emit content before <think>
                    if start_idx > 0:
                        output.append(self._buffer[:start_idx])
                    self._buffer = self._buffer[start_idx + len("<think>"):]
                    self._inside_think = True

        return "".join(output)

    def flush(self) -> str:
        """Flush any remaining buffered content."""
        if self._inside_think:
            self._buffer = ""
            return ""
        result = self._buffer
        self._buffer = ""
        return result

    def _safe_emit_length(self) -> int:
        """Return the length of buffer that can safely be emitted.

        Holds back a suffix that could be the start of a ``<think>`` tag.
        """
        tag = "<think>"
        buf = self._buffer
        for i in range(1, len(tag)):
            if buf.endswith(tag[:i]):
                return len(buf) - i
        return len(buf)


def _inject_no_think(messages: list[dict]) -> list[dict]:
    """Append ``/no_think`` to the last user message to disable Qwen3 thinking.

    Qwen3 chat templates recognise this directive and skip the
    ``<think>`` generation phase entirely, saving significant compute.
    """
    messages = [m.copy() for m in messages]
    for m in reversed(messages):
        if m["role"] == "user":
            m["content"] = m["content"].rstrip() + "\n/no_think"
            break
    return messages


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

    # Qwen3 models default to a verbose "thinking" mode that generates
    # thousands of hidden tokens before the real answer.  Disable it.
    qwen3 = _is_qwen3_model(client.model_id)
    if qwen3:
        messages = _inject_no_think(messages)

    if client.backend_type == "openai_compat":
        collected = _stream_openai_compat(
            client.model, client.model_id, messages, on_chunk,
            filter_thinking=qwen3,
        )
    elif client.backend_type == "llama_cpp":
        collected = _stream_llama_cpp(
            client.model, messages, on_chunk, filter_thinking=qwen3,
        )
    else:
        collected = _stream_transformers(
            client.model, client.tokenizer, messages, on_chunk,
            filter_thinking=qwen3,
        )

    on_chunk("\n")
    return "".join(collected), model


def _stream_openai_compat(
    client: Any,
    model_id: str,
    messages: list[dict],
    on_chunk: Callable[[str], None],
    *,
    filter_thinking: bool = False,
) -> list[str]:
    """Stream tokens from an OpenAI-compatible API (e.g. LM Studio)."""
    collected: list[str] = []
    think_filter = _ThinkFilter() if filter_thinking else None

    extra_body: dict[str, Any] = {}
    if filter_thinking:
        extra_body["enable_thinking"] = False

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
        stream=True,
        **(dict(extra_body=extra_body) if extra_body else {}),
    )
    for chunk in response:
        delta = chunk.choices[0].delta
        text = delta.content or ""
        if text:
            if think_filter:
                text = think_filter.feed(text)
            if text:
                on_chunk(text)
                collected.append(text)

    if think_filter:
        remaining = think_filter.flush()
        if remaining:
            on_chunk(remaining)
            collected.append(remaining)

    return collected


def _stream_llama_cpp(
    model: Any,
    messages: list[dict],
    on_chunk: Callable[[str], None],
    *,
    filter_thinking: bool = False,
) -> list[str]:
    """Stream tokens from a llama-cpp model."""
    collected: list[str] = []
    think_filter = _ThinkFilter() if filter_thinking else None

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
            if think_filter:
                text = think_filter.feed(text)
            if text:
                on_chunk(text)
                collected.append(text)

    if think_filter:
        remaining = think_filter.flush()
        if remaining:
            on_chunk(remaining)
            collected.append(remaining)

    return collected


def _stream_transformers(
    model: Any,
    tokenizer: Any,
    messages: list[dict],
    on_chunk: Callable[[str], None],
    *,
    filter_thinking: bool = False,
) -> list[str]:
    """Stream tokens from a transformers model via TextIteratorStreamer."""
    import threading
    import time
    from transformers import TextIteratorStreamer

    think_filter = _ThinkFilter() if filter_thinking else None

    # Apply chat template if the tokenizer supports it, otherwise
    # concatenate with simple role tags.
    if hasattr(tokenizer, "apply_chat_template"):
        template_kwargs: dict[str, Any] = dict(
            tokenize=False, add_generation_prompt=True,
        )
        # Qwen3 tokenizers accept enable_thinking to skip the <think> phase.
        if filter_thinking:
            template_kwargs["enable_thinking"] = False
        input_text = tokenizer.apply_chat_template(
            messages, **template_kwargs,
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

    # Show a waiting indicator while the first token is being generated
    first_token = True
    t0 = time.time()
    print("  Generating (waiting for first token", end="", flush=True)
    wait_printed = True

    collected: list[str] = []
    token_count = 0
    last_status = time.time()
    for text in streamer:
        if first_token:
            elapsed = time.time() - t0
            print(f" — {elapsed:.1f}s)\n", flush=True)
            wait_printed = False
            first_token = False
        if text:
            if think_filter:
                text = think_filter.feed(text)
            if text:
                on_chunk(text)
                collected.append(text)
            token_count += 1
            # Print a progress update every 10 seconds
            now = time.time()
            if now - last_status >= 10:
                elapsed = now - t0
                tps = token_count / elapsed if elapsed > 0 else 0
                print(
                    f"\r  [{token_count} tokens, {elapsed:.0f}s, {tps:.1f} tok/s]",
                    end="", flush=True,
                )
                last_status = now

    if wait_printed:
        # Generation finished with no tokens (edge case)
        print(")", flush=True)

    thread.join()

    if think_filter:
        remaining = think_filter.flush()
        if remaining:
            on_chunk(remaining)
            collected.append(remaining)

    elapsed = time.time() - t0
    tps = token_count / elapsed if elapsed > 0 else 0
    print(f"\n  Done: {token_count} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)", flush=True)
    return collected


# ── Cleanup ──────────────────────────────────────────────────────────────────

def cleanup(client: LocalModel) -> None:
    """Release model resources and free VRAM."""
    if client.backend_type == "openai_compat":
        pass  # nothing to release — API client only
    elif client.backend_type == "transformers" and _TORCH_OK:
        import torch
        del client.model
        del client.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
