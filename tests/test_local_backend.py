"""Tests for review_engine.backends.local — Local LLM backend."""

from __future__ import annotations

import pytest

import review_engine.backends.local as local_mod
from review_engine.backends.local import (
    LocalSetupError,
    _detect_device,
    _is_gguf_model,
    build_content,
    validate_startup,
)


# ── validate_startup ────────────────────────────────────────────────────────

class TestValidateStartup:
    def test_missing_markitdown(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_MARKITDOWN_OK", False)
        monkeypatch.setattr(local_mod, "_LLAMA_CPP_OK", True)
        monkeypatch.setattr(local_mod, "_TRANSFORMERS_OK", True)
        with pytest.raises(LocalSetupError, match="markitdown"):
            validate_startup()

    def test_missing_all_inference(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_MARKITDOWN_OK", True)
        monkeypatch.setattr(local_mod, "_LLAMA_CPP_OK", False)
        monkeypatch.setattr(local_mod, "_TRANSFORMERS_OK", False)
        with pytest.raises(LocalSetupError, match="No inference backend"):
            validate_startup()

    def test_ok_with_llama_cpp(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_MARKITDOWN_OK", True)
        monkeypatch.setattr(local_mod, "_LLAMA_CPP_OK", True)
        monkeypatch.setattr(local_mod, "_TRANSFORMERS_OK", False)
        validate_startup()  # should not raise

    def test_ok_with_transformers(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_MARKITDOWN_OK", True)
        monkeypatch.setattr(local_mod, "_LLAMA_CPP_OK", False)
        monkeypatch.setattr(local_mod, "_TRANSFORMERS_OK", True)
        validate_startup()  # should not raise

    def test_ok_with_both(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_MARKITDOWN_OK", True)
        monkeypatch.setattr(local_mod, "_LLAMA_CPP_OK", True)
        monkeypatch.setattr(local_mod, "_TRANSFORMERS_OK", True)
        validate_startup()  # should not raise


# ── _is_gguf_model ──────────────────────────────────────────────────────────

class TestIsGgufModel:
    def test_gguf_extension(self):
        assert _is_gguf_model("/path/to/model.gguf") is True

    def test_gguf_extension_uppercase(self):
        assert _is_gguf_model("/path/to/MODEL.GGUF") is True

    def test_gguf_in_repo_name(self):
        assert _is_gguf_model(
            "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
        ) is True

    def test_standard_hf_model(self):
        assert _is_gguf_model("meta-llama/Llama-3-8B") is False

    def test_empty_string(self):
        assert _is_gguf_model("") is False

    def test_gguf_in_path_component(self):
        assert _is_gguf_model("some-gguf-model/repo") is True


# ── _detect_device ──────────────────────────────────────────────────────────

class TestDetectDevice:
    def test_returns_valid_device(self):
        device = _detect_device()
        assert device in ("cuda", "mps", "cpu")

    def test_no_torch_returns_cpu(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_TORCH_OK", False)
        assert _detect_device() == "cpu"


# ── build_content ───────────────────────────────────────────────────────────

class TestBuildContent:
    def test_prompt_only(self):
        result = build_content("What do you think?")
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "What do you think?"

    def test_with_manuscript(self):
        result = build_content(
            "Review this.",
            manuscript_text="The study found...",
        )
        assert len(result) == 1
        assert "The study found..." in result[0]["content"]
        assert "Review this." in result[0]["content"]
        assert "## Document (Revised)" in result[0]["content"]

    def test_custom_heading(self):
        result = build_content(
            "Review this.",
            manuscript_text="text",
            revised_heading="Revised Manuscript",
        )
        assert "## Revised Manuscript" in result[0]["content"]

    def test_no_manuscript_text_no_heading(self):
        result = build_content("Just a prompt")
        assert "##" not in result[0]["content"]


# ── convert_pdf (skip if markitdown not installed) ──────────────────────────

class TestConvertPdf:
    @pytest.mark.skipif(
        not local_mod._MARKITDOWN_OK,
        reason="markitdown not installed",
    )
    def test_convert_nonexistent_file(self):
        """Ensure convert_pdf raises for a missing file."""
        with pytest.raises(Exception):
            local_mod.convert_pdf("/nonexistent/file.pdf")


# ── LocalModel dataclass ───────────────────────────────────────────────────

class TestLocalModel:
    def test_construction(self):
        from review_engine.backends.local import LocalModel

        lm = LocalModel(
            model="dummy",
            tokenizer=None,
            backend_type="llama_cpp",
            model_id="test/model",
            device="cpu",
        )
        assert lm.backend_type == "llama_cpp"
        assert lm.device == "cpu"
        assert lm.tokenizer is None
