"""
Review orchestrator — the heart of the unified engine.

Handles all three iteration modes (iterative, single_pass, fixed_rounds)
through one run_review() entry point.  All domain-specific logic comes
from ReviewConfig; all API calls go through the backend module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from review_engine.config import ReviewConfig
from review_engine.helpers import (
    banner,
    format_critiques,
    convert_to_pdf,
    build_combined_reviewer_system,
    build_challenge_system,
    load_prompt,
    parse_combined_critiques,
    merge_challenge_addenda,
    parse_decision,
    extract_revision,
    build_revised_text,
)
from _shared.text import generate_toc


# ── ReviewSession ───────────────────────────────────────────────────────────

class ReviewSession:
    """Encapsulates all state for a single review run.

    Replaces the ad-hoc closures and scattered variables with explicit
    state and well-defined methods.
    """

    def __init__(
        self,
        config: ReviewConfig,
        backend: str,
        pdf_path: str,
        model: str,
        max_rounds: int,
        output_path: str,
        on_chunk: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.pdf_path = pdf_path
        self.model = model
        self.max_rounds = max_rounds
        self.output_path = output_path
        self.out = on_chunk or (lambda t: print(t, end="", flush=True))
        self.status = on_status or (lambda s: print(s))
        self.log: list[str] = []

        # Backend-specific state (set during _setup_backend)
        self._client = None
        self._pdf_b64: str | None = None
        self._file_uri: str | None = None
        self._file_name: str | None = None
        self._manuscript_text: str | None = None  # local: PDF→markdown

        # Backend function references (set during _setup_backend)
        self._build_fn = None
        self._stream_fn = None
        self._cleanup_fn = None

        # System prompts (loaded in run())
        self._combined_system = ""
        self._challenge_system = ""
        self._synthesizer_system = ""
        self._author_system = ""

    # ── Logging helpers ──────────────────────────────────────────────────

    def record(self, heading: str, content: str, level: int = 3) -> None:
        self.log.append(f"{'#' * level} {heading}\n\n{content}\n")

    def emit_decision(self, decision: str, suffix: str = "") -> None:
        label = self.config.decision.decision_label
        tag = f"{label}{suffix}: {decision}"
        self.out(f"\n{'='*72}\n  {tag}\n{'='*72}\n\n")
        self.log.append(f"### {tag}\n")

    def print_revision(
        self,
        sections: dict[str, str | None],
        rnd: int,
        decision: str = "",
    ) -> None:
        label = f"Revised — Round {rnd}" + (f" ({decision})" if decision else "")
        self.out(banner(label, "-"))
        for rs in self.config.revision_sections:
            content = sections.get(rs.key)
            if content:
                self.out(f"-- {rs.heading.lstrip('#').strip()} --\n\n")
                preview = content[:800]
                self.out(preview)
                if len(content) > 800:
                    self.out(
                        f"\n[... {len(content) - 800} more chars"
                        " -- see output file ...]\n"
                    )
                self.out("\n\n")

    # ── Backend abstraction ──────────────────────────────────────────────

    def _setup_backend(self) -> None:
        pdf = Path(self.pdf_path)
        if self.backend == "claude":
            from review_engine.backends.claude import (
                make_client, encode_pdf, build_content, stream_agent,
            )
            self._client = make_client()
            self._build_fn = build_content
            self._stream_fn = stream_agent
            self.out(banner(f"Loading {pdf.name}  [model: {self.model}]", "="))
            self._pdf_b64 = encode_pdf(self.pdf_path)
            self.out(f"  Encoded {pdf.stat().st_size // 1024} KB -> base64  OK\n\n")
        elif self.backend == "gemini":
            from review_engine.backends.gemini import (
                make_client, upload_pdf, cleanup_file, build_parts, stream_agent,
            )
            self._client = make_client()
            self._build_fn = build_parts
            self._stream_fn = stream_agent
            self._cleanup_fn = cleanup_file
            self.out(banner(f"Uploading {pdf.name}  [model: {self.model}]", "="))
            self._file_uri, self._file_name = upload_pdf(self._client, self.pdf_path)
            self.out(f"  Uploaded -> {self._file_uri}\n\n")
        elif self.backend == "local":
            from review_engine.backends.local import (
                make_client, convert_pdf, build_content, stream_agent,
                cleanup,
            )
            self.out(banner(f"Loading model for {pdf.name}", "="))
            self._client = make_client(self.model)
            self.out(
                f"  Model loaded: {self._client.model_id}"
                f" ({self._client.backend_type})\n"
                f"  Device: {self._client.device}\n"
            )
            self._build_fn = build_content
            self._stream_fn = stream_agent
            self._cleanup_fn = cleanup
            self.out(banner(f"Converting {pdf.name} to markdown", "="))
            self._manuscript_text = convert_pdf(self.pdf_path)
            self.out(
                f"  Converted {pdf.stat().st_size // 1024} KB PDF -> "
                f"{len(self._manuscript_text)} chars markdown  OK\n\n"
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

    def _cleanup_backend(self) -> None:
        if self.backend == "gemini" and self._file_name and self._cleanup_fn:
            self._cleanup_fn(self._client, self._file_name)
            self.out(f"\nCleaned up remote file: {self._file_name}\n")
        elif self.backend == "local" and self._client and self._cleanup_fn:
            self._cleanup_fn(self._client)
            self.out("\nModel resources released.\n")

    def build(
        self,
        prompt: str,
        *,
        text: str | None = None,
        use_pdf: bool = True,
    ) -> list:
        heading = self.config.revised_document_heading
        if self.backend == "claude":
            return self._build_fn(
                prompt,
                pdf_base64=self._pdf_b64 if use_pdf else None,
                manuscript_text=text,
                revised_heading=heading,
            )
        elif self.backend == "gemini":
            return self._build_fn(
                prompt,
                file_uri=self._file_uri if use_pdf else None,
                manuscript_text=text,
                revised_heading=heading,
            )
        else:  # local
            # Round 1: use PDF-converted markdown; Round 2+: use revised text
            ms_text = text if text is not None else (
                self._manuscript_text if use_pdf else None
            )
            return self._build_fn(
                prompt,
                manuscript_text=ms_text,
                revised_heading=heading,
            )

    def stream(self, sys_prompt: str, content_blocks: list, label: str) -> str:
        if self.backend == "claude":
            result, self.model = self._stream_fn(
                self._client, self.model, sys_prompt, content_blocks, label,
                on_chunk=self.out,
            )
        elif self.backend == "gemini":
            result, self.model = self._stream_fn(
                self._client, self.model, sys_prompt, content_blocks, label,
                fallback_chain=self.config.gemini_fallback_chain,
                on_chunk=self.out,
            )
        else:  # local
            result, self.model = self._stream_fn(
                self._client, self.model, sys_prompt, content_blocks, label,
                on_chunk=self.out,
            )
        return result

    # ── Shared review cycle ──────────────────────────────────────────────

    def review_cycle(
        self,
        *,
        build_synth_prompt: Callable[[str], str],
        synth_label: str,
        synth_record_heading: str,
        text: str | None = None,
        use_pdf: bool = True,
        status_prefix: str = "",
        record_suffix: str = "",
        reviewer_extra: str = "",
    ) -> tuple[str, str, str]:
        """Run reviewers -> challenge pass -> synthesizer.

        Args:
            build_synth_prompt:   Callable that receives critiques_text and
                                  returns the full synthesizer user prompt.
            synth_label:          Label for the synthesizer stream.
            synth_record_heading: Heading for the synthesizer record entry.
            text:                 Revised manuscript text (None = use PDF).
            use_pdf:              Whether to attach the original PDF.
            status_prefix:        Prefix for status messages (e.g. "Round 1: ").
            record_suffix:        Suffix for record headings (e.g. "Round 2").
            reviewer_extra:       Extra text before {noun} in reviewer prompt
                                  (e.g. "REVISED ").

        Returns:
            (critiques_text, synth_out, decision)
        """
        cfg = self.config
        n = cfg.reviewer_count_word
        noun = cfg.document_noun

        # ── Combined reviewers ───────────────────────────────────────────
        self.status(f"{status_prefix}Running {n} reviewers...")
        reviewer_instruction = (
            f"Review the {reviewer_extra}{noun} above. Produce all {n} "
            "reviewer critiques between their sentinel markers, "
            "following each role's instructions exactly."
        )
        panel_label = f"Review Panel ({n} reviewers -- combined)"
        if record_suffix:
            panel_label = f"Review Panel ({record_suffix} -- {n} reviewers combined)"

        combined_out = self.stream(
            self._combined_system,
            self.build(reviewer_instruction, text=text, use_pdf=use_pdf),
            panel_label,
        )
        critiques = parse_combined_critiques(cfg, combined_out)
        for name, txt in critiques.items():
            heading = f"{name} ({record_suffix})" if record_suffix else name
            self.record(heading, txt)

        # ── Challenge pass ───────────────────────────────────────────────
        self.status(f"{status_prefix}Running independence audit...")
        challenge_label = "Independence Auditor -- Challenge Pass"
        if record_suffix:
            challenge_label += f" ({record_suffix})"

        challenge_out = self.stream(
            self._challenge_system,
            self.build(
                f"Below are {n} reviewer critiques of the same {noun}. "
                "Evaluate each for independence biases and produce addenda "
                "as instructed.\n\n"
                + format_critiques(critiques),
                text=text, use_pdf=use_pdf,
            ),
            challenge_label,
        )
        critiques = merge_challenge_addenda(cfg, critiques, challenge_out)
        for name, txt in critiques.items():
            if "### Independence Auditor" in txt:
                parts = [name]
                if record_suffix:
                    parts.append(record_suffix)
                parts.append("with addendum")
                self.record(f"{name} ({', '.join(parts[1:])})", txt)
        critiques_text = format_critiques(critiques)

        # ── Synthesizer ──────────────────────────────────────────────────
        self.status(f"{status_prefix}Running {cfg.synthesizer_role_name}...")
        synth_prompt = build_synth_prompt(critiques_text)
        synth_out = self.stream(
            self._synthesizer_system,
            self.build(synth_prompt, text=text, use_pdf=use_pdf),
            synth_label,
        )
        self.record(synth_record_heading, synth_out, level=2)

        decision = parse_decision(cfg, synth_out)
        return critiques_text, synth_out, decision

    # ── Shared revision helpers ──────────────────────────────────────────

    @property
    def marker_instructions(self) -> str:
        return "\n".join(
            f"Place the {rs.heading.lstrip('#').strip()} between "
            f"<<<{rs.sentinel_prefix}_START>>> and <<<{rs.sentinel_prefix}_END>>>."
            for rs in self.config.revision_sections
        )

    @property
    def revision_sentinel_info(self) -> str:
        return "  ".join(
            f"{rs.heading.lstrip('#').strip()}: "
            f"<<<{rs.sentinel_prefix}_START>>> ... <<<{rs.sentinel_prefix}_END>>>"
            for rs in self.config.revision_sections
        )

    # ── Save ─────────────────────────────────────────────────────────────

    def _build_final_markdown(self) -> str:
        """Assemble the final markdown document with TOC.

        Inserts a Table of Contents between the metadata section and
        the review content.  The TOC is generated from all headings in
        the assembled document.
        """
        # Add completion timestamp to metadata table
        end_time = datetime.now(timezone.utc)
        elapsed = end_time - self._start_time
        minutes = int(elapsed.total_seconds() // 60)
        seconds = int(elapsed.total_seconds() % 60)

        # Build the document without TOC first so we can scan headings
        raw = "\n".join(self.log)

        toc_body = generate_toc(raw, max_depth=2)
        if toc_body:
            toc_section = f"## Table of Contents\n\n{toc_body}\n"
            # Insert TOC after metadata (log[0] = title, log[1] = metadata)
            parts = list(self.log)
            parts.insert(2, toc_section)
        else:
            parts = list(self.log)

        # Append duration to the metadata table (inside log[1])
        parts[1] = parts[1].rstrip("\n") + (
            f"| **Completed** | {end_time.strftime('%Y-%m-%d %H:%M UTC')} |\n"
            f"| **Duration** | {minutes}m {seconds}s |\n"
        )

        return "\n".join(parts)

    def save(self) -> None:
        final_md = self._build_final_markdown()
        Path(self.output_path).write_text(final_md, encoding="utf-8")
        self.out(f"Output saved -> {self.output_path}\n")
        pdf_out = str(Path(self.output_path).with_suffix(".pdf"))
        if convert_to_pdf(
            self.output_path, pdf_out,
            header_title=self.config.display_name,
        ):
            self.out(f"PDF saved    -> {pdf_out}\n")
        else:
            self.out("[PDF] Could not export PDF. Install fpdf2 or weasyprint.\n")

    # ── Main entry point ─────────────────────────────────────────────────

    def run(self) -> str:
        pdf = Path(self.pdf_path)
        if not pdf.exists():
            raise FileNotFoundError(f"File not found: {self.pdf_path}")

        # Build system prompts
        cfg = self.config
        self._combined_system = build_combined_reviewer_system(cfg)
        self._challenge_system = build_challenge_system(cfg)
        self._synthesizer_system = load_prompt(
            cfg, cfg.synthesizer_prompt_file,
            scoring=cfg.synthesizer_include_scoring,
        )
        self._author_system = load_prompt(cfg, cfg.author_prompt_file)

        self._setup_backend()
        self._start_time = datetime.now(timezone.utc)

        _backend_labels = {"claude": "Claude", "gemini": "Gemini", "local": "Local"}
        backend_label = _backend_labels.get(self.backend, self.backend.title())
        self.log.append(
            f"# {cfg.display_name} ({backend_label}): {pdf.name}\n"
        )

        # Metadata section — inserted right after the title.
        # The TOC will be injected between this and the review content
        # at save-time (see _insert_toc()).
        mode_label = cfg.iteration.mode.replace("_", " ").title()
        self.log.append(
            f"## Review Metadata\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Backend** | {backend_label} |\n"
            f"| **Model** | `{self.model}` |\n"
            f"| **Review Type** | {cfg.display_name} |\n"
            f"| **Mode** | {mode_label} |\n"
            f"| **Max Rounds** | {self.max_rounds} |\n"
            f"| **Document** | {pdf.name} |\n"
            f"| **Started** | {self._start_time.strftime('%Y-%m-%d %H:%M UTC')} |\n"
        )

        try:
            if cfg.iteration.mode == "single_pass":
                self._run_single_pass()
            elif cfg.iteration.mode == "fixed_rounds":
                self._run_fixed_rounds()
            else:
                self._run_iterative()
        finally:
            self._cleanup_backend()

        self.save()
        return Path(self.output_path).read_text(encoding="utf-8")

    # ── Single Pass (Foundation) ─────────────────────────────────────────

    def _run_single_pass(self) -> None:
        cfg = self.config
        noun = cfg.document_noun

        critiques_text, synth_out, decision = self.review_cycle(
            build_synth_prompt=lambda ct: (
                "The following critiques have been submitted by the review panel "
                f"for this {noun}.\n\n"
                f"{ct}\n\n"
                "Synthesize these into a Panel Recommendation Letter following "
                "your output format exactly."
            ),
            synth_label=f"{cfg.synthesizer_role_name} -- Recommendation Letter",
            synth_record_heading=(
                f"{cfg.synthesizer_role_name} -- Recommendation Letter"
            ),
        )

        self.emit_decision(decision)

        # Revision
        self.status(f"Running {cfg.author_role_name} revision...")
        author_out = self.stream(
            self._author_system,
            self.build(
                f"You have received the review panel critiques and the "
                f"{cfg.synthesizer_role_name}'s Recommendation Letter below. "
                f"Revise your application accordingly.\n\n"
                f"{critiques_text}\n\n"
                f"{cfg.synthesizer_role_name} Recommendation Letter:\n{synth_out}\n\n"
                f"Follow your output format exactly, including the sentinel markers.\n"
                f"{self.marker_instructions}",
            ),
            f"{cfg.author_role_name} -- Revision",
        )
        self.record(f"{cfg.author_role_name} -- Response & Revision", author_out)

        sections = extract_revision(cfg, author_out)
        revised = build_revised_text(cfg, sections, fallback=author_out)
        self.print_revision(sections, rnd=1)
        self.record("Revised Application", revised, level=2)

    # ── Iterative (NIH) ──────────────────────────────────────────────────

    def _run_iterative(self) -> None:
        cfg = self.config
        noun = cfg.document_noun
        current_text: str | None = None
        use_pdf = True
        prev_synth_out = ""

        for rnd in range(1, self.max_rounds + 1):
            self.out(banner(f"REVIEW ROUND {rnd} / {self.max_rounds}", "="))
            self.log.append(f"## Review Round {rnd}\n")

            prefix = f"Round {rnd}: "

            def _build_synth(ct: str) -> str:
                prompt = (
                    "The following critiques have been submitted by the review "
                    f"panel for this {noun}.\n\n"
                    f"{ct}\n\n"
                    "Synthesize these into an official NIH Summary Statement "
                    "and issue your fundability decision following your output "
                    "format."
                )
                if prev_synth_out:
                    prompt = (
                        f"Previous Summary Statement (prior submission):\n"
                        f"{prev_synth_out}\n\n" + prompt
                    )
                return prompt

            critiques_text, synth_out, decision = self.review_cycle(
                build_synth_prompt=_build_synth,
                synth_label=(
                    f"{cfg.synthesizer_role_name} -- Summary Statement"
                ),
                synth_record_heading=(
                    f"Summary Statement ({cfg.synthesizer_role_name})"
                ),
                text=current_text,
                use_pdf=use_pdf,
                status_prefix=prefix,
            )
            prev_synth_out = synth_out

            self.emit_decision(decision)

            # Terminal: positive
            if decision in cfg.decision.terminal_positive:
                self.out(banner(f"APPLICATION DEEMED {decision.upper()}", "="))
                if current_text:
                    self.log.append(
                        f"\n# Final Revised Application\n\n{current_text}\n"
                    )
                else:
                    self.log.append(
                        "\n# Final Revised Application\n\n"
                        "[Original PDF funded without revision]\n"
                    )
                break

            # Terminal: negative
            if decision in cfg.decision.terminal_negative:
                self.out(banner(f"NOT RECOMMENDED ({decision})", "="))
                self.log.append(
                    f"\n### Application {decision} after Round {rnd}\n"
                )
                if current_text:
                    self.log.append(
                        f"\n# Last Revised Application\n\n{current_text}\n"
                    )
                break

            # Max rounds
            if rnd == self.max_rounds:
                self.out(banner(
                    f"MAX ROUNDS ({self.max_rounds}) REACHED"
                    " -- not yet fundable", "="
                ))
                self.log.append(
                    f"\n### Stopped: Max Rounds ({self.max_rounds}) Reached — Final Decision: {decision}\n"
                )
                if current_text:
                    self.log.append(
                        f"\n# Last Revised Application\n\n{current_text}\n"
                    )
                break

            # PI Revision
            self.status(f"{prefix}Running {cfg.author_role_name} revision...")
            pi_out = self.stream(
                self._author_system,
                self.build(
                    f"You have received the review panel critiques and the "
                    f"official Summary Statement below. Revise your application "
                    f"accordingly.\n\n"
                    f"{critiques_text}\n\n"
                    f"Summary Statement:\n{synth_out}\n\n"
                    f"Follow your output format exactly, including all sentinel "
                    f"markers.\n{self.marker_instructions}",
                    text=current_text, use_pdf=use_pdf,
                ),
                f"{cfg.author_role_name} -- Revision",
            )
            self.record(
                f"{cfg.author_role_name} Response & Revised Application",
                pi_out,
            )

            sections = extract_revision(cfg, pi_out)
            current_text = build_revised_text(cfg, sections, fallback=pi_out)
            use_pdf = False
            self.print_revision(sections, rnd=rnd)
            self.record(
                f"Revised Application -- Round {rnd}", current_text, level=2
            )

    # ── Fixed Rounds (Journal) ───────────────────────────────────────────

    def _run_fixed_rounds(self) -> None:
        cfg = self.config
        noun = cfg.document_noun
        n = cfg.reviewer_count_word

        # ── ROUND 1 ─────────────────────────────────────────────────────
        self.out(banner("REVIEW ROUND 1 -- Initial Submission", "="))
        self.log.append("## Review Round 1 -- Initial Submission\n")

        critiques_text, synth_out, decision = self.review_cycle(
            build_synth_prompt=lambda ct: (
                f"The following {n} peer-reviewer critiques were submitted "
                f"for this {noun}.\n\n"
                f"{ct}\n\n"
                "Synthesize these into an official editorial decision letter "
                "following your output format. This is Round 1; valid decisions "
                "are Accept, Minor Revision, or Major Revision."
            ),
            synth_label=(
                f"{cfg.synthesizer_role_name} -- Decision Letter (Round 1)"
            ),
            synth_record_heading="Editorial Decision Letter (Round 1)",
            status_prefix="Round 1: ",
        )
        prev_synth_out = synth_out

        self.emit_decision(decision, suffix=" (Round 1)")

        # Build author prompt for R1
        if decision == "Accept":
            ms_start = f"<<<{cfg.revision_sections[-1].sentinel_prefix}_START>>>"
            ms_end = f"<<<{cfg.revision_sections[-1].sentinel_prefix}_END>>>"
            author_prompt = (
                f"The editor has accepted your {noun}. Congratulations.\n\n"
                "Produce a final polished version. Apply only light "
                "copyediting, clarify passages flagged by reviewers, correct "
                "minor issues. No response letter is required.\n\n"
                f"Reviewer feedback for reference:\n{critiques_text}\n\n"
                f"Editorial Decision Letter:\n{synth_out}\n\n"
                f"Place the final {noun} between {ms_start} and {ms_end}."
            )
        else:
            author_prompt = (
                "You have received the following reviewer critiques and "
                f"editorial decision letter for your {noun}.\n\n"
                f"Reviewer Critiques:\n{critiques_text}\n\n"
                f"Editorial Decision Letter:\n{synth_out}\n\n"
                "Produce:\n"
                "  1. A point-by-point response letter.\n"
                f"  2. A full revised {noun}.\n\n"
                f"  {self.revision_sentinel_info}"
            )

        self.status(f"Round 1: Running {cfg.author_role_name} revision...")
        author_label = (
            "Final Polish" if decision == "Accept"
            else "Revision (Round 1)"
        )
        author_out = self.stream(
            self._author_system,
            self.build(author_prompt),
            f"{cfg.author_role_name} -- {author_label}",
        )
        record_label = (
            "Final Polished Manuscript" if decision == "Accept"
            else "Response & Revised Manuscript (Round 1)"
        )
        self.record(f"{cfg.author_role_name} -- {record_label}", author_out)

        sections = extract_revision(cfg, author_out)
        current_text = build_revised_text(cfg, sections, fallback=author_out)
        self.print_revision(sections, rnd=1, decision=decision)

        if decision == "Accept":
            self.out(banner("MANUSCRIPT ACCEPTED -- Round 1", "="))
            self.log.append(f"\n# Final Manuscript\n\n{current_text}\n")
            return

        # ── ROUND 2 ─────────────────────────────────────────────────────
        self.out(banner(
            f"REVIEW ROUND 2 -- Revised Submission ({decision})", "="
        ))
        self.log.append(
            f"## Review Round 2 -- Revised Submission ({decision})\n"
        )

        r2_decisions = (
            cfg.iteration.r2_valid_decisions
            or "Accept, Minor Revision, or Reject"
        )

        critiques_text_r2, synth_out_r2, decision_r2 = self.review_cycle(
            build_synth_prompt=lambda ct: (
                f"Round 1 Decision Letter (for context):\n{prev_synth_out}\n\n"
                f"Round 2 Reviewer Critiques:\n{ct}\n\n"
                "Synthesize into a final editorial decision letter. "
                f"This is Round 2 (final); valid decisions are {r2_decisions}."
            ),
            synth_label=(
                f"{cfg.synthesizer_role_name}"
                " -- Final Decision Letter (Round 2)"
            ),
            synth_record_heading=(
                "Editorial Decision Letter (Round 2 -- Final)"
            ),
            text=current_text,
            use_pdf=False,
            status_prefix="Round 2: ",
            record_suffix="Round 2",
            reviewer_extra="REVISED ",
        )

        self.emit_decision(decision_r2, suffix=" (Round 2 -- Final)")

        # R2 author prompt
        if decision_r2 == "Reject":
            ms_start = f"<<<{cfg.revision_sections[-1].sentinel_prefix}_START>>>"
            ms_end = f"<<<{cfg.revision_sections[-1].sentinel_prefix}_END>>>"
            author_prompt_r2 = (
                f"The editor has rejected your {noun} after Round 2. "
                "Produce a fully revised version addressing all concerns, "
                "for future resubmission elsewhere.\n\n"
                f"Round 2 Reviewer Critiques:\n{critiques_text_r2}\n\n"
                f"Final Editorial Decision Letter:\n{synth_out_r2}\n\n"
                f"Place the final {noun} between {ms_start} and {ms_end}. "
                "No response letter required."
            )
        else:
            author_prompt_r2 = (
                "You have received the final editorial decision.\n\n"
                f"Round 2 Reviewer Critiques:\n{critiques_text_r2}\n\n"
                f"Final Editorial Decision Letter:\n{synth_out_r2}\n\n"
                "Produce:\n"
                "  1. A point-by-point response letter for remaining "
                "concerns.\n"
                f"  2. A final revised {noun}.\n\n"
                f"  {self.revision_sentinel_info}"
            )

        self.status(
            f"Round 2: Running {cfg.author_role_name} final version..."
        )
        author_out_r2 = self.stream(
            self._author_system,
            self.build(author_prompt_r2, text=current_text, use_pdf=False),
            f"{cfg.author_role_name}"
            f" -- Final Version (Round 2 -- {decision_r2})",
        )
        self.record(
            f"{cfg.author_role_name}"
            f" -- Final Version (Round 2 -- {decision_r2})",
            author_out_r2,
        )

        sections_r2 = extract_revision(cfg, author_out_r2)
        final_text = build_revised_text(
            cfg, sections_r2, fallback=author_out_r2
        )
        self.print_revision(sections_r2, rnd=2, decision=decision_r2)
        self.log.append(
            f"\n# Final Manuscript (after Round 2)\n\n{final_text}\n"
        )

        if decision_r2 in ("Accept", "Minor Revision"):
            label = (
                "ACCEPTED" if decision_r2 == "Accept"
                else "CONDITIONALLY ACCEPTED"
            )
            self.out(banner(f"MANUSCRIPT {label} -- Round 2", "="))
        else:
            self.out(banner(
                "MANUSCRIPT REJECTED"
                " -- Final version saved for resubmission", "="
            ))


# ── Public API (thin wrapper) ───────────────────────────────────────────────

def run_review(
    config: ReviewConfig,
    backend: str,
    pdf_path: str,
    model: str,
    max_rounds: int,
    output_path: str,
    on_chunk: Callable[[str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> str:
    """Run a complete multi-agent review.

    Args:
        config:      ReviewConfig for the chosen review type
        backend:     "claude" or "gemini"
        pdf_path:    Path to the input PDF
        model:       Model ID to use
        max_rounds:  Maximum number of review rounds
        output_path: Where to write the markdown output
        on_chunk:    Streaming text callback (None = stdout)
        on_status:   Status/progress callback (None = stdout)

    Returns:
        The markdown output as a string.
    """
    session = ReviewSession(
        config=config,
        backend=backend,
        pdf_path=pdf_path,
        model=model,
        max_rounds=max_rounds,
        output_path=output_path,
        on_chunk=on_chunk,
        on_status=on_status,
    )
    return session.run()
