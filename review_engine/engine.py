"""
Review orchestrator — the heart of the unified engine.

Handles all three iteration modes (iterative, single_pass, fixed_rounds)
through one run_review() entry point.  All domain-specific logic comes
from ReviewConfig; all API calls go through the backend module.
"""

from __future__ import annotations

import sys
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save(log: list[str], output_path: str,
          on_chunk: Callable[[str], None] | None = None) -> None:
    out = on_chunk or (lambda t: print(t, end="", flush=True))
    Path(output_path).write_text("\n".join(log), encoding="utf-8")
    out(f"Output saved -> {output_path}\n")
    pdf_out = str(Path(output_path).with_suffix(".pdf"))
    if convert_to_pdf(output_path, pdf_out):
        out(f"PDF saved    -> {pdf_out}\n")
    else:
        out("[PDF] Could not export PDF. Install with: pip install reportlab\n")


def _print_revision(
    config: ReviewConfig,
    sections: dict[str, str | None],
    rnd: int,
    decision: str = "",
    on_chunk: Callable[[str], None] | None = None,
) -> None:
    out = on_chunk or (lambda t: print(t, end="", flush=True))
    label = f"Revised — Round {rnd}" + (f" ({decision})" if decision else "")
    out(banner(label, "-"))
    for rs in config.revision_sections:
        content = sections.get(rs.key)
        if content:
            out(f"-- {rs.heading.lstrip('#').strip()} --\n\n")
            preview = content[:800]
            out(preview)
            if len(content) > 800:
                out(f"\n[... {len(content) - 800} more chars -- see output file ...]\n")
            out("\n\n")


# ── Orchestrator ─────────────────────────────────────────────────────────────

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
    """
    Run a complete multi-agent review.

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
    out = on_chunk or (lambda t: print(t, end="", flush=True))
    status = on_status or (lambda s: print(s))

    log: list[str] = []

    def record(heading: str, content: str, level: int = 3) -> None:
        log.append(f"{'#' * level} {heading}\n\n{content}\n")

    # ── Validate ─────────────────────────────────────────────────────────
    pdf = Path(pdf_path)
    if not pdf.exists():
        sys.exit(f"Error: file not found -- {pdf_path}")

    # ── Build system prompts ─────────────────────────────────────────────
    combined_system = build_combined_reviewer_system(config)
    challenge_system = build_challenge_system(config)
    synthesizer_system = load_prompt(
        config, config.synthesizer_prompt_file,
        scoring=config.synthesizer_include_scoring,
    )
    author_system = load_prompt(config, config.author_prompt_file)

    # ── Backend setup ────────────────────────────────────────────────────
    if backend == "claude":
        from review_engine.backends.claude import (
            make_client, encode_pdf, build_content, stream_agent,
        )
        client = make_client()
        out(banner(f"Loading {pdf.name}  [model: {model}]", "="))
        pdf_b64 = encode_pdf(pdf_path)
        out(f"  Encoded {pdf.stat().st_size // 1024} KB -> base64  OK\n\n")

        # State
        current_pdf_b64: str | None = pdf_b64
        file_uri: str | None = None
        file_name: str | None = None

        def _build(prompt, *, revised_heading=config.revised_document_heading,
                   text=None, use_pdf=True):
            return build_content(
                prompt,
                pdf_base64=current_pdf_b64 if use_pdf else None,
                manuscript_text=text,
                revised_heading=revised_heading,
            )

        def _stream(sys_prompt, content_blocks, label):
            nonlocal model
            result, model = stream_agent(
                client, model, sys_prompt, content_blocks, label,
                on_chunk=on_chunk,
            )
            return result

    else:  # gemini
        from review_engine.backends.gemini import (
            make_client, upload_pdf, cleanup_file, build_parts, stream_agent,
        )
        client = make_client()
        out(banner(f"Uploading {pdf.name}  [model: {model}]", "="))
        file_uri, file_name = upload_pdf(client, pdf_path)
        out(f"  Uploaded -> {file_uri}\n\n")

        current_pdf_b64 = None

        def _build(prompt, *, revised_heading=config.revised_document_heading,
                   text=None, use_pdf=True):
            return build_parts(
                prompt,
                file_uri=file_uri if use_pdf else None,
                manuscript_text=text,
                revised_heading=revised_heading,
            )

        def _stream(sys_prompt, content_blocks, label):
            nonlocal model
            result, model = stream_agent(
                client, model, sys_prompt, content_blocks, label,
                fallback_chain=config.gemini_fallback_chain,
                on_chunk=on_chunk,
            )
            return result

    # ── Header ───────────────────────────────────────────────────────────
    backend_label = "Claude" if backend == "claude" else "Gemini"
    log.append(
        f"# {config.display_name} ({backend_label}): {pdf.name}\n\n"
        f"Model: `{model}`\n"
    )

    # ── Dispatch by iteration mode ───────────────────────────────────────
    try:
        if config.iteration.mode == "single_pass":
            _run_single_pass(
                config, log, record, _build, _stream, out, status,
                combined_system, challenge_system, synthesizer_system,
                author_system,
            )
        elif config.iteration.mode == "fixed_rounds":
            _run_fixed_rounds(
                config, log, record, _build, _stream, out, status,
                combined_system, challenge_system, synthesizer_system,
                author_system, max_rounds,
            )
        else:  # "iterative"
            _run_iterative(
                config, log, record, _build, _stream, out, status,
                combined_system, challenge_system, synthesizer_system,
                author_system, max_rounds,
            )
    finally:
        # Gemini cleanup
        if backend == "gemini" and file_name:
            cleanup_file(client, file_name)
            out(f"\nCleaned up remote file: {file_name}\n")

    _save(log, output_path, on_chunk=on_chunk)
    return "\n".join(log)


# ── Single Pass (Foundation) ─────────────────────────────────────────────────

def _run_single_pass(config, log, record, _build, _stream, out, status,
                     combined_system, challenge_system, synthesizer_system,
                     author_system):
    """Foundation-style: reviewers -> audit -> synthesizer -> revision, once."""
    n = config.reviewer_count_word
    noun = config.document_noun

    status(f"Running {n} reviewers...")

    # Combined reviewers
    combined_out = _stream(
        combined_system,
        _build(
            f"Review the {noun} above. Produce all {n} "
            "reviewer critiques between their sentinel markers, "
            "following each role's instructions exactly.",
        ),
        f"Review Panel ({n} reviewers -- combined)",
    )
    critiques = parse_combined_critiques(config, combined_out)
    for name, text in critiques.items():
        record(name, text)

    # Challenge pass
    status("Running independence audit...")
    challenge_out = _stream(
        challenge_system,
        _build(
            f"Below are {n} reviewer critiques of the same {noun}. "
            "Evaluate each for independence biases and produce addenda as instructed.\n\n"
            + format_critiques(critiques),
        ),
        "Independence Auditor -- Challenge Pass",
    )
    critiques = merge_challenge_addenda(config, critiques, challenge_out)
    for name, text in critiques.items():
        if "### Independence Auditor" in text:
            record(f"{name} (with addendum)", text)
    critiques_text = format_critiques(critiques)

    # Synthesizer
    status(f"Running {config.synthesizer_role_name}...")
    synth_out = _stream(
        synthesizer_system,
        _build(
            "The following critiques have been submitted by the review panel "
            f"for this {noun}.\n\n"
            f"{critiques_text}\n\n"
            f"Synthesize these into a Panel Recommendation Letter following "
            "your output format exactly.",
        ),
        f"{config.synthesizer_role_name} -- Recommendation Letter",
    )
    record(f"{config.synthesizer_role_name} -- Recommendation Letter", synth_out, level=2)

    decision = parse_decision(config, synth_out)
    out(f"\n{'='*72}\n  {config.decision.decision_label}: {decision}\n{'='*72}\n\n")
    log.append(f"**{config.decision.decision_label}: {decision}**\n")

    # Revision
    status(f"Running {config.author_role_name} revision...")
    markers = config.revision_markers
    marker_instructions = "\n".join(
        f"Place the {rs.heading.lstrip('#').strip()} between "
        f"<<<{rs.sentinel_prefix}_START>>> and <<<{rs.sentinel_prefix}_END>>>."
        for rs in config.revision_sections
    )
    author_out = _stream(
        author_system,
        _build(
            f"You have received the review panel critiques and the "
            f"{config.synthesizer_role_name}'s Recommendation Letter below. "
            f"Revise your application accordingly.\n\n"
            f"{critiques_text}\n\n"
            f"{config.synthesizer_role_name} Recommendation Letter:\n{synth_out}\n\n"
            f"Follow your output format exactly, including the sentinel markers.\n"
            f"{marker_instructions}",
        ),
        f"{config.author_role_name} -- Revision",
    )
    record(f"{config.author_role_name} -- Response & Revision", author_out)

    sections = extract_revision(config, author_out)
    revised = build_revised_text(config, sections, fallback=author_out)
    _print_revision(config, sections, rnd=1)
    record("Revised Application", revised, level=2)


# ── Iterative (NIH) ─────────────────────────────────────────────────────────

def _run_iterative(config, log, record, _build, _stream, out, status,
                   combined_system, challenge_system, synthesizer_system,
                   author_system, max_rounds):
    """NIH-style: loop rounds until fundable, NRFC, or max_rounds."""
    n = config.reviewer_count_word
    noun = config.document_noun
    current_text: str | None = None
    use_pdf = True
    prev_synth_out = ""

    for rnd in range(1, max_rounds + 1):
        out(banner(f"REVIEW ROUND {rnd} / {max_rounds}", "="))
        log.append(f"## Review Round {rnd}\n")

        # Combined reviewers
        status(f"Round {rnd}: Running {n} reviewers...")
        combined_out = _stream(
            combined_system,
            _build(
                f"Review the {noun} above. Produce all {n} "
                "reviewer critiques between their sentinel markers, "
                "following each role's instructions exactly.",
                text=current_text, use_pdf=use_pdf,
            ),
            f"Review Panel ({n} reviewers -- combined)",
        )
        critiques = parse_combined_critiques(config, combined_out)
        for name, text in critiques.items():
            record(name, text)

        # Challenge pass
        status(f"Round {rnd}: Running independence audit...")
        challenge_out = _stream(
            challenge_system,
            _build(
                f"Below are {n} reviewer critiques of the same {noun}. "
                "Evaluate each for independence biases and produce addenda as instructed.\n\n"
                + format_critiques(critiques),
                text=current_text, use_pdf=use_pdf,
            ),
            "Independence Auditor -- Challenge Pass",
        )
        critiques = merge_challenge_addenda(config, critiques, challenge_out)
        for name, text in critiques.items():
            if "### Independence Auditor" in text:
                record(f"{name} (with addendum)", text)
        critiques_text = format_critiques(critiques)

        # Synthesizer
        status(f"Round {rnd}: Running {config.synthesizer_role_name}...")
        synth_prompt = (
            "The following critiques have been submitted by the review panel "
            f"for this {noun}.\n\n"
            f"{critiques_text}\n\n"
            "Synthesize these into an official NIH Summary Statement and issue "
            "your fundability decision following your output format."
        )
        if prev_synth_out:
            synth_prompt = (
                f"Previous Summary Statement (prior submission):\n{prev_synth_out}\n\n"
                + synth_prompt
            )
        synth_out = _stream(
            synthesizer_system,
            _build(synth_prompt, text=current_text, use_pdf=use_pdf),
            f"{config.synthesizer_role_name} -- Summary Statement",
        )
        prev_synth_out = synth_out
        record(f"Summary Statement ({config.synthesizer_role_name})", synth_out)

        decision = parse_decision(config, synth_out)
        out(f"\n{'='*72}\n  {config.decision.decision_label}: {decision}\n{'='*72}\n\n")
        log.append(f"**{config.decision.decision_label}: {decision}**\n")

        # Terminal: positive
        if decision in config.decision.terminal_positive:
            out(banner(f"APPLICATION DEEMED {decision.upper()}", "="))
            if current_text:
                log.append(f"\n# Final Revised Application\n\n{current_text}\n")
            else:
                log.append(f"\n# Final Revised Application\n\n[Original PDF funded without revision: {Path(pdf_path).name if 'pdf_path' in dir() else 'document'}]\n")
            break

        # Terminal: negative
        if decision in config.decision.terminal_negative:
            out(banner(f"NOT RECOMMENDED ({decision})", "="))
            log.append(f"\n**Application {decision} after round {rnd}.**\n")
            if current_text:
                log.append(f"\n# Last Revised Application\n\n{current_text}\n")
            break

        # Max rounds
        if rnd == max_rounds:
            out(banner(f"MAX ROUNDS ({max_rounds}) REACHED -- not yet fundable", "="))
            log.append(
                f"\n**Stopped: max rounds ({max_rounds}) reached. "
                f"Final decision: {decision}**\n"
            )
            if current_text:
                log.append(f"\n# Last Revised Application\n\n{current_text}\n")
            break

        # PI Revision
        status(f"Round {rnd}: Running {config.author_role_name} revision...")
        markers = config.revision_markers
        marker_instructions = "\n".join(
            f"Place the {rs.heading.lstrip('#').strip()} between "
            f"<<<{rs.sentinel_prefix}_START>>> and <<<{rs.sentinel_prefix}_END>>>."
            for rs in config.revision_sections
        )
        pi_out = _stream(
            author_system,
            _build(
                f"You have received the review panel critiques and the official "
                f"Summary Statement below. Revise your application accordingly.\n\n"
                f"{critiques_text}\n\n"
                f"Summary Statement:\n{synth_out}\n\n"
                f"Follow your output format exactly, including all sentinel markers.\n"
                f"{marker_instructions}",
                text=current_text, use_pdf=use_pdf,
            ),
            f"{config.author_role_name} -- Revision",
        )
        record(f"{config.author_role_name} Response & Revised Application", pi_out)

        sections = extract_revision(config, pi_out)
        current_text = build_revised_text(config, sections, fallback=pi_out)
        use_pdf = False
        _print_revision(config, sections, rnd=rnd, on_chunk=out)
        record(f"Revised Application -- Round {rnd}", current_text, level=2)


# ── Fixed Rounds (Journal) ──────────────────────────────────────────────────

def _run_fixed_rounds(config, log, record, _build, _stream, out, status,
                      combined_system, challenge_system, synthesizer_system,
                      author_system, max_rounds):
    """Journal-style: up to 2 rounds with round-specific decision logic."""
    n = config.reviewer_count_word
    noun = config.document_noun
    current_text: str | None = None
    use_pdf = True
    prev_synth_out = ""

    # ── ROUND 1 ──────────────────────────────────────────────────────────
    out(banner("REVIEW ROUND 1 -- Initial Submission", "="))
    log.append("## Review Round 1 -- Initial Submission\n")

    status("Round 1: Running reviewers...")
    combined_out = _stream(
        combined_system,
        _build(
            f"Review the {noun} above. Produce all {n} reviewer critiques "
            "between their sentinel markers, following each role's instructions exactly.",
        ),
        f"Review Panel ({n} reviewers -- combined)",
    )
    critiques = parse_combined_critiques(config, combined_out)
    for name, text in critiques.items():
        record(name, text)

    status("Round 1: Running independence audit...")
    challenge_out = _stream(
        challenge_system,
        _build(
            f"Below are {n} reviewer critiques of the same {noun}. "
            "Evaluate each for independence biases and produce addenda as instructed.\n\n"
            + format_critiques(critiques),
        ),
        "Independence Auditor -- Challenge Pass",
    )
    critiques = merge_challenge_addenda(config, critiques, challenge_out)
    for name, text in critiques.items():
        if "### Independence Auditor" in text:
            record(f"{name} (with addendum)", text)
    critiques_text = format_critiques(critiques)

    status(f"Round 1: Running {config.synthesizer_role_name}...")
    synth_out = _stream(
        synthesizer_system,
        _build(
            f"The following {n} peer-reviewer critiques were submitted "
            f"for this {noun}.\n\n"
            f"{critiques_text}\n\n"
            "Synthesize these into an official editorial decision letter following "
            "your output format. This is Round 1; valid decisions are Accept, "
            "Minor Revision, or Major Revision.",
        ),
        f"{config.synthesizer_role_name} -- Decision Letter (Round 1)",
    )
    prev_synth_out = synth_out
    record("Editorial Decision Letter (Round 1)", synth_out)

    decision = parse_decision(config, synth_out)
    out(f"\n{'='*72}\n  {config.decision.decision_label} (Round 1): {decision}\n{'='*72}\n\n")
    log.append(f"**{config.decision.decision_label} (Round 1): {decision}**\n")

    # Build author prompt for R1
    markers = config.revision_markers
    revision_sentinel_info = "  ".join(
        f"{rs.heading.lstrip('#').strip()}: "
        f"<<<{rs.sentinel_prefix}_START>>> ... <<<{rs.sentinel_prefix}_END>>>"
        for rs in config.revision_sections
    )

    if decision == "Accept":
        # Accepted at R1: light polish only
        ms_start = f"<<<{config.revision_sections[-1].sentinel_prefix}_START>>>"
        ms_end = f"<<<{config.revision_sections[-1].sentinel_prefix}_END>>>"
        author_prompt = (
            f"The editor has accepted your {noun}. Congratulations.\n\n"
            "Produce a final polished version. Apply only light copyediting, "
            "clarify passages flagged by reviewers, correct minor issues. "
            "No response letter is required.\n\n"
            f"Reviewer feedback for reference:\n{critiques_text}\n\n"
            f"Editorial Decision Letter:\n{synth_out}\n\n"
            f"Place the final {noun} between {ms_start} and {ms_end}."
        )
    else:
        author_prompt = (
            "You have received the following reviewer critiques and editorial "
            f"decision letter for your {noun}.\n\n"
            f"Reviewer Critiques:\n{critiques_text}\n\n"
            f"Editorial Decision Letter:\n{synth_out}\n\n"
            "Produce:\n"
            "  1. A point-by-point response letter.\n"
            f"  2. A full revised {noun}.\n\n"
            f"  {revision_sentinel_info}"
        )

    status(f"Round 1: Running {config.author_role_name} revision...")
    author_out = _stream(
        author_system,
        _build(author_prompt),
        f"{config.author_role_name} -- {'Final Polish' if decision == 'Accept' else 'Revision (Round 1)'}",
    )
    label_r1 = "Final Polished Manuscript" if decision == "Accept" else "Response & Revised Manuscript (Round 1)"
    record(f"{config.author_role_name} -- {label_r1}", author_out)

    sections = extract_revision(config, author_out)
    current_text = build_revised_text(config, sections, fallback=author_out)
    use_pdf = False
    _print_revision(config, sections, rnd=1, decision=decision, on_chunk=out)

    if decision == "Accept":
        out(banner("MANUSCRIPT ACCEPTED -- Round 1", "="))
        log.append(f"\n# Final Manuscript\n\n{current_text}\n")
        return

    # ── ROUND 2 ──────────────────────────────────────────────────────────
    out(banner(f"REVIEW ROUND 2 -- Revised Submission ({decision})", "="))
    log.append(f"## Review Round 2 -- Revised Submission ({decision})\n")

    status("Round 2: Running reviewers...")
    combined_out_r2 = _stream(
        combined_system,
        _build(
            f"Review the REVISED {noun} below. Produce all {n} reviewer "
            "critiques between their sentinel markers. This is a revised "
            "submission -- assess whether prior concerns were addressed.",
            text=current_text, use_pdf=False,
        ),
        f"Review Panel (Round 2 -- {n} reviewers combined)",
    )
    critiques_r2 = parse_combined_critiques(config, combined_out_r2)
    for name, text in critiques_r2.items():
        record(f"{name} (Round 2)", text)

    status("Round 2: Running independence audit...")
    challenge_out_r2 = _stream(
        challenge_system,
        _build(
            f"Below are {n} Round 2 reviewer critiques of a revised {noun}. "
            "Evaluate each for independence biases and produce addenda as instructed.\n\n"
            + format_critiques(critiques_r2),
            text=current_text, use_pdf=False,
        ),
        "Independence Auditor -- Challenge Pass (Round 2)",
    )
    critiques_r2 = merge_challenge_addenda(config, critiques_r2, challenge_out_r2)
    for name, text in critiques_r2.items():
        if "### Independence Auditor" in text:
            record(f"{name} (Round 2, with addendum)", text)
    critiques_text_r2 = format_critiques(critiques_r2)

    r2_decisions = config.iteration.r2_valid_decisions or "Accept, Minor Revision, or Reject"
    status(f"Round 2: Running {config.synthesizer_role_name}...")
    synth_out_r2 = _stream(
        synthesizer_system,
        _build(
            f"Round 1 Decision Letter (for context):\n{prev_synth_out}\n\n"
            f"Round 2 Reviewer Critiques:\n{critiques_text_r2}\n\n"
            "Synthesize into a final editorial decision letter. "
            f"This is Round 2 (final); valid decisions are {r2_decisions}.",
            text=current_text, use_pdf=False,
        ),
        f"{config.synthesizer_role_name} -- Final Decision Letter (Round 2)",
    )
    record("Editorial Decision Letter (Round 2 -- Final)", synth_out_r2)

    decision_r2 = parse_decision(config, synth_out_r2)
    out(f"\n{'='*72}\n  {config.decision.decision_label} (Round 2 -- Final): {decision_r2}\n{'='*72}\n\n")
    log.append(f"**{config.decision.decision_label} (Round 2 -- Final): {decision_r2}**\n")

    # R2 author prompt
    if decision_r2 == "Reject":
        ms_start = f"<<<{config.revision_sections[-1].sentinel_prefix}_START>>>"
        ms_end = f"<<<{config.revision_sections[-1].sentinel_prefix}_END>>>"
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
            "  1. A point-by-point response letter for remaining concerns.\n"
            f"  2. A final revised {noun}.\n\n"
            f"  {revision_sentinel_info}"
        )

    status(f"Round 2: Running {config.author_role_name} final version...")
    author_out_r2 = _stream(
        author_system,
        _build(author_prompt_r2, text=current_text, use_pdf=False),
        f"{config.author_role_name} -- Final Version (Round 2 -- {decision_r2})",
    )
    record(f"{config.author_role_name} -- Final Version (Round 2 -- {decision_r2})", author_out_r2)

    sections_r2 = extract_revision(config, author_out_r2)
    final_text = build_revised_text(config, sections_r2, fallback=author_out_r2)
    _print_revision(config, sections_r2, rnd=2, decision=decision_r2, on_chunk=out)
    log.append(f"\n# Final Manuscript (after Round 2)\n\n{final_text}\n")

    if decision_r2 in ("Accept", "Minor Revision"):
        label = "ACCEPTED" if decision_r2 == "Accept" else "CONDITIONALLY ACCEPTED"
        out(banner(f"MANUSCRIPT {label} -- Round 2", "="))
    else:
        out(banner("MANUSCRIPT REJECTED -- Final version saved for resubmission", "="))
