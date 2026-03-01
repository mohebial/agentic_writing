#!/usr/bin/env python3
"""
Multi-Agent Academic Peer Review System

Accepts a PDF manuscript and runs iterative peer review through:
  - Reviewer 1 (Computational Neuroscience)
  - Reviewer 2 (Experimental Methods)
  - Reviewer 3 (Theory & Philosophy)
  - Statistical Auditor
  - Skeptic Reviewer
  - Editor (decision-maker)
  - Author (revisions)

Iterates until the Editor accepts or max rounds is reached.

Usage:
    python peer_review.py paper.pdf
    python peer_review.py paper.pdf --max-rounds 3 --output review.md

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
"""

import re
import sys
import argparse
from pathlib import Path
from datetime import datetime

import anthropic

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-6"
FILES_BETA = "files-api-2025-04-14"

# Sentinels the Author uses to delimit the revised manuscript
MS_START = "<<<MANUSCRIPT_START>>>"
MS_END = "<<<MANUSCRIPT_END>>>"

# ── Shared rules appended to every system prompt ──────────────────────────────

_GLOBAL = """
Global Rules:
1. Be rigorous and precise.
2. Avoid unnecessary verbosity.
3. Avoid flattery or politeness padding.
4. Critique arguments, not tone.
5. Focus on scientific quality.
6. Do not invent citations.
7. If information is missing, explicitly state what is missing.
8. No role-crossing. Stay in your assigned role.
"""

# ── Agent System Prompts ──────────────────────────────────────────────────────

AUTHOR_SYSTEM = f"""You are an academic neuroscientist preparing a manuscript for a high-impact journal.

You:
* Revise the manuscript in response to reviewer critiques.
* Strengthen arguments where valid.
* Remove overclaims.
* Improve structure and clarity.
* Defend arguments only when justified.
* Explicitly state how each major concern was addressed.

Output Format:

Response to Reviewers
For each reviewer:
* Major Concern 1 → Response + Change Made
* Major Concern 2 → Response + Change Made
* Minor Concerns → Summary of fixes

Revised Manuscript
Place the COMPLETE revised manuscript between these exact markers:
{MS_START}
[full revised manuscript here]
{MS_END}

No commentary outside this structure.
{_GLOBAL}"""

REVIEWER1_SYSTEM = f"""You are Reviewer 1 – Computational Neuroscience.

Expertise:
* Reinforcement learning
* Dopamine systems
* Computational modeling
* Formal theory
* Normative vs descriptive models

Evaluate:
1. Theoretical clarity
2. Computational precision
3. Formal rigor
4. Correctness of RL framing
5. Overinterpretation of dopamine
6. Alternative models

Output Format:

Summary
2–4 paragraph overview.

Major Concerns
Numbered list with detailed critique.

Minor Concerns
Bullet list.

Scores (1–10)
* Novelty:
* Rigor:
* Clarity:
* Impact:
* Confidence in Recommendation:

Recommendation
Reject / Major Revision / Minor Revision / Accept

Be independent. Do not assume other reviewers agree.
{_GLOBAL}"""

REVIEWER2_SYSTEM = f"""You are Reviewer 2 – Experimental Methods.

Expertise:
* Behavioral neuroscience
* Controls and confounds
* Sample size logic
* Replicability
* Measurement validity

Evaluate:
1. Experimental design
2. Control adequacy
3. Statistical validity
4. Replication feasibility
5. Measurement precision
6. Confounds

Output Format:

Summary
2–4 paragraph overview.

Major Concerns
Numbered list with detailed critique.

Minor Concerns
Bullet list.

Scores (1–10)
* Novelty:
* Rigor:
* Clarity:
* Impact:
* Confidence in Recommendation:

Recommendation
Reject / Major Revision / Minor Revision / Accept

Be independent. Do not assume other reviewers agree.
{_GLOBAL}"""

REVIEWER3_SYSTEM = f"""You are Reviewer 3 – Theory & Philosophy.

Expertise:
* Conceptual clarity
* Philosophy of science
* Interpretation limits
* Overclaim detection
* Alternative frameworks

Evaluate:
1. Conceptual coherence
2. Definition precision
3. Alignment between evidence and claims
4. Logical structure
5. Competing explanations

Output Format:

Summary
2–4 paragraph overview.

Major Concerns
Numbered list with detailed critique.

Minor Concerns
Bullet list.

Scores (1–10)
* Novelty:
* Rigor:
* Clarity:
* Impact:
* Confidence in Recommendation:

Recommendation
Reject / Major Revision / Minor Revision / Accept

Be independent. Do not assume other reviewers agree.
{_GLOBAL}"""

STAT_AUDITOR_SYSTEM = f"""You are the Statistical Auditor.

You focus strictly on quantitative reasoning.

Evaluate:
1. Statistical validity
2. Model assumptions
3. Power logic
4. Correct inference
5. Causal claims
6. Missing analyses

If information is missing, explicitly request it.

Output Format:

Critical Statistical Issues
Numbered list.

Suggested Analyses
Concrete suggestions.

Severity Rating
Low / Moderate / High
{_GLOBAL}"""

SKEPTIC_SYSTEM = f"""You are the Skeptic Reviewer.

You assume the central hypothesis is wrong.

Your job:
* Find alternative explanations.
* Identify circular reasoning.
* Attack causal interpretations.
* Challenge novelty.
* Identify hidden assumptions.

Be adversarial but rational.

Output Format:

Central Weakness
Primary vulnerability of the paper.

Alternative Explanations
Numbered list.

Fatal vs Fixable Issues
Classify each.

Recommendation
Reject / Major Revision / Minor Revision / Accept
{_GLOBAL}"""

EDITOR_SYSTEM = f"""You are the handling Editor.

You:
* Read the manuscript.
* Read all reviews.
* Determine whether concerns were addressed.
* Weigh severity and convergence of critique.
* Make a decision.

Decision Options:
* Reject
* Major Revision
* Minor Revision
* Accept

Output Format:

Decision
[write exactly one of: Reject, Major Revision, Minor Revision, Accept]

Rationale
Detailed justification.

Required Actions (if revision)
Numbered list.

Improvement Assessment
Has the manuscript improved relative to the previous round?
Yes / No / N/A (first round)
Explain.

Accept ONLY if ALL of the following are satisfied:
* No unresolved major concerns remain.
* Causal claims match evidence.
* Statistical reasoning is sound.
* Conceptual clarity is high.
* Reviewer scores average ≥ 8 on Rigor and Clarity.
* Skeptic no longer identifies fatal flaws.
{_GLOBAL}"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(title: str, char: str = "─", width: int = 72) -> str:
    bar = char * width
    return f"\n{bar}\n  {title}\n{bar}\n"


def stream_agent(
    client: anthropic.Anthropic,
    system: str,
    messages: list,
    label: str,
    max_tokens: int = 8192,
    use_beta: bool = False,
) -> str:
    """
    Call an agent, stream its response to stdout, and return the full text.

    use_beta=True → client.beta.messages.stream() with Files API beta header
                    (required when the content includes a file_id document block)
    use_beta=False → client.messages.stream()
    """
    print(_banner(label))

    parts: list[str] = []

    kwargs = dict(
        model=MODEL,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=system,
        messages=messages,
    )

    stream_ctx = (
        client.beta.messages.stream(**kwargs, betas=[FILES_BETA])
        if use_beta
        else client.messages.stream(**kwargs)
    )

    with stream_ctx as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            parts.append(text)

    print("\n")
    return "".join(parts)


def _file_block(file_id: str) -> dict:
    """Document content block referencing an uploaded PDF."""
    return {
        "type": "document",
        "source": {"type": "file", "file_id": file_id},
        "title": "Manuscript Under Review",
    }


def _text_block(text: str, label: str = "Manuscript Under Review") -> dict:
    return {"type": "text", "text": f"## {label}\n\n{text}"}


def _user(content: list) -> dict:
    return {"role": "user", "content": content}


def _extract_manuscript(author_text: str) -> str:
    """Pull the revised manuscript from between sentinel markers."""
    start = author_text.find(MS_START)
    end = author_text.find(MS_END)
    if start != -1 and end != -1:
        return author_text[start + len(MS_START) : end].strip()
    # Fallback: everything after the last "Revised Manuscript" heading
    for pattern in (
        f"\n{MS_START}\n",
        "## Revised Manuscript\n",
        "# Revised Manuscript\n",
        "Revised Manuscript\n\n",
    ):
        idx = author_text.find(pattern)
        if idx != -1:
            return author_text[idx + len(pattern) :].strip()
    return author_text  # last resort


def _parse_decision(editor_text: str) -> str:
    """
    Extract the editor's decision string.
    Returns one of: 'Accept', 'Reject', 'Major Revision', 'Minor Revision', 'Unknown'.
    """
    m = re.search(
        r"Decision\s*\n+\s*(Accept|Reject|Major Revision|Minor Revision)",
        editor_text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().title()
    # Looser: check for a bare "Accept" line
    if re.search(r"^\s*Accept\s*$", editor_text, re.MULTILINE | re.IGNORECASE):
        return "Accept"
    return "Unknown"


def _format_reviews(reviews: dict[str, str]) -> str:
    sep = "─" * 60
    return "\n\n".join(
        f"{sep}\n{name}\n{sep}\n\n{text}" for name, text in reviews.items()
    )


# ── Main Review Loop ──────────────────────────────────────────────────────────

REVIEWERS = [
    ("Reviewer 1 – Computational Neuroscience", REVIEWER1_SYSTEM),
    ("Reviewer 2 – Experimental Methods",       REVIEWER2_SYSTEM),
    ("Reviewer 3 – Theory & Philosophy",        REVIEWER3_SYSTEM),
    ("Statistical Auditor",                     STAT_AUDITOR_SYSTEM),
    ("Skeptic Reviewer",                        SKEPTIC_SYSTEM),
]


def run(pdf_path: str, max_rounds: int, output_path: str) -> None:
    client = anthropic.Anthropic()
    log: list[str] = []

    def record(heading: str, content: str, level: int = 3) -> None:
        prefix = "#" * level
        log.append(f"{prefix} {heading}\n\n{content}\n")

    # ── Upload PDF ────────────────────────────────────────────────────────────
    pdf = Path(pdf_path)
    if not pdf.exists():
        sys.exit(f"Error: file not found — {pdf_path}")

    print(_banner(f"Uploading {pdf.name}", "═"))
    with pdf.open("rb") as fh:
        uploaded = client.beta.files.upload(
            file=(pdf.name, fh, "application/pdf"),
        )
    file_id = uploaded.id
    print(f"  Uploaded → {file_id}\n")

    log.append(f"# Peer Review: {pdf.name}\n\nFile ID: `{file_id}`\n")

    # ── State ─────────────────────────────────────────────────────────────────
    current_file_id: str | None = file_id   # None after round 1
    current_text: str | None = None          # populated after first author revision
    prev_editor_out: str = ""

    try:
        for rnd in range(1, max_rounds + 1):
            use_beta = current_file_id is not None  # True only in round 1

            print(_banner(f"ROUND {rnd} / {max_rounds}", "═"))
            log.append(f"## Round {rnd}\n")

            ms_block = (
                _file_block(current_file_id)
                if use_beta
                else _text_block(current_text)
            )

            # ── Reviewer pass ─────────────────────────────────────────────────
            reviews: dict[str, str] = {}
            for name, sys_prompt in REVIEWERS:
                msg = _user([
                    ms_block,
                    _text_block(
                        "Please review the manuscript above according to your role and output format.",
                        label="Instructions",
                    ),
                ])
                out = stream_agent(client, sys_prompt, [msg], name, use_beta=use_beta)
                reviews[name] = out
                record(name, out)

            reviews_text = _format_reviews(reviews)

            # ── Editor ───────────────────────────────────────────────────────
            editor_instructions = (
                "The following reviewer reports have been submitted for this manuscript.\n\n"
                f"{reviews_text}\n\n"
                "Issue your editorial decision following your output format."
            )
            if prev_editor_out:
                editor_instructions = (
                    f"Previous editorial decision:\n{prev_editor_out}\n\n"
                    + editor_instructions
                )

            editor_msg = _user([
                ms_block,
                _text_block(editor_instructions, label="Reviewer Reports & Instructions"),
            ])
            editor_out = stream_agent(
                client, EDITOR_SYSTEM, [editor_msg], "Editor",
                max_tokens=4096, use_beta=use_beta,
            )
            prev_editor_out = editor_out
            record("Editor Decision", editor_out)

            decision = _parse_decision(editor_out)
            print(f"\n{'━'*72}")
            print(f"  Decision: {decision}")
            print(f"{'━'*72}\n")
            log.append(f"**Decision: {decision}**\n")

            # ── Check terminal conditions ─────────────────────────────────────
            if decision == "Accept":
                print(_banner("✓ MANUSCRIPT ACCEPTED", "═"))
                if current_text:
                    print(_banner("FINAL ACCEPTED MANUSCRIPT", "─"))
                    print(current_text)
                    print()
                    log.append(f"\n# Final Manuscript\n\n{current_text}\n")
                else:
                    note = f"[Original PDF accepted without revision: {pdf.name}]"
                    print(note)
                    log.append(f"\n# Final Manuscript\n\n{note}\n")
                break

            if decision == "Reject":
                print(_banner("✗ MANUSCRIPT REJECTED", "═"))
                log.append(f"\n**Manuscript rejected after round {rnd}.**\n")
                if current_text:
                    print(_banner("Last Revised Manuscript (Rejected)", "─"))
                    print(current_text)
                    print()
                    log.append(f"\n# Last Revised Manuscript\n\n{current_text}\n")
                break

            if rnd == max_rounds:
                print(_banner(f"MAX ROUNDS ({max_rounds}) REACHED — not accepted", "═"))
                log.append(
                    f"\n**Stopped: max rounds ({max_rounds}) reached. "
                    f"Final decision: {decision}**\n"
                )
                if current_text:
                    print(_banner("Last Revised Manuscript", "─"))
                    print(current_text)
                    print()
                    log.append(f"\n# Last Revised Manuscript\n\n{current_text}\n")
                break

            # ── Author revision ───────────────────────────────────────────────
            author_instructions = (
                "You have received the following reviewer reports and the editorial decision.\n\n"
                f"{reviews_text}\n\n"
                f"Editorial Decision:\n{editor_out}\n\n"
                "Revise the manuscript addressing all major concerns. "
                "Follow your output format exactly.\n"
                f"Place the COMPLETE revised manuscript between these markers:\n"
                f"{MS_START}\n"
                "[full revised manuscript]\n"
                f"{MS_END}"
            )
            author_msg = _user([
                ms_block,
                _text_block(author_instructions, label="Reviews & Instructions"),
            ])
            author_out = stream_agent(
                client, AUTHOR_SYSTEM, [author_msg], "Author – Revision",
                max_tokens=32000, use_beta=use_beta,
            )
            record("Author Response & Revision", author_out)

            # Extract and display the revised manuscript
            current_text = _extract_manuscript(author_out)
            current_file_id = None  # switch to text from round 2 onward

            # ── Print revised manuscript prominently ──────────────────────────
            print(_banner(f"Revised Manuscript — Round {rnd}", "─"))
            print(current_text)
            print()
            record(f"Revised Manuscript — Round {rnd}", current_text, level=2)

    finally:
        # ── Clean up uploaded file ────────────────────────────────────────────
        try:
            client.beta.files.delete(file_id)
            print(f"\nCleaned up remote file: {file_id}")
        except Exception:
            pass

        # ── Write output ──────────────────────────────────────────────────────
        Path(output_path).write_text("\n".join(log), encoding="utf-8")
        print(f"Output saved → {output_path}\n")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Academic Peer Review System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python peer_review.py paper.pdf\n"
            "  python peer_review.py paper.pdf --max-rounds 3\n"
            "  python peer_review.py paper.pdf --output results.md\n"
        ),
    )
    parser.add_argument("pdf", help="Path to the PDF manuscript")
    parser.add_argument(
        "--max-rounds", "-r",
        type=int,
        default=5,
        metavar="N",
        help="Maximum review rounds (default: 5)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Output markdown file (default: peer_review_<name>_<timestamp>.md)",
    )
    args = parser.parse_args()

    if not args.output:
        stem = Path(args.pdf).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"peer_review_{stem}_{ts}.md"

    run(args.pdf, args.max_rounds, args.output)


if __name__ == "__main__":
    main()
