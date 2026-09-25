"""
Turn retrieved chunks + a question into a cited answer.

Deliberately provider-agnostic: swap `LLM_PROVIDER` in .env between
"anthropic" and "none" (a rule-based stub used for offline testing, so
the rest of the pipeline -- prompt construction, citation formatting --
can be exercised without an API key).
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import GenerationConfig  # noqa: E402
from src.retrieve import RetrievedChunk  # noqa: E402

CITATION_RE = re.compile(r"\[([a-zA-Z0-9_]+),\s*p\.(\d+)\]")


@dataclass
class Answer:
    text: str
    citations: list[tuple[str, int]]  # (doc_id, page_number)
    refused: bool
    raw_context_used: list[str]


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        tags = [f"effective {c.effective_date}"]
        if c.faculty:
            tags.append(f"faculty: {c.faculty}")
        if c.approx_pages:
            tags.append("NOTE: page number is an approximate citation unit, not the original PDF page")
        parts.append(
            f"### Source [{c.doc_id}, p.{c.page_number}] -- {c.title} ({'; '.join(tags)})\n{c.text}"
        )
    return "\n\n".join(parts)


REFUSAL_TEXT = "I couldn't find this in the documents I have access to."


def _stub_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    A deterministic, no-network stand-in for the LLM call, used so the
    pipeline can be smoke-tested (and CI-tested) without an API key.
    It never invents facts: it just surfaces the single most relevant
    retrieved sentence with its citation, or refuses if nothing was
    retrieved with a reasonable score.
    """
    if not chunks or chunks[0].fused_score < 0.05:
        return REFUSAL_TEXT
    top = chunks[0]
    sentence = top.text.strip().split(". ")[0].strip()
    if not sentence.endswith("."):
        sentence += "."
    return f"{sentence} [{top.doc_id}, p.{top.page_number}]"


def generate_answer(
    question: str, chunks: list[RetrievedChunk], cfg: GenerationConfig | None = None
) -> Answer:
    cfg = cfg or GenerationConfig()
    context = build_context_block(chunks)

    if cfg.provider == "none":
        text = _stub_answer(question, chunks)
    elif cfg.provider == "anthropic":
        text = _call_anthropic(question, context, cfg)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {cfg.provider}")

    citations = [(m.group(1), int(m.group(2))) for m in CITATION_RE.finditer(text)]
    refused = REFUSAL_TEXT.lower() in text.lower()
    return Answer(
        text=text,
        citations=citations,
        refused=refused,
        raw_context_used=[c.chunk_id for c in chunks],
    )


def _call_anthropic(question: str, context: str, cfg: GenerationConfig) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your "
            "key, or set LLM_PROVIDER=none to smoke-test without a live model."
        )
    client = anthropic.Anthropic(api_key=api_key)
    user_msg = (
        f"Context passages:\n\n{context}\n\n---\n\nQuestion: {question}\n\n"
        "Answer, citing every claim as [doc_id, p.N]."
    )
    resp = client.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        system=cfg.system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")
