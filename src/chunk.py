"""
Structure-aware chunking.

A naive fixed-size splitter cuts numbered clauses and tables in half --
exactly the kind of chunking bug this project is meant to catch with its
own eval set. Instead we:

  1. Split each page on blank lines into paragraphs.
  2. Try to detect list/clause boundaries (e.g. "1.", "a.", "8.10.6.1")
     and keep a clause's heading glued to its body.
  3. Pack paragraphs into ~target_tokens windows with overlap, but never
     split a single paragraph across two chunks unless it alone exceeds
     the target size.

Run directly to (re)build data/processed/chunks.jsonl from pages.jsonl:
    python -m src.chunk
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ChunkConfig, DATA_PROCESSED, CHUNKS_PATH  # noqa: E402

CLAUSE_RE = re.compile(r"^\s*(\d+(\.\d+)*\.?|[a-z]\.)\s+\S")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _word_count(text: str) -> int:
    return len(text.split())


def _split_oversized(paragraph: str, max_words: int) -> list[str]:
    """
    Some source pages (e.g. Student Handbook p.38) have almost no blank
    lines at all, so split_paragraphs() hands back one giant "paragraph"
    spanning several unrelated rules. Left alone, that produced a single
    450-word chunk in early testing -- more than double the target size,
    and vague enough to hurt retrieval precision. Break it on sentence
    boundaries instead of returning it whole.
    """
    if _word_count(paragraph) <= max_words:
        return [paragraph]
    sentences = SENTENCE_RE.split(paragraph)
    pieces, buf, buf_words = [], [], 0
    for sent in sentences:
        w = _word_count(sent)
        if buf_words + w > max_words and buf:
            pieces.append(" ".join(buf))
            buf, buf_words = [], 0
        buf.append(sent)
        buf_words += w
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def split_paragraphs(page_text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
    if len(paras) <= 1:
        # Some PDF extractions have no blank lines at all -- fall back to
        # splitting on lines that look like a new numbered clause.
        lines = page_text.split("\n")
        paras, current = [], []
        for line in lines:
            if CLAUSE_RE.match(line) and current:
                paras.append(" ".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            paras.append(" ".join(current).strip())
        paras = [p for p in paras if p.strip()]
    return paras or [page_text]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    doc_type: str
    effective_date: str
    supersedes: list
    page_number: int
    text: str
    ocr: bool = False
    faculty: str | None = None
    approx_pages: bool = False


def chunk_page(page: dict, cfg: ChunkConfig, running_idx: int) -> list[Chunk]:
    paras = split_paragraphs(page["text"])
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_words = 0

    def flush():
        nonlocal buf, buf_words, running_idx
        text = "\n\n".join(buf).strip()
        if text and _word_count(text) >= cfg.min_tokens:
            chunks.append(
                Chunk(
                    chunk_id=f"{page['doc_id']}_p{page['page_number']}_{running_idx}",
                    doc_id=page["doc_id"],
                    title=page["title"],
                    doc_type=page["doc_type"],
                    effective_date=page["effective_date"],
                    supersedes=page["supersedes"],
                    page_number=page["page_number"],
                    text=text,
                    ocr=page.get("ocr", False),
                    faculty=page.get("faculty"),
                    approx_pages=page.get("approx_pages", False),
                )
            )
            running_idx += 1
        buf, buf_words = [], 0

    # Break up any paragraph that alone exceeds the target size (see
    # _split_oversized) before packing, so one wall-of-text page can't
    # produce one wall-of-text chunk.
    max_para_words = int(cfg.target_tokens * 1.3)
    expanded_paras: list[str] = []
    for para in paras:
        expanded_paras.extend(_split_oversized(para, max_para_words))
    paras = expanded_paras

    for para in paras:
        pw = _word_count(para)
        if buf_words + pw > cfg.target_tokens and buf:
            flush()
            # carry a small overlap tail forward for context continuity
            if cfg.overlap_tokens and chunks:
                tail_words = chunks[-1].text.split()[-cfg.overlap_tokens :]
                if tail_words:
                    buf = [" ".join(tail_words)]
                    buf_words = len(tail_words)
        buf.append(para)
        buf_words += pw
    flush()
    return chunks


def build_chunks(cfg: ChunkConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkConfig()
    pages_path = DATA_PROCESSED / "pages.jsonl"
    if not pages_path.exists():
        raise SystemExit("Run `python -m src.ingest` first.")

    all_chunks: list[Chunk] = []
    idx = 0
    with open(pages_path, encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)
            page_chunks = chunk_page(page, cfg, idx)
            idx += len(page_chunks)
            all_chunks.extend(page_chunks)
    return all_chunks


def main():
    chunks = build_chunks()
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c.doc_id] = by_doc.get(c.doc_id, 0) + 1
    print(f"Wrote {len(chunks)} chunks -> {CHUNKS_PATH}\n")
    for doc_id, n in by_doc.items():
        print(f"  {doc_id:30s} {n:4d} chunks")


if __name__ == "__main__":
    main()
