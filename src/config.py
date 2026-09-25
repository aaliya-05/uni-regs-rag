"""
Central configuration for the pipeline.

Kept deliberately simple (no pydantic-settings, no YAML) so the whole
project can be read top-to-bottom in one sitting -- that is itself a
portfolio decision worth defending in an interview.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH = DATA_PROCESSED / "chunks.jsonl"
BM25_INDEX_PATH = DATA_PROCESSED / "bm25.pkl"
DENSE_INDEX_PATH = DATA_PROCESSED / "dense.npz"

# Each source document, with the metadata that makes citations and
# "which document wins" questions possible. `effective_date` and
# `supersedes` are what let the retriever and the eval set reason about
# the amended 2025 by-law overriding the 2014 examination regulations.
DOCUMENTS = [
    dict(
        doc_id="student_handbook_2022_23",
        path=DATA_RAW / "Student-Handbook-2022-2023.txt",
        title="Student Handbook 2022/2023 -- Faculty of Applied Sciences, Rajarata University of Sri Lanka",
        doc_type="handbook",
        faculty="Applied Sciences",
        effective_date="2022-01-01",
        supersedes=[],
        # This .txt export repeats this footer at (almost) every original
        # page break -- see parse_txt() in ingest.py. Page numbers derived
        # from it are approximate; see README / error_analysis finding #5.
        page_footer="Handbook 2022/23",
    ),
    dict(
        doc_id="fot_handbook_2021",
        path=DATA_RAW / "FOT_Handbook_2021.pdf",
        title="Student Handbook 2021 -- Faculty of Technology, Rajarata University of Sri Lanka",
        doc_type="handbook",
        faculty="Technology",
        effective_date="2021-01-01",
        supersedes=[],
    ),
    dict(
        doc_id="fmas_handbook_2025",
        path=DATA_RAW / "FMAS-Student-Handbook-2025.txt",
        title="Student Handbook 2025 (MBBS) -- Faculty of Medicine and Allied Sciences, Rajarata University of Sri Lanka",
        doc_type="handbook",
        faculty="Medicine and Allied Sciences",
        effective_date="2025-11-01",
        # Its own 4.2 "Examination irregularities/offences and punishments"
        # (approved by Faculty Board Nov 2023, Senate Nov 2023, Council
        # Jul 2024) is a FACULTY-SPECIFIC punishment table that covers the
        # same offences (including plagiarism) as the university-wide
        # exam_bylaw_2025. This is deliberately NOT modelled as
        # "supersedes" -- it doesn't replace the university by-law, it
        # sits alongside it for one faculty's MBBS programme. The
        # retriever has no notion of that faculty-scoping; see
        # error_analysis.md finding #7.
        supersedes=[],
        # No reliable repeating footer in this text export (unlike the
        # Applied Sciences handbook above) -- ingest.py falls back to
        # fixed-size pseudo-pages for it. Treat its page numbers as
        # coarse citation units, not real PDF pages.
        approx_pages=True,
    ),
    dict(
        doc_id="agriculture_handbook_2023",
        path=DATA_RAW / "Agriculture-Student-Handbook-2023.pdf",
        title="Student Handbook 2023 (4th Edition) -- Faculty of Agriculture, Rajarata University of Sri Lanka",
        doc_type="handbook",
        faculty="Agriculture",
        effective_date="2023-01-01",
        supersedes=[],
        # Unlike the other handbooks, this one is orientation/student-life
        # focused (admissions, facilities, gold medals, anti-harassment
        # policy) and does NOT contain GPA, credit-rating or examination-
        # offence rules -- see error_analysis.md finding #6. Included
        # anyway because a real institutional corpus is this uneven.
    ),
    dict(
        doc_id="social_sciences_handbook_2021_25",
        path=DATA_RAW / "Social-Sciences-Humanities-Handbook-2021-25.docx",
        title="Student Handbook 2021-2025 -- Faculty of Social Sciences and Humanities, Rajarata University of Sri Lanka",
        doc_type="handbook",
        faculty="Social Sciences and Humanities",
        effective_date="2021-01-01",
        supersedes=[],
        # Unlike FMAS, this handbook's own "4.14 Punishments" clause
        # explicitly DEFERS to the university-wide Senate procedure
        # rather than restating its own table -- a useful contrast with
        # finding #7 (not every faculty overrides the general by-law;
        # some just point back to it).
        approx_pages=True,
    ),
    dict(
        doc_id="management_studies_handbook_2023",
        path=DATA_RAW / "Management-Studies-Handbook-2023.docx",
        title="Student Handbook 2021-2023 -- Faculty of Management Studies, Rajarata University of Sri Lanka",
        doc_type="handbook",
        faculty="Management Studies",
        effective_date="2023-01-01",
        supersedes=[],
        approx_pages=True,
        # This .docx export is CUT OFF partway through -- its own table
        # of contents promises "Evaluation Procedure" (p.86) and
        # "Examination Procedure, Offences & Punishments" (p.92), but the
        # extracted text stops at p.70 (mid-way through the department
        # list) and neither section is actually present. Not a parsing
        # bug: the source file itself doesn't contain that content. See
        # error_analysis.md finding #11 -- a RAG system has no way to
        # detect a silently incomplete source document on its own.
        known_incomplete=True,
    ),
    dict(
        doc_id="exam_regs_2014",
        path=DATA_RAW / "Rules-and-Regulations-on-Examination-Procedures-Examination-Irregularities-and-Punishments-English.pdf",
        title="Regulations Concerning Examination Procedures, Examination Irregularities and Punishments (Amended 2014)",
        doc_type="by_law",
        effective_date="2015-08-20",
        supersedes=[],
    ),
    dict(
        doc_id="exam_bylaw_2025",
        path=DATA_RAW / "Amended-By-Law-EDC-05-02-2025_Exam.pdf",
        title="By-law No. 01/2024 on the Conduct of Candidates at Examinations and Punishments for Violation of Examination Rules",
        doc_type="by_law",
        effective_date="2025-03-01",
        # This is the document a correct RAG system must prefer whenever
        # it conflicts with exam_regs_2014 -- it is the newer, currently
        # in-force by-law covering the same subject matter.
        supersedes=["exam_regs_2014"],
    ),
    dict(
        doc_id="ugc_circular_919_ragging",
        path=DATA_RAW / "UGC_Circular_919_guidelines_to_curb_ragging.pdf",
        title="UGC Circular No. 919 -- Guidelines to Curb Ragging",
        doc_type="circular",
        effective_date="2010-01-01",
        supersedes=[],
        # This file has no extractable text layer -- it is a scan.
        # ingest.py routes it through OCR. Flagging it here documents
        # *why*, instead of silently degrading.
        requires_ocr=True,
    ),
    dict(
        doc_id="ugc_circular_946_discipline",
        path=DATA_RAW / "Common_guidelines_of_student_discipline_UGC__Circular_No_946.pdf",
        title="UGC Circular No. 946 -- Common Guidelines on Student Discipline",
        doc_type="circular",
        effective_date="2011-02-10",
        supersedes=[],
    ),
    dict(
        doc_id="student_character_charter",
        path=DATA_RAW / "Student_Character_English.pdf",
        title="University Student Charter (UGC)",
        doc_type="charter",
        effective_date="2010-01-01",
        supersedes=[],
    ),
]


@dataclass
class ChunkConfig:
    target_tokens: int = 220          # ~ words; small enough for precise citations
    overlap_tokens: int = 40
    min_tokens: int = 30              # drop crumbs (running headers, page numbers)


@dataclass
class RetrievalConfig:
    top_k_bm25: int = 15
    top_k_dense: int = 15
    top_k_fused: int = 8              # candidates handed to the reranker
    top_k_final: int = 4              # chunks handed to the LLM
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    # "sentence-transformers" needs internet the first time it downloads
    # a model (blocked in this sandbox's proxy -- see README). "tfidf_svd"
    # is a fully offline fallback (a latent-semantic-analysis embedding)
    # so the pipeline can be built and tested end to end without network
    # access, then swapped for real embeddings in one line.
    embedding_backend: str = os.environ.get("EMBEDDING_BACKEND", "tfidf_svd")
    embedding_model_name: str = os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    svd_dims: int = 256
    use_reranker: bool = os.environ.get("USE_RERANKER", "0") == "1"
    reranker_model_name: str = os.environ.get(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


@dataclass
class GenerationConfig:
    provider: str = os.environ.get("LLM_PROVIDER", "anthropic")
    model: str = os.environ.get("LLM_MODEL", "claude-sonnet-5")
    max_tokens: int = 800
    temperature: float = 0.0
    system_prompt: str = (
        "You are a university regulations assistant. Answer ONLY using the "
        "provided context passages. Every factual claim must be followed by "
        "a citation like [doc_id, p.N]. If the context does not contain the "
        "answer, say exactly: \"I couldn't find this in the documents I have "
        "access to.\" Never guess. If two passages conflict, prefer the one "
        "from the document with the later effective_date and say so. If a "
        "passage comes from a specific faculty's handbook, say which "
        "faculty it applies to rather than presenting it as a university-"
        "wide rule -- other faculties may have their own version."
    )
