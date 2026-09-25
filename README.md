# University Regulations Assistant (RAG)

A retrieval-augmented question-answering system over the real academic
regulations, handbooks and disciplinary circulars of **Rajarata University
of Sri Lanka** -- built as a portfolio-grade RAG project, not a tutorial
demo. It answers a student's question by retrieving the exact clauses that
apply, citing them, and refusing when the answer genuinely isn't in the
corpus.

The point of this repo isn't "it works" -- it's the eval harness that
measures *how well* it works, and the honest write-up of where it doesn't
(`eval/error_analysis.md`). Read that file before this one if you only have
five minutes; it's the part worth discussing in an interview.

## Why this document set

Eleven real, public documents spanning six faculties of one university:
- `Student-Handbook-2022-2023.txt` -- Faculty of Applied Sciences handbook
- `FOT_Handbook_2021.pdf` -- Faculty of Technology handbook (a second,
  independent source covering some of the same ground -- see the
  multi-document eval questions, `q15`/`q24`)
- `FMAS-Student-Handbook-2025.txt` -- Faculty of Medicine and Allied
  Sciences (MBBS) handbook, with its **own** faculty-specific
  examination-offence punishment table -- a third generation of the same
  plagiarism rule, and a genuinely unhandled conflict (see
  `eval/error_analysis.md`, finding #7)
- `Agriculture-Student-Handbook-2023.pdf` -- Faculty of Agriculture
  handbook, which turns out to be orientation/student-life focused with
  none of the GPA/attendance/plagiarism content the other three have
  (finding #9) -- a real institutional corpus is this uneven
- `Social-Sciences-Humanities-Handbook-2021-25.docx` -- Faculty of Social
  Sciences and Humanities handbook, arrived as a `.docx` export rather
  than a PDF; its own "4.14 Punishments" clause *defers* to the
  university-wide Senate procedure instead of restating a table -- a
  useful contrast with FMAS's faculty-specific override above
- `Management-Studies-Handbook-2023.docx` -- Faculty of Management
  Studies handbook, also a `.docx` export, and **silently incomplete**:
  its own table of contents promises an "Evaluation Procedure" and an
  "Examination Offences & Punishments" section that the extracted text
  never actually reaches (finding #11) -- included anyway, because a
  real corpus sometimes hands you a truncated file and the honest move
  is to detect and document that, not to pretend it isn't there
- Examination regulations, two generations of the same subject matter:
  the 2014 regulations and the **2025 by-law that supersedes them** --
  a genuine "which source wins" conflict, not a synthetic one
- Two UGC circulars (student discipline; anti-ragging -- the latter is a
  scanned PDF with no text layer, exercising the OCR path for real)
- The UGC's University Student Charter

## Architecture

```
data/raw/*.pdf, *.txt, *.docx
        |
        v
  src/ingest.py -------- PyMuPDF text extraction; python-docx extraction
        |                (paragraphs AND tables, in document order); OCR
        |                (Tesseract) fallback for scanned pages
        v
  src/chunk.py ---------- paragraph/clause-aware chunking, oversized-
        |                 paragraph splitting, small overlap
        v
  src/build_index.py ---- BM25 (rank_bm25) + dense index (pluggable
        |                 embedding backend, see below)
        v
  src/retrieve.py -------- hybrid retrieval: score-fused BM25 + dense,
        |                  reranking, "prefer the superseding document"
        v
  src/generate.py -------- Claude call with a citation-enforcing system
        |                  prompt and an explicit refusal instruction
        v
  src/pipeline.py --------- glue + timing
        |
        +--> src/app.py (FastAPI) --> ui/index.html
        +--> eval/run_eval.py --> eval/results.jsonl, eval/error_analysis.md
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY, or leave LLM_PROVIDER=none

bash scripts/run_pipeline.sh          # ingest -> chunk -> index -> eval (retrieval only)
python -m eval.run_eval --generate    # also scores refusals/citations (needs a real key)

uvicorn src.app:app --reload --port 8000   # then open ui/index.html in a browser
```

Ask something directly from the CLI without starting the server:

```bash
python -m src.pipeline "What happens if I miss the final exam without permission from the Faculty Board?"
```

## Evaluation results (retrieval, k=4, n=29 real questions)

| mode   | hit_rate@4 | MRR   |
|--------|-----------:|------:|
| bm25   | 0.897      | 0.750 |
| dense  | 0.966      | 0.922 |
| hybrid | 0.931      | 0.868 |

These numbers use the **offline TF-IDF+SVD embedding fallback** (see
"Environment notes" below) -- re-run `eval/run_eval.py` after switching to
`EMBEDDING_BACKEND=sentence_transformers` and update this table before
presenting the project; don't present offline-fallback numbers as final.
Full per-question detail, and *why* hybrid doesn't simply beat both
single retrievers here, is in `eval/error_analysis.md` (finding #3).

Generation is measured separately once `ANTHROPIC_API_KEY` is set:
`python -m eval.run_eval --generate` scores whether the model correctly
refuses the deliberately unanswerable / incomplete-source questions and
whether every answer's citation actually points at the expected source
document.

## The 29-question eval set (`eval/eval_questions.jsonl`)

Not Titanic-style toy questions -- every one is grounded in text actually
present in the corpus, with a verified source page:

- **19 factual** -- credit requirements, GPA formula, attendance rules,
  appeal deadlines, reporting channels for misconduct, and
  MBBS-specific, Agriculture-specific, Social-Sciences-specific and
  Management-Studies-specific facts
- **4 conflict** -- the same offence (plagiarism; a repeated offence)
  governed by up to *three* generations/scopes of rule at once: the 2014
  university regulations, the 2025 university by-law that supersedes
  them, and a faculty's own punishment table -- the hardest and most
  instructive questions in the set (error analysis, findings #1 and #7)
- **1 OCR-sourced** -- answerable only because the scanned ragging
  circular went through the OCR fallback
- **2 multi-document/ambiguous** -- attendance rules are stated in some
  faculty handbooks and silent in others; the "right" answer names which
  source it used rather than generalizing one faculty's rule to all
  (finding #8)
- **2 unanswerable** -- Wi-Fi password, tuition fees -- neither exists in
  the corpus; a correct system refuses instead of guessing
- **1 incomplete_source** (new category) -- the Management Studies
  handbook's own examination-offences section, which its table of
  contents promises but the actual export never reaches (finding #11).
  Distinct from "unanswerable": the fact isn't absent from the world,
  it's absent from *this particular copy* of the source document -- a
  correct system still has to refuse rather than guess, but the honest
  explanation is different, and the eval set makes that distinction
  explicit rather than lumping both into one "refused correctly" bucket

## Environment notes (read before judging the retrieval numbers)

This project was first built inside a network-restricted sandbox that
allows PyPI and the Anthropic API but blocks Hugging Face Hub downloads
(`sentence-transformers`' pretrained models come from there). Rather than
fake the semantic-search step, `src/embeddings.py` implements a fully
offline fallback (TF-IDF + truncated SVD, i.e. classic LSA) so the whole
pipeline could be built and measured end to end without network access.
Switching to real sentence embeddings anywhere with normal internet
access is one line: `EMBEDDING_BACKEND=sentence_transformers` in `.env`.
Do this and re-run the eval before treating the numbers above as final --
`eval/error_analysis.md` (finding #4) explains the trade-off in full and
is worth quoting directly if asked about it.

The same applies to the reranker: `USE_RERANKER=1` switches on a real
`cross-encoder` model (also needs the Hugging Face download); until then,
`retrieve.py` uses a cheap lexical-overlap fallback.

## What I'd improve next

1. Table-aware chunking for the examination by-law's offence schedule
   (finding #1) -- the single highest-value fix, since it's the exact
   kind of clause a student would actually search for.
2. Faculty-aware retrieval: detect a faculty mentioned in the query and
   boost chunks whose `faculty` metadata matches, so FMAS's own
   plagiarism table stops losing to the generic university by-law
   (finding #7), and a faculty-less query stops being resolved in favour
   of whichever document repeats the term most (finding #8).
3. Suppress front-matter noise (welcome messages, dean's messages): they
   can outrank the specific numbered clause a query is actually asking
   about, purely on lexical/semantic similarity to generic phrasing
   (finding #12, reproduced by `q26`) -- downweighting by position, or
   biasing toward chunks containing numbers/defined terms when the query
   does too, are both worth trying.
4. Detect a silently incomplete source document instead of only
   discovering it by accident (finding #11): cross-check a handbook's
   own table of contents against which sections actually produced
   chunks, and surface a warning rather than a confident-looking corpus.
5. Tune BM25/dense fusion weights against a larger eval set instead of a
   flat 0.5/0.5 split (finding #3).
6. Surface OCR confidence per chunk so low-quality scanned pages are
   visibly flagged rather than silently trusted (finding #2).
7. Harden `.docx` extraction further: `_iter_block_items` now covers
   paragraphs and top-level table cells in document order (finding #10),
   but nested tables and text boxes are still unhandled.
8. Grow the eval set past 29 questions and add an LLM-as-judge pass for
   answer faithfulness, not just citation-presence and refusal-accuracy.
9. Swap in real embeddings + a cross-encoder reranker outside the sandbox
   and republish the results table with those numbers.

## Repo layout

```
data/raw/            the 11 source documents
data/processed/       generated: pages.jsonl, chunks.jsonl, bm25.pkl, dense.npz (gitignored)
src/                  ingest, chunk, embeddings, build_index, retrieve, generate, pipeline, app, config
ui/index.html         minimal frontend, no build step
eval/                 eval_questions.jsonl, metrics.py, run_eval.py, error_analysis.md, results.jsonl
scripts/run_pipeline.sh
Dockerfile, requirements.txt, .env.example
```
