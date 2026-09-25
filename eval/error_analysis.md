# Error analysis

Twelve real findings from building and evaluating this pipeline on the actual
Rajarata University documents, in the order they were discovered. This file
is meant to be read alongside `eval/results.jsonl` and is the single most
interview-relevant artifact in the repo -- it is the difference between
"I built a RAG app" and "I measured one and know where it breaks."

## 1. The hybrid retriever misses the exact clause it should find for `q07`

**Question:** "What is the punishment for plagiarism in an examination or
assignment?"

**Expected:** `exam_bylaw_2025`, p.8 (the offence schedule's plagiarism row).

**What actually happens:** the top-4 hybrid results are all *procedural*
pages of the same by-law -- "Procedure for Inquiry" (p.10), "Enforcement of
Punishments" (p.11), "Appeals" (p.12) -- not the schedule table on p.8 that
actually states the punishment. Traced it to the source:

- The schedule on p.8 is a table. PDF text extraction flattens tables into
  short, disjointed lines ("Rules: 1.3 x, 1.4 b, 1.4 c" / "Cancellation of
  the candidature ..."), so the resulting chunk has comparatively few of the
  query's terms repeated -- BM25 score 16.1, versus 13.5-16.5 for the
  procedural chunks that repeat "examination offence" and "punishment"
  several times in flowing prose.
- On the dense side, the TF-IDF+SVD fallback embedding (see finding #3)
  compresses this short, list-like chunk into a vector that is *less*
  similar to the query than the procedural prose is, because SVD favours
  chunks with more overlapping running vocabulary.
- Net effect: the correct chunk (`exam_bylaw_2025_p8_495`) never makes the
  top-8 fused candidates, even though it *is* in the BM25-only top-3.

**Fix path, not yet implemented:** table-aware chunking that keeps each
offence row intact with its own heading ("e. Plagiarism ...") repeated in
every fragment of that row, and/or a real cross-encoder reranker (the
lexical-overlap fallback in `retrieve.py` isn't strong enough to recover
this). This is exactly the "improve, with evidence" step the plan calls
for in Week 3 -- the eval harness is what caught it.

## 2. OCR on the ragging circular is unreliable exactly where it matters least (and that's a warning, not a relief)

`UGC_Circular_919_guidelines_to_curb_ragging.pdf` has no text layer at all
(0 characters across 5 pages via PyMuPDF) -- it's a scan. `ingest.py`
routes it through Tesseract automatically (see `Page.ocr` flag). OCR
recovers the operative legal text on pages 1-4 well enough to answer `q14`
correctly. Page 5, a bilingual declaration template, OCRs as near-garbage
("Macca gu ccs 90 agg SSS ENGNG..."). It happens to be a form, not a rule,
so it doesn't cost us an eval question here -- but the corpus has no
mechanism to *flag low-confidence OCR* to a user, so a future document
with legally-relevant content on a similarly garbled page would fail
silently. Worth adding: a confidence score (Tesseract reports one per word)
surfaced in the citation, so the UI can visually flag "this citation comes
from a low-confidence OCR page."

## 3. 50/50 score fusion is not a free win over the better single retriever

Running `eval/run_eval.py` on the current 29-question, 11-document set
(see findings #7-#12 for the harder questions added as the corpus grew)
gives:

| mode   | hit_rate@4 | MRR   |
|--------|-----------:|------:|
| bm25   | 0.897      | 0.750 |
| dense  | 0.966      | 0.922 |
| hybrid | 0.931      | 0.868 |

Hybrid ties dense on hit-rate but has a *lower* MRR than dense alone -- on
the original 20-question, 7-document corpus the same pattern showed up
the other way round (hybrid tied BM25's hit-rate but trailed dense's MRR:
1.000/0.875 vs 0.950/0.950 vs 1.000/0.917). Either way, plain 50/50 fusion
sometimes buries a top-1-worthy single-retriever hit under an
equally-scored-but-worse result from the other signal. This is a real,
measured result, not a hypothetical: it means the fusion weights
(`RetrievalConfig.bm25_weight` / `dense_weight`) need actual tuning against
a larger eval set rather than being left at 0.5/0.5, and it's a better
"what I'd improve next" line for an interview than "I added hybrid search"
on its own.

Caveat that cuts the other way: the "dense" signal here is TF-IDF+SVD
(LSA), not a real sentence embedding (see finding below on why), so these
numbers should be re-run once a real embedding model is swapped in before
trusting them as a final verdict on fusion weighting.

## 4. The offline embedding fallback is a real design decision, not a shortcut

`sentence-transformers` needs to download a pretrained model from Hugging
Face on first use. The sandboxed environment this project was first built
in blocks that download (`httpx.ProxyError: 403 Forbidden` from
`huggingface.co` through the outbound proxy, while `pypi.org` and
`api.anthropic.com` are allowed). Rather than stub out retrieval entirely,
`src/embeddings.py` implements a second backend -- TF-IDF vectors
compressed with truncated SVD (classic LSA) -- so the *entire* pipeline
(ingest -> chunk -> index -> retrieve -> generate -> eval) could be built
and measured end to end without network access. Switching to real
embeddings anywhere outside that sandbox is one environment variable:
`EMBEDDING_BACKEND=sentence_transformers`. The eval harness is what makes
this an honest trade rather than a hidden one -- rerun it after switching
and compare the hit-rate/MRR table above.

## 5. The Student Handbook's page numbers are approximate

`Student-Handbook-2022-2023.txt` arrived as a pre-extracted text file, not
the original PDF, so `ingest.py` recovers page boundaries by splitting on
the recurring "Handbook 2022/23" footer string rather than from real PDF
page metadata (see `parse_txt()`). This mostly works, but a few of this
document's true page numbers used in `eval/eval_questions.jsonl` (verified
against the footer-delimited text, not the original PDF, which was not
available) may be off by one or two pages if a footer instance was merged
or missed during the original text extraction. Every other document in the
corpus was parsed straight from its own PDF with PyMuPDF, so its page
numbers are exact. This is disclosed here rather than silently presented
as precise -- a citation a user can't independently verify is worse than
no citation.

## 6. The offline generation stub cannot judge when to refuse

With `LLM_PROVIDER=none`, `refusal_correct` scores 0.900 and drops on
exactly the two genuinely unanswerable questions (`q16` Wi-Fi password,
`q17` tuition fees) -- the stub has no judgement, it just echoes the
top-retrieved sentence regardless of relevance. That is expected and
intentional: the stub exists only to smoke-test the retrieval-to-generation
wiring without an API key. The refusal behaviour this project is actually
claiming ("say you don't know rather than guess") is enforced by the
system prompt in `GenerationConfig.system_prompt` and only takes effect
once `LLM_PROVIDER=anthropic` is set with a real key -- re-run
`python -m eval.run_eval --generate` after adding one and update the
numbers in the README before presenting this project.

## 7. Faculty-specific rules aren't scoped -- a real, unhandled conflict (`q22`)

Adding the FMAS (Faculty of Medicine and Allied Sciences) 2025 handbook
surfaced a genuinely new failure mode. FMAS has its own examination-offence
punishment table (`4.2`, Table 13) approved by its own Faculty Board,
Senate and Council in 2023/2024 -- a plagiarism rule that exists
*alongside*, not instead of, the university-wide `exam_bylaw_2025`. For
`q22` ("An MBBS student plagiarises a continuous-assessment assignment --
what punishment applies?"):

```
[exam_bylaw_2025 p.10]      fused=0.879   <- wrong doc for this question
[fmas_handbook_2025 p.67]   fused=0.793   <- right doc, WRONG page (career guidance)
[fmas_handbook_2025 p.7]    fused=0.500
[fmas_handbook_2025 p.45]   fused=0.500
```

The real answer -- FMAS's own Table 13, on pseudo-page 61 -- doesn't
appear even in the top 8. A doc-level hit-rate metric (`hit_at_k` in
`eval/metrics.py`) would score this "correct" because `fmas_handbook_2025`
does appear at rank 2 -- which is exactly why `eval/run_eval.py` reports
doc-level metrics and this file exists to go one level deeper. The
retriever has no representation at all of "this document is scoped to one
faculty and should be preferred for that faculty's questions" -- it's
purely a term-and-vector match. `config.py` records each handbook's
`faculty`, but nothing downstream uses it yet.

**Fix path:** either (a) detect a faculty mention in the query and boost
chunks whose `faculty` field matches, or (b) ask a clarifying question
("which faculty are you asking about?") when a query matches
faculty-scoped chunks from more than one faculty with similar top scores.

## 8. A document that repeats a term densely can outrank the document you actually meant (`q15`, `q24`)

`q15`/`q24` ask about the "80% attendance" rule with no faculty named. The
Applied Sciences handbook states it once, cleanly, on p.38. The FMAS
handbook's assessment-structure section (`3.4.6`) repeats "80% attendance
to specified components" five times across nearby paragraphs describing
different course modules. Once chunked, that produces *several*
FMAS chunks that each score higher on BM25 than the single Applied
Sciences chunk -- not because FMAS is more relevant, but because it says
the phrase more often:

```
[fmas_handbook_2025 p.62]           fused=0.898
[fmas_handbook_2025 p.62]           fused=0.875
[fmas_handbook_2025 p.66]           fused=0.779
[fmas_handbook_2025 p.62]           fused=0.742
[student_handbook_2022_23 p.38]     fused=0.694   <- pushed to rank 5, outside top_k_final=4
```

The correct source is retrievable (it's rank 5, not absent), but the
default `top_k_final=4` cuts it off. This is the same underlying issue as
finding #7 from the other direction: without faculty-awareness, a query
that doesn't name a faculty gets silently resolved in favour of whichever
faculty's document happens to repeat the matching phrase most, not
necessarily the one the student meant.

## 9. Not every "handbook" is a regulations document -- corpus heterogeneity is real, not a data-cleaning failure

The Faculty of Agriculture handbook (`agriculture_handbook_2023`) has
zero occurrences of "plagiarism," "GPA," or an attendance percentage --
by table of contents, it's an orientation/student-life handbook (campus
map, admissions, facilities, gold medals, anti-harassment policy), not an
academic-regulations one. It was tempting to treat this as a parsing bug
and go looking for the "missing" GPA section; there isn't one. A
production system pointed at a real institution's document set needs to
handle this gracefully -- i.e. answer what the Agriculture handbook
*does* cover (`q23` exercises this) and correctly refuse or redirect
questions it structurally cannot answer, rather than assuming every
"Student Handbook" PDF has the same sections.

## 10. python-docx's `.paragraphs` silently skips table content

Two more handbooks (Social Sciences and Humanities; Management Studies)
arrived as `.docx` exports rather than PDFs -- much smaller files, but a
new parsing trap. `python-docx`'s convenient `Document.paragraphs`
property only returns paragraphs that live directly in the document body;
any paragraph inside a table cell is invisible to it. For a handbook that
puts a syllabus or a grading table in an actual Word table (both of these
did, to some extent -- 23 tables in the Management Studies file alone),
trusting `.paragraphs` alone would silently drop that content with no
error. `ingest.py`'s `_iter_block_items()` walks the document body's XML
directly instead, yielding paragraphs and tables in the order they
actually appear, so nothing in a table is lost. In this particular
corpus the tables mostly held staff listings, not regulations -- but the
bug class is real, and would matter a great deal on a document where the
grading scale itself is a table (as the exam by-law's is, per finding #1).

## 11. A source document can be silently incomplete, and nothing in the pipeline notices

The Management Studies handbook's own table of contents promises an
"Evaluation Procedure" section (p.86) and an "Examination Procedure,
Offences & Punishments" section (p.92) -- but the `.docx` export actually
available cuts off around p.70, mid-way through the department listings.
Confirmed two ways: zero occurrences of "GPA", "attendance", or
"plagiarism" anywhere in the extracted text, and the last extracted
paragraph is visibly mid-department-description, not a document ending.
This is not a parsing bug -- `ingest.py` faithfully extracts everything
that's actually in the file. It's a **data acquisition** problem: the
export the file came from was truncated before it reached this project,
almost certainly to get it under an upload size limit.

The unsettling part: nothing in the pipeline can tell the difference
between "this handbook has no examination-offences section" (a fact) and
"this copy of the handbook is missing its examination-offences section"
(a data gap). `q28` exercises exactly this — a correct system still has
to say "I couldn't find this," but that refusal means something
different here than it does for `q16`/`q17` (a question with no real
answer anywhere). **Fix path:** cross-check a handbook's own table of
contents against which of its named sections actually produced chunks,
and surface a warning for any that didn't — turning a silent gap into a
visible one.

## 12. Front-matter (welcome messages, dean's messages) can outrank the specific clause you're asking about

`q26` asks for BA (Honours) graduation requirements -- a precise,
numbered clause on Social Sciences handbook p.11 (96/120-127 credits,
minimum 2.00 GPA, no grade below C, 6-year limit). The top-4 hybrid
results are, instead, the Dean's welcome message (p.2), the Vice
Chancellor's welcome message (p.3), and an Economics department blurb
(p.21) -- p.11 doesn't make the top-4 at all (it's rank 5, fused=0.825
vs the welcome message's 0.992). These front-matter pages repeat exactly
the phrases a graduation question would use -- "Faculty of Social
Sciences and Humanities," "degree," "students," "programme" -- many
times each, in flowing promotional prose, which is precisely the kind of
text BM25 (and this project's TF-IDF+SVD dense fallback) rewards. This
is a different mechanism from findings #7/#8 (which were about
*competing documents*) -- here it's *noise within the right document*
winning over the substance.

**Fix path:** either downweight or filter out boilerplate front matter
during chunking (Vice Chancellor/Dean messages, welcome pages, are
detectable by their position — always the first few pages — and by
their prose style), or bias retrieval toward chunks containing numbers,
dates or defined terms when the query itself contains them ("requirements",
"minimum", "GPA") — a promotional paragraph almost never does.
