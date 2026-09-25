"""
FastAPI backend for the RAG assistant.

    uvicorn src.app:app --reload --port 8000

Then open ui/index.html directly in a browser (it calls http://localhost:8000).
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import RagPipeline  # noqa: E402

app = FastAPI(title="University Regulations Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class Citation(BaseModel):
    doc_id: str
    page: int


class Source(BaseModel):
    doc_id: str
    title: str
    page_number: int
    faculty: str | None = None
    approx_pages: bool = False
    fused_score: float
    text_preview: str


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    citations: list[Citation]
    sources: list[Source]
    retrieval_ms: float
    generation_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    pipeline = get_pipeline()
    result = pipeline.answer(req.question, top_k=req.top_k)
    return AskResponse(
        question=result.question,
        answer=result.answer.text,
        refused=result.answer.refused,
        citations=[Citation(doc_id=d, page=p) for d, p in result.answer.citations],
        sources=[
            Source(
                doc_id=c.doc_id,
                title=c.title,
                page_number=c.page_number,
                faculty=c.faculty,
                approx_pages=c.approx_pages,
                fused_score=round(c.fused_score, 3),
                text_preview=c.text[:280],
            )
            for c in result.retrieved
        ],
        retrieval_ms=round(result.retrieval_ms, 1),
        generation_ms=round(result.generation_ms, 1),
    )
