"""
Glue: question -> retrieve -> generate -> Answer, with basic timing so
the app and the eval harness can both report latency.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RetrievalConfig, GenerationConfig  # noqa: E402
from src.retrieve import Retriever, RetrievedChunk  # noqa: E402
from src.generate import generate_answer, Answer  # noqa: E402


@dataclass
class PipelineResult:
    question: str
    answer: Answer
    retrieved: list[RetrievedChunk]
    retrieval_ms: float
    generation_ms: float


class RagPipeline:
    def __init__(
        self,
        retrieval_cfg: RetrievalConfig | None = None,
        generation_cfg: GenerationConfig | None = None,
    ):
        self.retriever = Retriever(retrieval_cfg)
        self.generation_cfg = generation_cfg or GenerationConfig()

    def answer(self, question: str, top_k: int | None = None) -> PipelineResult:
        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(question, top_k=top_k)
        t1 = time.perf_counter()
        answer = generate_answer(question, chunks, self.generation_cfg)
        t2 = time.perf_counter()
        return PipelineResult(
            question=question,
            answer=answer,
            retrieved=chunks,
            retrieval_ms=(t1 - t0) * 1000,
            generation_ms=(t2 - t1) * 1000,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()

    pipeline = RagPipeline()
    result = pipeline.answer(args.question)
    print(f"\nQ: {result.question}\n")
    print(f"A: {result.answer.text}\n")
    print(f"retrieval: {result.retrieval_ms:.1f} ms | generation: {result.generation_ms:.1f} ms")
    print("\nRetrieved:")
    for c in result.retrieved:
        print(f"  [{c.doc_id} p.{c.page_number}] fused={c.fused_score:.3f}")
