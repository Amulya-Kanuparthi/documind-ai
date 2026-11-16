
import re
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase searchable tokens."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """Keyword-based document retrieval using BM25."""

    def __init__(self, chunks: list[dict]):
        if not chunks:
            raise ValueError("At least one document chunk is required.")

        self.chunks = chunks
        tokenized = [tokenize(chunk["text"]) for chunk in chunks]
        self.model = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the highest-ranked matching chunks."""
        if not query.strip() or top_k <= 0:
            return []

        scores = self.model.get_scores(tokenize(query))
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []
        for index in ranked_indices[:top_k]:
            if scores[index] <= 0:
                continue

            results.append({
                **self.chunks[index],
                "score": float(scores[index]),
            })

        return results

def reciprocal_rank_fusion(
    semantic_results: list[dict],
    keyword_results: list[dict],
    k: int = 60,
    top_k: int = 3,
) -> list[dict]:
    """Combine FAISS and BM25 rankings using RRF."""
    if k <= 0:
        raise ValueError("k must be positive")

    if top_k <= 0:
        return []

    scores = {}
    documents = {}

    for results in (semantic_results, keyword_results):
        seen = set()

        for rank, document in enumerate(results, start=1):
            # Use a stable chunk identifier when available.
            doc_id = str(document.get(
                "id",
                (
                    document.get("source"),
                    document.get("page"),
                    document.get("text"),
                ),
            ))

            if doc_id in seen:
                continue

            seen.add(doc_id)
            scores[doc_id] = (
                scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            )
            documents[doc_id] = document

    ranked_ids = sorted(
        scores,
        key=lambda doc_id: scores[doc_id],
        reverse=True,
    )

    return [
        {
            **documents[doc_id],
            "rrf_score": scores[doc_id],
        }
        for doc_id in ranked_ids[:top_k]
    ]

import json
from pathlib import Path


def hybrid_search(
    index_dir: Path,
    question: str,
    top_k: int = 3,
    candidate_k: int = 20,
) -> list[dict]:
    """Combine FAISS and BM25 retrieval using RRF."""
    from documind.core import search as faiss_search

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if top_k < 1 or candidate_k < 1:
        raise ValueError("top_k and candidate_k must be positive")

    index_dir = Path(index_dir)

    chunks = json.loads(
        (index_dir / "chunks.json").read_text(
            encoding="utf-8"
        )
    )

    if not chunks:
        return []

    # Retrieve more candidates than the final result count.
    limit = min(len(chunks), max(top_k, candidate_k))

    semantic_results = faiss_search(
        index_dir,
        question,
        top_k=limit,
    )

    keyword_retriever = BM25Retriever(chunks)

    keyword_results = keyword_retriever.search(
        question,
        top_k=limit,
    )

    # Normalize identifiers so RRF matches the same chunk
    # across both retrieval methods.
    for result in semantic_results + keyword_results:
        result["id"] = str(result["chunk_id"])

    return reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=top_k,
    )
