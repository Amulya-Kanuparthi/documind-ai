
import json
from pathlib import Path

from documind.core import search
from documind.hybrid import BM25Retriever, hybrid_search

INDEX_DIR = Path("artifacts/milestone5-index")
QUESTIONS_FILE = Path("evaluation/questions_v3.json")

questions = json.loads(
    QUESTIONS_FILE.read_text(encoding="utf-8")
)

chunks = json.loads(
    (INDEX_DIR / "chunks.json").read_text(encoding="utf-8")
)

bm25 = BM25Retriever(chunks)

methods = {
    "FAISS": lambda q: search(INDEX_DIR, q, top_k=5),
    "BM25": lambda q: bm25.search(q, top_k=5),
    "Hybrid": lambda q: hybrid_search(
        INDEX_DIR, q, top_k=5
    ),
}

for method, retrieve in methods.items():
    print(f"\n{'=' * 55}")
    print(f"{method} - Retrieval Failures")
    print("=" * 55)

    failures = 0

    for item in questions:
        question = item["question"]
        relevant = set(item["relevant_sources"])

        results = retrieve(question)
        sources = [r["source"] for r in results]

        if not sources or sources[0] not in relevant:
            failures += 1

            print(f"\nQuestion: {question}")
            print(f"Expected: {sorted(relevant)}")
            print(f"Top 5: {sources}")

            matching_ranks = [
                i for i, source in enumerate(sources, 1)
                if source in relevant
            ]

            print(
                "Relevant rank:",
                matching_ranks[0] if matching_ranks else "Not in top 5"
            )

    print(f"\nTotal Hit@1 failures: {failures}")
