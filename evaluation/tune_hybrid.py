
import json
from pathlib import Path

from documind.hybrid import hybrid_search

INDEX_DIR = Path("artifacts/milestone5-index")
QUESTIONS_FILE = Path("evaluation/questions_v3.json")

WEIGHTS = [
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
]


def main():
    questions = json.loads(
        QUESTIONS_FILE.read_text(encoding="utf-8")
    )

    print("\nHYBRID RETRIEVAL WEIGHT TUNING")
    print("-" * 65)

    print(
        f"{'FAISS':<10}"
        f"{'BM25':<10}"
        f"{'Recall@1':<12}"
        f"{'Recall@3':<12}"
        f"{'Recall@5':<12}"
        f"{'MRR@5':<10}"
    )

    for semantic_weight, keyword_weight in WEIGHTS:
        hits = {1: 0, 3: 0, 5: 0}
        reciprocal_rank_sum = 0.0

        for item in questions:
            results = hybrid_search(
                INDEX_DIR,
                item["question"],
                top_k=5,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
            )

            retrieved = [
                r["source"] for r in results
            ]

            relevant = set(item["relevant_sources"])

            for k in hits:
                if any(
                    source in relevant
                    for source in retrieved[:k]
                ):
                    hits[k] += 1

            for rank, source in enumerate(
                retrieved, start=1
            ):
                if source in relevant:
                    reciprocal_rank_sum += 1 / rank
                    break

        total = len(questions)

        print(
            f"{semantic_weight:<10.2f}"
            f"{keyword_weight:<10.2f}"
            f"{hits[1] / total:<12.3f}"
            f"{hits[3] / total:<12.3f}"
            f"{hits[5] / total:<12.3f}"
            f"{reciprocal_rank_sum / total:<10.3f}"
        )


if __name__ == "__main__":
    main()
