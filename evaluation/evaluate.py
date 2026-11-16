
import argparse
import json
from pathlib import Path

from documind.core import search
from documind.hybrid import BM25Retriever, hybrid_search


def evaluate_rankings(results, relevant_sources, k=3):
    """Evaluate retrieval using document-level relevance."""
    sources = [r["source"] for r in results]
    relevant = set(relevant_sources)

    hit_at_1 = int(bool(sources) and sources[0] in relevant)

    recall_at_k = len(
        set(sources[:k]) & relevant
    ) / len(relevant)

    reciprocal_rank = 0.0
    for rank, source in enumerate(sources, start=1):
        if source in relevant:
            reciprocal_rank = 1.0 / rank
            break

    return hit_at_1, recall_at_k, reciprocal_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/benchmark-index"),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("evaluation/questions.json"),
    )
    args = parser.parse_args()

    questions = json.loads(
        args.questions.read_text(encoding="utf-8")
    )

    chunks = json.loads(
        (args.index / "chunks.json").read_text(
            encoding="utf-8"
        )
    )

    bm25 = BM25Retriever(chunks)

    methods = {
        "FAISS": [],
        "BM25": [],
        "Hybrid": [],
    }

    for item in questions:
        question = item["question"]
        relevant = item["relevant_sources"]

        results = {
            "FAISS": search(args.index, question, top_k=3),
            "BM25": bm25.search(question, top_k=3),
            "Hybrid": hybrid_search(
                args.index, question, top_k=3
            ),
        }

        for method, hits in results.items():
            metrics = evaluate_rankings(hits, relevant)
            methods[method].append(metrics)

    print("\nRetrieval Evaluation Results")
    print("-" * 52)
    print(
        f"{'Method':<12} "
        f"{'Hit@1':>10} "
        f"{'Recall@3':>12} "
        f"{'MRR':>10}"
    )

    summary = {}

    for method, metrics in methods.items():
        count = len(metrics)
        averages = [
            sum(values) / count
            for values in zip(*metrics)
        ]

        summary[method] = {
            "hit_at_1": averages[0],
            "recall_at_3": averages[1],
            "mrr": averages[2],
        }

        print(
            f"{method:<12} "
            f"{averages[0]:>10.3f} "
            f"{averages[1]:>12.3f} "
            f"{averages[2]:>10.3f}"
        )

    output = Path("evaluation/results.json")
    output.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()
