
import json
from pathlib import Path

from documind.core import search
from documind.hybrid import (
    BM25Retriever,
    reciprocal_rank_fusion,
)

INDEX = Path("artifacts/benchmark-v2-index")

BENCHMARKS = {
    "V2 - Direct": Path("evaluation/questions_v2.json"),
    "V3 - Paraphrased": Path("evaluation/questions_v3.json"),
}

WEIGHTS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def evaluate(hits, relevant):
    sources = [hit["source"] for hit in hits]
    relevant = set(relevant)

    hit1 = int(bool(sources) and sources[0] in relevant)
    recall3 = len(set(sources[:3]) & relevant) / len(relevant)

    mrr = 0.0
    for rank, source in enumerate(sources, 1):
        if source in relevant:
            mrr = 1.0 / rank
            break

    return hit1, recall3, mrr


def main():
    chunks = json.loads(
        (INDEX / "chunks.json").read_text(encoding="utf-8")
    )
    bm25 = BM25Retriever(chunks)

    all_results = {}

    for benchmark_name, path in BENCHMARKS.items():
        questions = json.loads(path.read_text(encoding="utf-8"))

        metrics = {str(w): [] for w in WEIGHTS}

        for item in questions:
            question = item["question"]

            # Retrieve candidate lists once per question.
            semantic = search(INDEX, question, top_k=20)
            keyword = bm25.search(question, top_k=20)

            for hit in semantic + keyword:
                hit["id"] = str(hit["chunk_id"])

            for weight in WEIGHTS:
                hits = reciprocal_rank_fusion(
                    semantic,
                    keyword,
                    top_k=3,
                    semantic_weight=weight,
                    keyword_weight=1.0 - weight,
                )

                metrics[str(weight)].append(
                    evaluate(hits, item["relevant_sources"])
                )

        print(f"\n{benchmark_name}")
        print("-" * 58)
        print(
            f"{'FAISS/BM25':<15}"
            f"{'Hit@1':>10}"
            f"{'Recall@3':>12}"
            f"{'MRR':>10}"
        )

        summary = {}

        for weight in WEIGHTS:
            scores = metrics[str(weight)]
            n = len(scores)

            averages = [
                sum(row[i] for row in scores) / n
                for i in range(3)
            ]

            label = f"{int(weight * 100)}/{int(round((1-weight)*100))}"

            summary[label] = {
                "hit_at_1": averages[0],
                "recall_at_3": averages[1],
                "mrr": averages[2],
            }

            print(
                f"{label:<15}"
                f"{averages[0]:>10.3f}"
                f"{averages[1]:>12.3f}"
                f"{averages[2]:>10.3f}"
            )

        all_results[benchmark_name] = summary

    output = Path("evaluation/weight_results.json")
    output.write_text(
        json.dumps(all_results, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()
