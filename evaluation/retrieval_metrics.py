
import csv
import json
from pathlib import Path

from documind.core import search
from documind.hybrid import BM25Retriever, hybrid_search

INDEX_DIR = Path("artifacts/milestone5-index")
QUESTIONS_FILE = Path("evaluation/questions_v3.json")
OUTPUT_DIR = Path("evaluation/results")

K_VALUES = [1, 3, 5]


def calculate_metrics(results, relevant_sources, k):
    """Calculate document-level retrieval metrics."""
    retrieved = [r["source"] for r in results[:k]]
    relevant = set(relevant_sources)

    relevant_found = len(set(retrieved) & relevant)

    recall = relevant_found / len(relevant) if relevant else 0.0
    precision = relevant_found / k

    reciprocal_rank = 0.0

    for rank, source in enumerate(retrieved, start=1):
        if source in relevant:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "recall": recall,
        "precision": precision,
        "reciprocal_rank": reciprocal_rank,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    questions = json.loads(
        QUESTIONS_FILE.read_text(encoding="utf-8")
    )

    chunks = json.loads(
        (INDEX_DIR / "chunks.json").read_text(
            encoding="utf-8"
        )
    )

    bm25 = BM25Retriever(chunks)

    methods = {
        "FAISS": lambda q: search(INDEX_DIR, q, top_k=5),
        "BM25": lambda q: bm25.search(q, top_k=5),
        "Hybrid": lambda q: hybrid_search(
            INDEX_DIR, q, top_k=5
        ),
    }

    detailed_results = []
    summary_results = []

    for method_name, retrieve in methods.items():
        print(f"\nEvaluating {method_name}...")

        method_rows = []

        for item in questions:
            question = item["question"]
            relevant = item["relevant_sources"]

            results = retrieve(question)

            row = {
                "method": method_name,
                "question": question,
                "expected_sources": "; ".join(relevant),
                "retrieved_sources": "; ".join(
                    r["source"] for r in results
                ),
            }

            for k in K_VALUES:
                metrics = calculate_metrics(
                    results, relevant, k
                )

                row[f"recall_at_{k}"] = metrics["recall"]
                row[f"precision_at_{k}"] = metrics["precision"]
                row[f"rr_at_{k}"] = metrics["reciprocal_rank"]

            detailed_results.append(row)
            method_rows.append(row)

        summary = {"method": method_name}

        for k in K_VALUES:
            for metric in ["recall", "precision", "rr"]:
                key = f"{metric}_at_{k}"
                summary[key] = sum(
                    row[key] for row in method_rows
                ) / len(method_rows)

        summary_results.append(summary)

    detailed_path = OUTPUT_DIR / "retrieval_detailed.csv"
    summary_path = OUTPUT_DIR / "retrieval_summary.csv"

    for path, rows in [
        (detailed_path, detailed_results),
        (summary_path, summary_results),
    ]:
        with path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

    print("\nRETRIEVAL EVALUATION SUMMARY")
    print("-" * 75)

    for row in summary_results:
        print(f"\n{row['method']}")

        for k in K_VALUES:
            print(
                f"  Recall@{k}: "
                f"{row[f'recall_at_{k}']:.3f} | "
                f"Precision@{k}: "
                f"{row[f'precision_at_{k}']:.3f} | "
                f"MRR@{k}: "
                f"{row[f'rr_at_{k}']:.3f}"
            )

    print(f"\nDetailed results: {detailed_path}")
    print(f"Summary results: {summary_path}")


if __name__ == "__main__":
    main()
