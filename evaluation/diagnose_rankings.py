
import argparse
import json
from pathlib import Path

from documind.core import search
from documind.hybrid import BM25Retriever, hybrid_search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/benchmark-v2-index"),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("evaluation/questions_v3.json"),
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
    diagnostics = []

    for number, item in enumerate(questions, start=1):
        question = item["question"]
        relevant = set(item["relevant_sources"])

        results = {
            "FAISS": search(args.index, question, top_k=3),
            "BM25": bm25.search(question, top_k=3),
            "Hybrid": hybrid_search(
                args.index, question, top_k=3
            ),
        }

        rankings = {
            method: [
                hit["source"] for hit in hits
            ]
            for method, hits in results.items()
        }

        correct = {
            method: bool(sources)
            and sources[0] in relevant
            for method, sources in rankings.items()
        }

        if not all(correct.values()):
            record = {
                "question_number": number,
                "question": question,
                "expected": sorted(relevant),
                "correct_at_1": correct,
                "rankings": rankings,
            }
            diagnostics.append(record)

            print(f"\nQuestion {number}: {question}")
            print(f"Expected: {sorted(relevant)}")

            for method, sources in rankings.items():
                status = (
                    "PASS" if correct[method] else "FAIL"
                )
                print(f"\n{method} [{status}]")

                for rank, source in enumerate(
                    sources, start=1
                ):
                    print(f"  {rank}. {source}")

            print("-" * 60)

    output = Path("evaluation/diagnostics_v3.json")
    output.write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )

    print(
        f"\nQuestions with at least one "
        f"incorrect top result: {len(diagnostics)}"
    )
    print(f"Saved diagnostics to {output}")


if __name__ == "__main__":
    main()
