
import argparse
from pathlib import Path

from documind.core import build_index, search
from documind.hybrid import hybrid_search


def print_results(results, score_field="score"):
    if not results:
        print("No matching passages found.")
        return

    for rank, hit in enumerate(results, start=1):
        score = hit.get(score_field, 0.0)

        print(
            f"\n#{rank} | {hit['source']} "
            f"| page {hit['page']} "
            f"| {score_field} {score:.4f}"
        )

        print(hit["text"])


def main():
    parser = argparse.ArgumentParser(
        description="DocuMind document retrieval"
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # Existing document indexing command
    build = sub.add_parser("index")

    build.add_argument(
        "--data",
        type=Path,
        default=Path("data/sample"),
    )

    build.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/index"),
    )

    # Existing FAISS semantic search command
    ask = sub.add_parser("search")

    ask.add_argument("question")

    ask.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/index"),
    )

    ask.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    # New hybrid retrieval command
    hybrid = sub.add_parser("hybrid-search")

    hybrid.add_argument("question")

    hybrid.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/index"),
    )

    hybrid.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    hybrid.add_argument(
        "--candidate-k",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    if args.command == "index":
        count = build_index(args.data, args.output)
        print(f"Indexed {count} chunks")

    elif args.command == "search":
        results = search(
            args.index,
            args.question,
            args.top_k,
        )

        print_results(results, "score")

    elif args.command == "hybrid-search":
        results = hybrid_search(
            args.index,
            args.question,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
        )

        print_results(results, "rrf_score")


if __name__ == "__main__":
    main()
