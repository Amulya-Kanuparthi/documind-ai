
import argparse
from pathlib import Path

from documind.rag import ask_documind


def main():
    parser = argparse.ArgumentParser(
        description="DocuMind AI interactive document assistant"
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/benchmark-index"),
        help="Path to the document index",
    )
    parser.add_argument(
        "--method",
        choices=["faiss", "hybrid"],
        default="faiss",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-20b",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    if not args.index.is_dir():
        parser.error(f"Index directory not found: {args.index}")

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    print("\n" + "=" * 55)
    print("DOCUMIND AI - Enterprise Document Intelligence")
    print("=" * 55)
    print(f"Retrieval: {args.method.upper()}")
    print(f"Model: {args.model}")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            result = ask_documind(
                question=question,
                index_dir=args.index,
                model_name=args.model,
                method=args.method,
                top_k=args.top_k,
            )

            print(f"\nDocuMind: {result['answer']}\n")

            sources = result.get("sources", [])

            if sources:
                print("Retrieved Sources:")
                for i, source in enumerate(sources, start=1):
                    print(
                        f"[{i}] {source['source']} "
                        f"(Page {source['page']})"
                    )
                print()

        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()
