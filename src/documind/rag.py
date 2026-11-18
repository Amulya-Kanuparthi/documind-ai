import re
from pathlib import Path

from documind.core import search
from documind.hybrid import hybrid_search
from documind.llm import generate_answer


def retrieve_context(
    question: str,
    index_dir: Path,
    method: str = "faiss",
    top_k: int = 3,
) -> list[dict]:
    """Retrieve supporting document chunks."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    if method == "faiss":
        return search(index_dir, question, top_k=top_k)

    if method == "hybrid":
        return hybrid_search(
            index_dir, question, top_k=top_k
        )

    raise ValueError(
        "Method must be 'faiss' or 'hybrid'."
    )


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks with source identifiers."""

    sections = []

    for i, chunk in enumerate(chunks, start=1):
        sections.append(
            f"[{i}] Source: {chunk['source']}, "
            f"Page: {chunk['page']}\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(sections)

def normalize_citations(answer: str) -> str:
    """Convert alternative citation brackets to [number]."""

    answer = re.sub(
        r"【\s*(\d+)\s*】",
        r"[\1]",
        answer,
    )

    return answer


def validate_citations(answer: str, source_count: int) -> bool:
    """Check that citation references point to retrieved sources."""

    if answer.strip() == (
        "Insufficient information in the documents."
    ):
        return True

    citations = re.findall(r"\[(\d+)\]", answer)

    if not citations:
        return False

    return all(
        1 <= int(citation) <= source_count
        for citation in citations
    )



def ask_documind(
    question: str,
    index_dir: Path,
    model_name: str = "openai/gpt-oss-20b",
    method: str = "faiss",
    top_k: int = 3,
) -> dict:
    """Retrieve evidence and generate a cited answer."""

    chunks = retrieve_context(
        question=question,
        index_dir=index_dir,
        method=method,
        top_k=top_k,
    )

    if not chunks:
        return {
            "question": question,
            "answer": "Insufficient information in the documents.",
            "sources": [],
            "citation_valid": True,
        }

    context = build_context(chunks)

    answer = generate_answer(
        question=question,
        context=context,
        model_name=model_name,
    )
    answer = normalize_citations(answer)

    sources = [
        {
            "citation": i,
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "evidence": chunk["text"],
        }
        for i, chunk in enumerate(chunks, start=1)
    ]

    citation_valid = validate_citations(
        answer,
        len(sources),
    )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "citation_valid": citation_valid,
    }
