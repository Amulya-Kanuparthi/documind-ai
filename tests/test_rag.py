
from pathlib import Path
from unittest.mock import patch
from documind.rag import validate_citations
from documind.rag import normalize_citations
import pytest

from documind.rag import (
    retrieve_context,
    build_context,
    ask_documind,
)


SAMPLE_CHUNKS = [
    {
        "text": (
            "Terminated employee accounts must "
            "be disabled within 24 hours."
        ),
        "source": "access_control.txt",
        "page": 1,
        "chunk_id": 0,
    }
]


def test_build_context():
    context = build_context(SAMPLE_CHUNKS)

    assert "access_control.txt" in context
    assert "Page: 1" in context
    assert "24 hours" in context


def test_empty_question():
    with pytest.raises(ValueError):
        retrieve_context(
            "",
            Path("artifacts/benchmark-index"),
        )


def test_invalid_top_k():
    with pytest.raises(ValueError):
        retrieve_context(
            "Test question",
            Path("artifacts/benchmark-index"),
            top_k=0,
        )


def test_invalid_retrieval_method():
    with pytest.raises(ValueError):
        retrieve_context(
            "Test question",
            Path("artifacts/benchmark-index"),
            method="invalid",
        )


@patch("documind.rag.generate_answer")
@patch("documind.rag.retrieve_context")
def test_grounded_answer(mock_retrieve, mock_generate):
    mock_retrieve.return_value = SAMPLE_CHUNKS
    mock_generate.return_value = "Within 24 hours."

    result = ask_documind(
        question="When must accounts be disabled?",
        index_dir=Path("artifacts/benchmark-index"),
    )

    assert result["answer"] == "Within 24 hours."
    assert result["sources"][0]["source"] == "access_control.txt"

    mock_generate.assert_called_once()


@patch("documind.rag.generate_answer")
@patch("documind.rag.retrieve_context")
def test_no_retrieved_documents(mock_retrieve, mock_generate):
    mock_retrieve.return_value = []

    result = ask_documind(
        question="What is the company revenue?",
        index_dir=Path("artifacts/benchmark-index"),
    )

    assert result["answer"] == (
        "Insufficient information in the documents."
    )
    assert result["sources"] == []

    mock_generate.assert_not_called()

def test_valid_citation():
    assert validate_citations(
        "Accounts must be disabled within 24 hours [1].",
        3,
    )


def test_multiple_valid_citations():
    assert validate_citations(
        "MFA is required [1]. Access is reviewed [2].",
        3,
    )


def test_invalid_citation():
    assert not validate_citations(
        "Accounts must be disabled [5].",
        3,
    )


def test_missing_citation():
    assert not validate_citations(
        "Accounts must be disabled within 24 hours.",
        3,
    )


def test_insufficient_information():
    assert validate_citations(
        "Insufficient information in the documents.",
        0,
    )


def test_normalize_citations():
    answer = (
        "Vendors must report breaches within "
        "24 hours【1】."
    )

    normalized = normalize_citations(answer)

    assert normalized == (
        "Vendors must report breaches within "
        "24 hours[1]."
    )

    assert validate_citations(normalized, 3)
