from documind.hybrid import reciprocal_rank_fusion
from documind.hybrid import BM25Retriever, tokenize


def test_tokenization():
    assert tokenize("Data Breach: 24 Hours!") == [
        "data", "breach", "24", "hours"
    ]


def test_bm25_retrieval():
    chunks = [
        {"id": "1", "text": "Vendors must report breaches within 24 hours."},
        {"id": "2", "text": "Employees must complete security training."},
        {"id": "3", "text": "All contracts require legal approval."},
        {"id": "4", "text": "Access rights are reviewed quarterly."},
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search("breaches vendors", top_k=2)

    assert len(results) >= 1
    assert results[0]["id"] == "1"
    assert results[0]["score"] > 0


def test_empty_query():
    retriever = BM25Retriever([
        {"id": "1", "text": "Security policy"}
    ])

    assert retriever.search("") == []


def test_rrf_combines_rankings():
    semantic = [
        {"id": "A", "text": "Security policy"},
        {"id": "B", "text": "Breach notification"},
    ]

    keyword = [
        {"id": "B", "text": "Breach notification"},
        {"id": "C", "text": "Vendor contract"},
    ]

    results = reciprocal_rank_fusion(
        semantic, keyword, top_k=3
    )

    assert len(results) == 3
    assert results[0]["id"] == "B"
    assert results[0]["rrf_score"] > 0


def test_rrf_empty_results():
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_invalid_k():
    import pytest

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], [], k=0)


def test_hybrid_search_integration(tmp_path, monkeypatch):
    import json
    from documind.hybrid import hybrid_search

    chunks = [
        {
            "chunk_id": 0,
            "text": "Vendors report breaches within 24 hours.",
            "source": "policy.txt",
            "page": 1,
        },
        {
            "chunk_id": 1,
            "text": "Employees complete annual training.",
            "source": "policy.txt",
            "page": 1,
        },
        {
            "chunk_id": 2,
            "text": "Contracts require legal approval.",
            "source": "policy.txt",
            "page": 1,
        },
        {
            "chunk_id": 3,
            "text": "Access permissions are reviewed quarterly.",
            "source": "policy.txt",
            "page": 1,
        },
    ]

    (tmp_path / "chunks.json").write_text(
        json.dumps(chunks),
        encoding="utf-8",
    )

    def fake_faiss_search(index_dir, question, top_k):
        return [
            {**chunks[0], "score": 0.9},
            {**chunks[1], "score": 0.3},
        ]

    monkeypatch.setattr(
        "documind.core.search",
        fake_faiss_search,
    )

    results = hybrid_search(
        tmp_path,
        "vendors breaches",
        top_k=2,
    )

    assert results
    assert results[0]["chunk_id"] == 0
    assert "rrf_score" in results[0]
