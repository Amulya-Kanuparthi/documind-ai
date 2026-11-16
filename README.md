# DocuMind AI — Enterprise Document Retrieval

Milestone 1: CPU-only PDF/TXT ingestion, overlapping word chunks, sentence-transformer embeddings, FAISS cosine similarity, source/page metadata, and CLI retrieval.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
python -m documind.cli index
python -m documind.cli search "How quickly must vendors report a suspected data breach?"
```

First indexing downloads the public embedding model and needs internet access. Subsequent runs can use the cached model. The demo document is fictional; no real contracts or private data are included.

## Milestone 1 architecture

PDF/TXT -> text extraction -> word chunks -> MiniLM embeddings -> FAISS index -> top-k retrieval + source/page.

## Limitations

No OCR for scanned PDFs, no generated answers, no authentication, no document-level access controls, and no hybrid retrieval yet. Do not upload confidential documents to a public repository.

## Next milestones

BM25 + fusion, retrieval benchmarks, cited LLM answers, FastAPI, Streamlit, Docker, and CI integration tests.
