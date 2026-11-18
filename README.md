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


## Interactive Document Question Answering

DocuMind AI is an enterprise document intelligence system
built using Retrieval-Augmented Generation (RAG).

It combines:
- FAISS semantic vector retrieval
- BM25 keyword retrieval
- Weighted Reciprocal Rank Fusion (RRF)
- Groq-hosted LLM answer generation
- Source citations and citation validation
- Hallucination prevention
- Automated regression testing

### Architecture

Document Collection
    |
    v
Document Parsing and Chunking
    |
    v
Embedding Generation
    |
    v
FAISS Vector Index
    |
    +---- BM25 Keyword Retrieval
    |
    v
Hybrid Retrieval (Weighted RRF)
    |
    v
Retrieved Evidence
    |
    v
Groq LLM
    |
    v
Grounded Answer + Source Citations

### Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

Never commit API keys to GitHub.

### Run the Interactive CLI

FAISS retrieval:

```powershell
python -m documind.chat --index .\artifacts\benchmark-index --method faiss
```

Hybrid retrieval:

```powershell
python -m documind.chat --index .\artifacts\benchmark-index --method hybrid
```

The index must be created before launching the CLI.

### Example

Question:
When must terminated employee accounts be disabled?

Answer:
Terminated employee accounts must be disabled within 24 hours. [1]

Source:
[1] access_control.txt (Page 1)

### Hallucination Prevention

When the retrieved documents do not contain sufficient
information, DocuMind AI returns:

"Insufficient information in the documents."

### Testing

Run the complete automated test suite:

```powershell
python -m pytest -q
```

Current result: 32 tests passed.

### Evaluation

Retrieval performance is evaluated using:

- Hit@1
- Recall@3
- Mean Reciprocal Rank (MRR)

The project includes direct and paraphrased question
benchmarks, ranking diagnostics, and fusion-weight tuning.

### Current Status

Milestone 1: Document ingestion and semantic retrieval
Milestone 2: Hybrid retrieval and evaluation
Milestone 3: RAG question answering, citations, and CLI

All three milestones have been implemented locally.

