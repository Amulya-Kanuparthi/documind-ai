
from __future__ import annotations
from functools import lru_cache
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import faiss
import pymupdf
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: int

def load_documents(folder: Path) -> list[tuple[str, int, str]]:
    records = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(folder))
        if path.suffix.lower() == ".txt":
            records.append((relative, 1, path.read_text(encoding="utf-8")))
        elif path.suffix.lower() == ".pdf":
            with pymupdf.open(path) as doc:
                for i, page in enumerate(doc):
                    records.append((relative, i + 1, page.get_text()))
    return records

def split_text(text: str, size: int = 120, overlap: int = 25) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step) if words[i:i + size]]

def make_chunks(records: list[tuple[str, int, str]]) -> list[Chunk]:
    result = []
    for source, page, text in records:
        for piece in split_text(text):
            result.append(Chunk(piece, source, page, len(result)))
    return result

def build_index(data_dir: Path, index_dir: Path, model_name: str = MODEL) -> int:
    chunks = make_chunks(load_documents(data_dir))
    if not chunks:
        raise ValueError(f"No extractable text found in {data_dir}")
    model = get_embedding_model(model_name)
    vectors = model.encode([c.text for c in chunks], normalize_embeddings=True, convert_to_numpy=True)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "vectors.faiss"))
    (index_dir / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks], indent=2), encoding="utf-8")
    (index_dir / "config.json").write_text(json.dumps({"model": model_name}), encoding="utf-8")
    return len(chunks)

def search(index_dir: Path, question: str, top_k: int = 3) -> list[dict]:
    if not question.strip():
        raise ValueError("Question cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_dir / "vectors.faiss"))
    model = get_embedding_model(config["model"])
    vector = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)
    scores, ids = index.search(np.ascontiguousarray(vector, dtype=np.float32), min(top_k, len(chunks)))
    return [{**chunks[int(idx)], "score": round(float(score), 4)} for score, idx in zip(scores[0], ids[0]) if idx >= 0]

@lru_cache(maxsize=2)
def get_embedding_model(model_name: str):
    return SentenceTransformer(model_name, device="cpu")