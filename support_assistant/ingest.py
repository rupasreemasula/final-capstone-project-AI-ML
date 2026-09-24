"""
Module 3 - Support Assistant: Task 1 (ingestion + embedding + storage)
Also houses the retrieval helper used by graph.py's retrieve_and_answer node
(retrieval always runs for real in both MOCK_LLM modes - no API key needed).

Loads the 8 Zepto policy documents, chunks them (one chunk per document -
each doc is a single short paragraph, so a per-document chunk is exactly
the "simple per-document chunk... fine given their length" option the task
allows), embeds each chunk locally with sentence-transformers'
all-MiniLM-L6-v2, and stores the embeddings in a persistent ChromaDB
collection on disk (support_assistant/chroma_db/).

Run standalone to (re)build the collection:
    python ingest.py
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "docs"
CHROMA_DIR = HERE / "chroma_db"
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    # First call downloads the ~80MB model from Hugging Face Hub (free, no
    # account/API key needed) and caches it locally; later calls reuse the
    # cache and need no network - same "download once, cache forever"
    # pattern as sns.load_dataset('titanic') in Module 2.
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    # ChromaDB defaults to L2 distance; the task requires cosine similarity,
    # so the collection is explicitly configured for it via hnsw:space.
    return _get_chroma_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> dict[str, str]:
    """One chunk per document: {'doc_01': '<full text>', ...}."""
    docs: dict[str, str] = {}
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        docs[path.stem] = path.read_text(encoding="utf-8").strip()
    return docs


def ingest(force: bool = False) -> int:
    """
    Load all 8 docs, embed each with all-MiniLM-L6-v2, store in the ChromaDB
    collection. Idempotent: skips re-embedding if already fully populated,
    unless force=True. Returns the resulting document count.
    """
    collection = get_collection()
    docs = load_documents()

    if not force and collection.count() >= len(docs):
        return collection.count()

    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    model = get_embedding_model()
    ids = list(docs.keys())
    texts = list(docs.values())
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source_doc": doc_id} for doc_id in ids],
    )
    return collection.count()


def retrieve_top_chunks(query: str, n_results: int = 3) -> list[dict]:
    """
    Embed `query` and retrieve the top-n most similar chunks from ChromaDB
    via cosine similarity. Runs for real in BOTH MOCK_LLM modes - no API key
    or network call needed (local embedding model + local vector store).
    """
    collection = get_collection()
    model = get_embedding_model()
    query_embedding = model.encode([query], show_progress_bar=False).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )
    chunks = []
    for doc_id, text, distance in zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ):
        chunks.append({"id": doc_id, "text": text, "distance": distance})
    return chunks


if __name__ == "__main__":
    n = ingest(force=True)
    print(f"Ingested {n} document(s) into ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_DIR}")

    # Quick sanity check
    sample = retrieve_top_chunks("How much does delivery cost?", n_results=3)
    print("\nSample retrieval for 'How much does delivery cost?':")
    for chunk in sample:
        print(f"  [{chunk['id']}] distance={chunk['distance']:.4f}  {chunk['text'][:80]}...")
