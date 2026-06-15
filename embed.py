"""
Stage 3 — Embedding + Vector Store (and Stage 4 — Retrieval).

Pipeline position (see architecture.png):
    ingest.py  ->  chunk.py  ->  [THIS FILE]  ->  generate (Stage 5)
    Reviews        Chunks         embed + store      grounded answer
                                  + retrieve

Per planning.md Retrieval Approach:
  - Embedding model: all-MiniLM-L6-v2 via sentence-transformers (384-dim, local,
    no API key, no rate limits).
  - Vector store: a local, persistent ChromaDB collection.
  - Retrieval: semantic similarity search returning the top-k (default k=4)
    most relevant chunks, each with its source metadata for later citation.

Usage:
    python embed.py            # build the index, then run a sample query
    python embed.py --rebuild  # wipe and rebuild the collection from scratch

    from embed import query_vector_db
    results = query_vector_db("How hard are Kreinovich's exams?", k=4)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunk import Chunk, chunk_reviews
from ingest import load_documents

MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIR = Path(__file__).parent / "chroma_db"   # gitignored
COLLECTION_NAME = "rmp_reviews"
DEFAULT_K = 4

# Module-level caches so we load the model / connect to Chroma only once.
_model: SentenceTransformer | None = None
_client: chromadb.ClientAPI | None = None


def get_model() -> SentenceTransformer:
    """Load (once) the local sentence-transformers embedding model."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_client() -> chromadb.ClientAPI:
    """Connect (once) to a persistent on-disk ChromaDB instance.

    PersistentClient writes the index to PERSIST_DIR so the embeddings survive
    between runs — you embed once, then query as many times as you want.
    """
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return _client


def get_collection() -> chromadb.Collection:
    """Get (or create) the reviews collection.

    metadata={"hnsw:space": "cosine"} tells Chroma to rank by cosine distance.
    all-MiniLM embeddings are direction-based, so cosine is the right metric
    (smaller distance = more similar). We pass our own vectors, so Chroma does
    NOT attach a default embedding function of its own.
    """
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings into normalized 384-dim vectors."""
    model = get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,   # unit-length vectors -> clean cosine scores
        show_progress_bar=len(texts) > 50,
    )
    return vectors.tolist()


def build_index(rebuild: bool = False) -> chromadb.Collection:
    """Embed every chunk from the ingestion+chunking pipeline and store it.

    Each Chroma record carries:
      - id:        stable chunk id (e.g. 'olac_fuentes.txt::r1::c0')
      - document:  the chunk text that was embedded
      - embedding: the 384-dim vector
      - metadata:  source_file, professor, course, chunk_index, ratings, etc.
                   (source_file + chunk_index are the attribution fields you'll
                   surface in citations later.)
    """
    if rebuild:
        # Drop any prior collection so we start clean (ignore if absent).
        try:
            get_client().delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = get_collection()

    chunks: list[Chunk] = chunk_reviews(load_documents())
    print(f"Embedding {len(chunks)} chunks with {MODEL_NAME} ...")

    embeddings = embed_texts([c.text for c in chunks])

    # upsert = insert-or-overwrite by id, so re-running never duplicates rows.
    collection.upsert(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[c.metadata for c in chunks],
    )

    print(f"Stored {collection.count()} chunks in collection '{COLLECTION_NAME}'.")
    return collection


def query_vector_db(query: str, k: int = DEFAULT_K) -> list[dict]:
    """Return the top-k most relevant chunks for a query string.

    Each result dict has:
      - text:       the chunk content (feed this to the LLM as context)
      - metadata:   full source metadata (use source_file + professor for citation)
      - distance:   cosine distance (lower = closer)
      - similarity: 1 - distance, an easier-to-read 0..1 relevance score
      - source:     short "professor — source_file" label for quick attribution
    """
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError(
            "The vector store is empty. Run `python embed.py --rebuild` first."
        )

    query_embedding = embed_texts([query])

    # n_results=k asks for the k nearest neighbours by cosine distance.
    raw = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns parallel lists nested one level per query; we sent one query.
    results: list[dict] = []
    for doc, meta, dist in zip(
        raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        results.append(
            {
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "similarity": round(1 - dist, 4),
                "source": f"{meta.get('professor', '?')} - {meta.get('source_file', '?')}",
            }
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the vector store and test a query.")
    parser.add_argument("--rebuild", action="store_true", help="Wipe and rebuild from scratch.")
    parser.add_argument(
        "--query",
        default="What advice do students give for passing Dr. Kreinovich's exams in CS3350?",
        help="A query to test retrieval after building.",
    )
    parser.add_argument("-k", type=int, default=DEFAULT_K, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    build_index(rebuild=args.rebuild)

    print(f"\nQuery: {args.query!r}  (k={args.k})\n" + "=" * 70)
    for i, r in enumerate(query_vector_db(args.query, k=args.k), 1):
        print(f"\n[{i}] similarity={r['similarity']}  source={r['source']}")
        print(r["text"])
