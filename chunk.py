"""
Stage 2 — Chunking.

Turns the structured Review records from ingest.py into embedding-ready chunks.

Per planning.md: chunk size 500 chars, overlap 100 chars. Most RMP reviews are
short and fit in a single chunk; only long reviews are actually split. When a
review IS split, the "Professor | University | Course" context header is
prepended to every resulting chunk so attribution survives the split (the
mitigation for entity confusion described in planning.md).

This is a native recursive character splitter — same idea as LangChain's
RecursiveCharacterTextSplitter (try paragraph, then sentence, then word, then
character boundaries) but with no extra dependency.

Run directly to preview the chunks:
    python chunk.py
"""

from __future__ import annotations

from dataclasses import dataclass

from ingest import Review, load_documents

CHUNK_SIZE = 500       # characters
CHUNK_OVERLAP = 100    # characters

# Tried in order: keep whole paragraphs together, then sentences, then words,
# then (last resort) raw characters.
SEPARATORS = ["\n\n", ". ", "\n", " ", ""]


@dataclass
class Chunk:
    """A single embedding-ready unit of text plus its provenance metadata."""

    id: str
    text: str
    metadata: dict


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Split text into atomic pieces each <= chunk_size, preferring high-level
    separators (paragraphs) before falling back to finer ones (words, chars)."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep, *rest = separators

    if sep == "":
        # Hard character split — last resort for a token with no break points.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    pieces: list[str] = []
    for part in text.split(sep):
        if not part:
            continue
        if len(part) <= chunk_size:
            pieces.append(part)
        else:
            # Still too big — recurse with the next finer separator.
            pieces.extend(_split_recursive(part, chunk_size, rest))
    return pieces


def _merge_with_overlap(
    pieces: list[str], chunk_size: int, overlap: int, sep: str = " "
) -> list[str]:
    """Greedily pack atomic pieces into chunks <= chunk_size, carrying `overlap`
    characters from the end of one chunk into the start of the next."""
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    def joined(parts: list[str]) -> str:
        return sep.join(parts).strip()

    for piece in pieces:
        added = len(piece) + (len(sep) if current else 0)
        if current and length + added > chunk_size:
            chunks.append(joined(current))
            # Drop pieces from the front until we're within the overlap budget.
            while current and length > overlap:
                removed = current.pop(0)
                length -= len(removed) + len(sep)
        current.append(piece)
        length += len(piece) + (len(sep) if len(current) > 1 else 0)

    if current:
        chunks.append(joined(current))
    return [c for c in chunks if c]


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split a raw string into overlapping chunks no larger than chunk_size."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    pieces = _split_recursive(text, chunk_size, SEPARATORS)
    return _merge_with_overlap(pieces, chunk_size, overlap)


def _context_header(review: Review) -> str:
    """The attribution line prepended to every chunk of a review."""
    return (
        f"Professor: {review.professor} | University: {review.university} | "
        f"Course: {review.course or 'N/A'}"
    )


def chunk_reviews(
    reviews: list[Review],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Convert Review records into embedding-ready Chunk records.

    The context header is reserved out of the size budget and re-attached to
    each body chunk, so the professor/course attribution is never split off.
    """
    chunks: list[Chunk] = []

    for review in reviews:
        header = _context_header(review)
        tags = f"Tags: {', '.join(review.tags)}" if review.tags else ""

        # Reserve room for the header (+newline) so header + body stays <= size.
        body_budget = max(chunk_size - len(header) - 1, 100)
        body_chunks = chunk_text(review.text, body_budget, overlap)

        for j, body in enumerate(body_chunks):
            parts = [header, body]
            # Attach tags only to the final chunk of the review.
            if tags and j == len(body_chunks) - 1:
                parts.append(tags)
            text = "\n".join(parts)

            metadata = review.metadata()
            metadata["chunk_index"] = j
            chunk_id = f"{review.source_file}::r{review.review_index}::c{j}"

            chunks.append(Chunk(id=chunk_id, text=text, metadata=metadata))

    return chunks


if __name__ == "__main__":
    reviews = load_documents()
    chunks = chunk_reviews(reviews)

    split_reviews = len({c.id.rsplit("::c", 1)[0] for c in chunks if c.metadata["chunk_index"] > 0})

    print(f"{len(reviews)} reviews -> {len(chunks)} chunks")
    print(f"Reviews that needed splitting (>1 chunk): {split_reviews}")
    sizes = [len(c.text) for c in chunks]
    print(f"Chunk char length: min={min(sizes)} max={max(sizes)} avg={sum(sizes)//len(sizes)}")

    print("\n--- Sample chunk ---\n")
    print("id:", chunks[0].id)
    print(chunks[0].text)
    print("\nmetadata:", chunks[0].metadata)
