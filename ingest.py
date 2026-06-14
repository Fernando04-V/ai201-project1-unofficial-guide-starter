"""
Stage 1 — Document Ingestion and Cleaning.

Loads the Rate My Professors .txt files from documents/, parses the
file-level header (professor, university, ratings) and each individual
[REVIEW N] block into a structured record.

Each Review is emitted as its own record with the professor and university
attached, so that downstream chunking/embedding never confuses one professor's
reviews with another's (see "Semantic Mixing of Professor Names" in planning.md).

Run directly to preview what was parsed:
    python ingest.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).parent / "documents"

# Splits the file into its header section and its reviews section.
REVIEWS_DELIMITER = "--- START OF REVIEWS ---"

# Matches "[REVIEW 1]", "[REVIEW 12]", etc. — used to split the body into blocks.
REVIEW_HEADER_RE = re.compile(r"^\[REVIEW\s+\d+\]\s*$", re.MULTILINE)


@dataclass
class Review:
    """One student review, carrying both file-level and review-level metadata."""

    # File-level metadata (repeated on every review from the same file).
    source: str            # e.g. "Rate My Professors"
    professor: str         # e.g. "Dr. Vladik Kreinovich"
    university: str
    department: str
    overall_rating: float | None
    difficulty_rating: float | None
    would_take_again_pct: int | None

    # Review-level metadata.
    review_index: int
    date: str | None
    course: str | None
    quality: float | None
    difficulty: float | None
    grade: str | None
    would_take_again: str | None
    tags: list[str] = field(default_factory=list)
    text: str = ""

    # Provenance — used for citations in the generation stage.
    source_file: str = ""

    def to_chunk_text(self) -> str:
        """
        Render this review as a self-contained string for chunking/embedding.

        The professor, university, and course are prepended so each chunk stays
        attributable even after it is split — this is the mitigation for entity
        confusion described in planning.md.
        """
        header = (
            f"Professor: {self.professor} | University: {self.university} | "
            f"Course: {self.course or 'N/A'}"
        )
        ratings = (
            f"Review quality: {self.quality} | Difficulty: {self.difficulty} | "
            f"Grade: {self.grade or 'N/A'}"
        )
        tags = f"Tags: {', '.join(self.tags)}" if self.tags else ""
        parts = [header, ratings, self.text.strip()]
        if tags:
            parts.append(tags)
        return "\n".join(p for p in parts if p)

    def metadata(self) -> dict:
        """Flat metadata dict suitable for a ChromaDB record (no None values)."""
        raw = {
            "source": self.source,
            "professor": self.professor,
            "university": self.university,
            "department": self.department,
            "course": self.course,
            "date": self.date,
            "quality": self.quality,
            "difficulty": self.difficulty,
            "grade": self.grade,
            "would_take_again": self.would_take_again,
            "tags": ", ".join(self.tags) if self.tags else None,
            "source_file": self.source_file,
            "review_index": self.review_index,
        }
        # ChromaDB rejects None values, so drop empty keys.
        return {k: v for k, v in raw.items() if v is not None}


def _parse_float(value: str) -> float | None:
    """Pull the first number out of strings like '4.5 / 5.0' or '4.0'."""
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return float(match.group()) if match else None


def _parse_header(header_text: str) -> dict:
    """Parse the top-of-file 'KEY: value' lines into a dict of file metadata."""
    fields: dict[str, str] = {}
    for line in header_text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip().upper()] = value.strip()

    wta = fields.get("WOULD TAKE AGAIN", "")
    wta_match = re.search(r"\d+", wta)

    return {
        "source": fields.get("PRODUCER/SOURCE", "Rate My Professors"),
        "professor": fields.get("PROFESSOR", "Unknown"),
        "university": fields.get("UNIVERSITY", "Unknown"),
        "department": fields.get("DEPARTMENT", "Unknown"),
        "overall_rating": _parse_float(fields.get("OVERALL RATING", "")),
        "difficulty_rating": _parse_float(fields.get("DIFFICULTY RATING", "")),
        "would_take_again_pct": int(wta_match.group()) if wta_match else None,
    }


def _parse_kv_line(line: str) -> dict[str, str]:
    """Parse a pipe-separated 'Key: value | Key: value' line into a dict."""
    out: dict[str, str] = {}
    for part in line.split("|"):
        if ":" in part:
            key, _, value = part.partition(":")
            out[key.strip().lower()] = value.strip()
    return out


def _parse_review_block(block: str, index: int) -> Review | None:
    """Parse one [REVIEW N] block (header already stripped) into a Review.

    Returns None if the block has no review text (skips blank/garbage blocks).
    """
    meta: dict[str, str] = {}
    text_lines: list[str] = []
    tags: list[str] = []
    in_text = False

    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()
        if lower.startswith("review text:"):
            in_text = True
            text_lines.append(stripped.split(":", 1)[1].strip())
        elif lower.startswith("tags:"):
            in_text = False
            tag_str = stripped.split(":", 1)[1]
            tags = [t.strip() for t in tag_str.split(",") if t.strip()]
        elif in_text:
            # Continuation of a multi-line review body.
            text_lines.append(stripped)
        elif lower.startswith("course:"):
            meta["course"] = stripped.split(":", 1)[1].strip()
        elif lower.startswith("date:"):
            meta["date"] = stripped.split(":", 1)[1].strip()
        else:
            # Pipe-separated metadata lines (Quality/Difficulty, Attendance/Grade...).
            meta.update(_parse_kv_line(stripped))

    text = " ".join(text_lines).strip()
    if not text:
        return None

    return Review(
        source="",  # filled in by caller from file header
        professor="",
        university="",
        department="",
        overall_rating=None,
        difficulty_rating=None,
        would_take_again_pct=None,
        review_index=index,
        date=meta.get("date"),
        course=meta.get("course"),
        quality=_parse_float(meta.get("quality", "")),
        difficulty=_parse_float(meta.get("difficulty", "")),
        grade=meta.get("grade"),
        would_take_again=meta.get("would take again"),
        tags=tags,
        text=text,
    )


def parse_file(path: Path) -> list[Review]:
    """Parse a single .txt document into a list of Review records."""
    raw = path.read_text(encoding="utf-8")

    if REVIEWS_DELIMITER in raw:
        header_text, body = raw.split(REVIEWS_DELIMITER, 1)
    else:
        header_text, body = raw, ""

    file_meta = _parse_header(header_text)

    # Split the body on each [REVIEW N] header; first piece is pre-review noise.
    blocks = REVIEW_HEADER_RE.split(body)[1:]

    reviews: list[Review] = []
    for i, block in enumerate(blocks, start=1):
        review = _parse_review_block(block, i)
        if review is None:
            continue
        # Stamp file-level metadata onto the review.
        review.source = file_meta["source"]
        review.professor = file_meta["professor"]
        review.university = file_meta["university"]
        review.department = file_meta["department"]
        review.overall_rating = file_meta["overall_rating"]
        review.difficulty_rating = file_meta["difficulty_rating"]
        review.would_take_again_pct = file_meta["would_take_again_pct"]
        review.source_file = path.name
        reviews.append(review)

    return reviews


def load_documents(documents_dir: Path | str = DOCUMENTS_DIR) -> list[Review]:
    """Load and parse every .txt file in the documents directory."""
    documents_dir = Path(documents_dir)
    txt_files = sorted(documents_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {documents_dir.resolve()}")

    reviews: list[Review] = []
    for path in txt_files:
        reviews.extend(parse_file(path))
    return reviews


if __name__ == "__main__":
    reviews = load_documents()
    files = sorted({r.source_file for r in reviews})

    print(f"Parsed {len(reviews)} reviews from {len(files)} file(s):\n")
    for name in files:
        count = sum(1 for r in reviews if r.source_file == name)
        prof = next(r.professor for r in reviews if r.source_file == name)
        print(f"  {name:30s} {prof:28s} {count} reviews")

    print("\n--- Sample chunk-ready text (first review) ---\n")
    print(reviews[0].to_chunk_text())
    print("\n--- Its metadata ---\n")
    print(reviews[0].metadata())
