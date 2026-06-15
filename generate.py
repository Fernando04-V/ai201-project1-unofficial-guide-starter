"""
Stage 5 — Grounded Generation (via Groq).

Pipeline position (see architecture.png):
    ingest -> chunk -> embed/store -> retrieve (embed.py) -> [THIS FILE]

Takes a user question, retrieves the top-k review chunks from the vector store,
and asks an LLM to answer using ONLY those chunks, with inline source citations.

Grounding is enforced two ways (per planning.md):
  1. System prompt: answer only from the provided context; if the answer isn't
     there, say so; cite sources inline.
  2. Structure: each chunk is labelled [S1], [S2]... with its professor + source
     file, so the model has explicit, citable handles and cannot blur professors
     together (the entity-confusion mitigation, carried through to generation).

Usage:
    python generate.py "What advice do students give for Kreinovich's exams?"
    python generate.py            # interactive prompt

Requires GROQ_API_KEY in .env (copy .env.example -> .env, paste your free key).
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from groq import Groq

from embed import query_vector_db

load_dotenv()

MODEL = "llama-3.3-70b-versatile"   # per planning.md AI Tool Plan, Component 4
DEFAULT_K = 4

SYSTEM_PROMPT = """You are The Unofficial Guide, a study-advice assistant that \
answers questions about Computer Science professors at UTEP using ONLY student \
reviews from Rate My Professors.

Rules:
- Answer using ONLY the information in the provided sources below. Do not use \
outside knowledge or make assumptions.
- If the sources do not contain the answer, reply exactly: "I do not have \
enough student data to answer." Do not guess.
- Cite the sources you used inline with their tags, e.g. [S1] or [S2][S3]. \
Every claim must be backed by a citation.
- Be specific: students want actionable advice (how to study, exam style, \
workload, teaching style), not vague summaries.
- Different sources may describe different professors. Never attribute one \
professor's reviews to another — check the professor name in each source."""


def format_context(results: list[dict]) -> str:
    """Render retrieved chunks as labelled, citable sources for the prompt."""
    blocks = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        label = (
            f"[S{i}] Professor: {meta.get('professor', '?')} | "
            f"Course: {meta.get('course', 'N/A')} | "
            f"Source file: {meta.get('source_file', '?')}"
        )
        blocks.append(f"{label}\n{r['text']}")
    return "\n\n".join(blocks)


def answer_query(query: str, k: int = DEFAULT_K, model: str = MODEL) -> dict:
    """Retrieve context, generate a grounded answer, and return both.

    Returns a dict with:
      - answer:  the model's text response (with [S#] citations)
      - sources: list of {tag, professor, course, source_file, similarity}
                 so the caller can render a "Sources" footer for attribution
    """
    results = query_vector_db(query, k=k)

    context = format_context(results)
    user_message = (
        f"Sources:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the sources above, citing them inline as [S#]."
    )

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,   # low: stay close to the sources, minimize invention
    )
    answer = completion.choices[0].message.content

    sources = [
        {
            "tag": f"S{i}",
            "professor": r["metadata"].get("professor", "?"),
            "course": r["metadata"].get("course", "N/A"),
            "source_file": r["metadata"].get("source_file", "?"),
            "similarity": r["similarity"],
        }
        for i, r in enumerate(results, 1)
    ]
    return {"answer": answer, "sources": sources}


def _print_result(result: dict) -> None:
    print("\n" + result["answer"] + "\n")
    print("-" * 60)
    print("Sources:")
    for s in result["sources"]:
        print(
            f"  [{s['tag']}] {s['professor']} - {s['course']} "
            f"({s['source_file']}, similarity {s['similarity']})"
        )


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY") or os.environ["GROQ_API_KEY"] == "your_key_here":
        sys.exit(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste your "
            "free key from https://console.groq.com"
        )

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Ask about a UTEP CS professor: ").strip()

    _print_result(answer_query(query))
