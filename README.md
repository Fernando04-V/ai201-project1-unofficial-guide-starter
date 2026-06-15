# The Unofficial Guide — Project 1

A retrieval-augmented generation (RAG) system that answers questions about
Computer Science professors at the University of Texas at El Paso (UTEP) using
only real student reviews from Rate My Professors.

---

## Running the System

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Add your Groq API key

Copy `.env.example` to `.env` and paste your free key from
<https://console.groq.com>:

```
GROQ_API_KEY=your_key_here
```

### 3. Build the vector store (one time)

This embeds every review chunk into a local ChromaDB index. You only need to
re-run it if you change `ingest.py` or `chunk.py`.

```powershell
python embed.py --rebuild
```

### 4. Launch the web UI

```powershell
python app.py
```

Then open **<http://localhost:7860>** in your browser, type a question, and
press **Ask** (or hit Enter). Press `Ctrl+C` in the terminal to stop the server.

> First question takes ~10–20s while the `all-MiniLM-L6-v2` model loads; every
> question after that is fast.

### Other entry points (for testing individual stages)

| Command | What it does |
|---|---|
| `python ingest.py` | Preview parsed reviews |
| `python chunk.py` | Preview chunks |
| `python embed.py` | Rebuild index + run a sample query |
| `python generate.py "your question"` | Grounded answer from the command line |
| `python app.py` | Full Gradio web UI |

### Pipeline / file map

```
ingest.py   ->  chunk.py  ->  embed.py            ->  generate.py  ->  app.py
parse .txt      chunk +       embed + ChromaDB         Groq LLM         Gradio UI
into reviews    attach        store + retrieve         grounded answer
                metadata      (top-k=4)                + citations
```

---

## Domain

The domain for this project is student reviews of Computer Science professors at
the University of Texas at El Paso (UTEP). There is no official university
channel where students can learn what a professor is actually like before
registering — teaching style, exam format, and workload are passed around by
word of mouth. This system uses RAG to help CS students make better registration
decisions for upcoming semesters.

---

## Document Sources

All sources are Rate My Professors review pages, saved as structured `.txt`
files in `documents/`. Five professors were collected, yielding **33 individual
student reviews** total.

| # | Source | Type | File | Reviews |
|---|--------|------|------|---------|
| 1 | Rate My Professors | Reviews | `documents/olac_fuentes.txt` | 5 |
| 2 | Rate My Professors | Reviews | `documents/rmp_Daniel_Mejia.txt` | 15 |
| 3 | Rate My Professors | Reviews | `documents/rmp_Marcelo_Frias.txt` | 3 |
| 4 | Rate My Professors | Reviews | `documents/rmp_Natalia_Rosales.txt` | 5 |
| 5 | Rate My Professors | Reviews | `documents/rmp_Vladik_Kreinovich.txt` | 5 |

> Note: the corpus is skewed toward Dr. Mejia (15 of 33 reviews), which can bias
> retrieval toward him on generic queries.

---

## Chunking Strategy

**Chunk size:** 500 characters (target, applied to the review body).

**Overlap:** 100 characters.

**Preprocessing before chunking:**
- Each `.txt` file is split into a file-level header (professor, university,
  ratings) and individual `[REVIEW N]` blocks.
- Each review is parsed into structured fields (course, quality, difficulty,
  grade, tags, body text) by `ingest.py`.
- Each review becomes **one record**, and an attribution header
  (`Professor | University | Course`) plus a `Student rating:` line are
  prepended to every chunk so attribution and rating signal survive even if a
  long review is split.

**Why these choices fit the documents:** RMP reviews are short — almost every
review fits in a single chunk. A 500-character target with 100-character overlap
is a safety net for the occasional long review; it keeps a complete review (with
its ratings and advice) together rather than splitting mid-thought. We use a
native recursive character splitter (paragraph → sentence → word → character)
instead of adding a LangChain dependency.

**Final chunk count:** **33 chunks** (1 per review — the splitter never actually
fired on this corpus because no review exceeded the size limit).

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (384-dimensional
dense vectors). It runs locally with no API key and no rate limits, embeddings
are normalized, and the vectors are stored in a local, persistent ChromaDB
collection using **cosine** distance. Top-k retrieval defaults to **k=4**.

**Production tradeoff reflection:** If cost weren't a constraint, a cloud-hosted
model like OpenAI's `text-embedding-3-large` or Cohere's embeddings would offer
longer context windows, much stronger multilingual handling, and better accuracy
on domain-specific text — at the cost of per-call network latency and API spend.
For storage, ChromaDB is ideal locally, but a university-scale deployment would
need a managed vector database (Pinecone, Weaviate, or pgvector) to serve many
concurrent users. The main accuracy tradeoff we already hit: short embeddings
capture prose well but ignore numeric ratings unless those ratings are written
into the chunk text (see Spec Reflection).

---

## Grounded Generation

**System prompt grounding instruction:** The LLM (Groq
`llama-3.3-70b-versatile`) is instructed to answer **only** from the retrieved
sources and to refuse otherwise:

> "Answer using ONLY the information in the provided sources below. Do not use
> outside knowledge or make assumptions. If the sources do not contain the
> answer, reply exactly: 'I do not have enough student data to answer.' Cite the
> sources you used inline with their tags, e.g. [S1] or [S2][S3]. Never
> attribute one professor's reviews to another — check the professor name in
> each source."

Temperature is set to 0.2 to keep the model close to the sources.

**How source attribution is surfaced in the response:** Attribution works in two
independent layers. (1) The model cites sources inline as `[S1]`, `[S2]`, etc.,
matching the labelled context blocks it was given. (2) Regardless of what the
model writes, the application **programmatically** appends a source list built
from the retrieval metadata (professor, course, source file, similarity score).
So attribution is guaranteed even if the model omits its inline citations.

---

## Evaluation Report

Five in-domain questions plus one out-of-domain question, run end-to-end through
the live system.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What advice do students give for passing Dr. Kreinovich's exams in CS3350? | Go to lectures; his exam reviews predict the tests; cheat sheets allowed; study past exams (he reuses questions). | Said to attend lectures because "exam reviews are gold," cheat sheets are allowed, and to study past exams since he reuses questions. | Relevant (top 2 = Kreinovich CS3350) | Accurate |
| 2 | Which professor has the lowest difficulty / is the easiest? | Across the corpus a Kreinovich review reports Difficulty 1.0/5 — the true minimum. | Claimed Dr. Mejia (Difficulty 2.0/5) is the easiest. | Partially relevant | **Inaccurate** (see Failure Case Analysis) |
| 3 | How is Dr. Olac Fuentes at explaining material? | Not strong at explaining; provides materials and redirects students to the textbook. | Accurately reported he "may not be great at explaining" and tends to direct students back to the material. | Relevant (all 4 = Fuentes) | Accurate |
| 4 | What do students say about Daniel Mejia's lectures and exams? | Engaging/amazing lectures; exams difficult but predictable; extra credit; quizzes harder than exams. | Accurately described engaging lectures, difficult-but-fair exams, extra credit, and that quizzes are harder. | Relevant (all 4 = Mejia) | Accurate |
| 5 | What is Professor Natalia Rosales like for CS4342 Database Management? | Good professor but disorganized/confusing requirements; group projects; test heavy. | Balanced answer: generally good, high quality ratings, but disorganized with confusing requirements; group projects and test heavy. | Relevant (all 4 = Rosales CS4342, similarity 0.67–0.74) | Accurate |
| 6 | What do students say about campus parking and dining halls? | (Out of domain — no data.) | "I do not have enough student data to answer." | Off-target (max similarity ~0.34) | Accurate (correct refusal) |

**Retrieval quality:** Relevant for 4/5 in-domain questions; partially relevant
for the superlative question (#2).
**Response accuracy:** Accurate for 4/5 in-domain questions plus the correct
out-of-domain refusal; inaccurate for #2.

---

## Failure Case Analysis

**Question that failed:** "Which professor has the lowest difficulty / is the
easiest?" (#2)

**What the system returned:** "Dr. Daniel Mejia has the lowest difficulty rating
of 2.0/5." This is wrong — the corpus contains a Kreinovich CS3350 review with a
Difficulty rating of 1.0/5, which is lower.

**Root cause (tied to a specific pipeline stage):** This is a **retrieval-stage**
limitation, not a generation error. Superlative questions ("lowest," "easiest,"
"highest rated") require comparing across *all* reviews, but top-k retrieval only
puts **k=4 chunks** in front of the LLM. The model honestly reported the minimum
*within the four chunks it was given* — it never saw the 1.0/5 review because
that chunk didn't rank in the top 4 for this query. Semantic similarity ranks by
meaning, not by sorting a numeric field, so "easiest" is not a query semantic
search is built to answer.

**What you would change to fix it:** Aggregate/superlative questions should not
rely on semantic top-k alone. Options: (a) detect superlative queries and run a
metadata query that sorts ChromaDB results by the `difficulty` field instead of
by similarity; (b) increase k substantially (or retrieve all chunks) for
aggregate questions so the LLM can see every rating; or (c) precompute
per-professor average ratings and expose them as their own retrievable summary
documents.

---

## Spec Reflection

**One way the spec helped you during implementation:** The "Anticipated
Challenges" section named entity confusion (mixing one professor's reviews with
another's) before any code existed. That drove a concrete design decision:
prepend the `Professor | University | Course` header to every chunk and label
each retrieved source `[S#]` with its professor name in the prompt. Because the
risk was written down first, the mitigation was built into the chunking and
generation stages from the start instead of being patched in later.

**One way your implementation diverged from the spec, and why:** The spec
specified `RecursiveCharacterTextSplitter` from LangChain at 500/100. The
implementation uses an equivalent **native** recursive splitter (to avoid a
heavy dependency), and — more importantly — it **embeds each review's numeric
ratings into the chunk text** (`Student rating: Quality 5.0/5, Difficulty
2.0/5`), which the spec did not call for. We added this after inspecting chunks:
a query like "which professor is easiest?" had no rating signal to match
against, because the ratings lived only in metadata that the embedding model
never sees. Folding them into the text gave retrieval real signal. We also
learned the 500/100 split never actually fires, since every review is short
enough to be a single chunk (33 reviews → 33 chunks).

---

## AI Usage

**Instance 1 — Chunking implementation**

- *What I gave the AI:* The "Chunking Strategy" section (500-char size,
  100-char overlap) plus a sample review file showing the `[REVIEW N]` format,
  and asked for a `chunk_text()` function and a `chunk_reviews()` function that
  consumes the parsed reviews.
- *What it produced:* A native recursive character splitter and a chunker that
  prepends the professor/course attribution header to each chunk.
- *What I changed or overrode:* After printing and inspecting 5 chunks, I found
  short reviews like "Good professor" carried almost no semantic signal and that
  numeric ratings were invisible to the embedder. I directed the AI to fold the
  `Quality/Difficulty/Grade` ratings into the chunk text, and to give the body
  the full 500-char budget (an earlier version subtracted the header and
  needlessly split reviews from 33 into 48 chunks).

**Instance 2 — Embedding, retrieval, generation, and the UI**

- *What I gave the AI:* The architecture diagram and the "Retrieval Approach"
  spec (all-MiniLM-L6-v2, ChromaDB, k=4), then the grounding requirement for
  generation, and finally the Gradio skeleton for the interface.
- *What it produced:* `embed.py` (embedding + persistent ChromaDB + a
  `query_vector_db()` retrieval function), `generate.py` (Groq
  `llama-3.3-70b-versatile` with a strict grounding system prompt and a
  programmatic source list), and `app.py` (Gradio UI).
- *What I changed or overrode:* I set ChromaDB to **cosine** distance (the
  default L2 is wrong for normalized embeddings), required attribution to be
  appended programmatically rather than trusting the LLM's inline citations, and
  resolved a real dependency conflict — Gradio 6.x requires
  `huggingface-hub>=1.2` while the embedding stack requires `<1.0`, so I pinned
  Gradio to 5.x.
