"""
Milestone 5 — Query Interface (Gradio web UI).

The full pipeline behind this UI:
    ingest -> chunk -> embed/store -> retrieve -> grounded generation
                                                  (answer_query in generate.py)

The UI is intentionally thin: it calls answer_query() and renders the answer
plus the programmatically-collected source list. Source attribution comes from
the retrieved chunks, not from the LLM, so it is shown even if the model omits
its inline [S#] citations.

Run:
    python app.py
    # then open http://localhost:7860
"""

from __future__ import annotations

import os

from generate import DEFAULT_K, answer_query

EXAMPLE_QUESTIONS = [
    "What advice do students give for passing Dr. Kreinovich's exams in CS3350?",
    "Which professor is the easiest?",
    "How is Dr. Olac Fuentes at explaining material?",
    "What do students say about Daniel Mejia's lectures?",
]


def handle_query(question: str):
    """Run one question through the RAG pipeline and format UI output.

    Returns (answer_text, sources_text). The sources block is built from the
    retrieval metadata, so attribution is guaranteed regardless of the LLM.
    """
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""

    result = answer_query(question, k=DEFAULT_K)

    sources_lines = []
    for s in result["sources"]:
        sources_lines.append(
            f"• [{s['tag']}] {s['professor']} — {s['course']} "
            f"(file: {s['source_file']}, similarity {s['similarity']})"
        )
    sources_text = "\n".join(sources_lines)
    return result["answer"], sources_text


def build_demo():
    import gradio as gr

    with gr.Blocks(title="The Unofficial Guide — UTEP CS Professors") as demo:
        gr.Markdown(
            "# The Unofficial Guide\n"
            "Ask about UTEP Computer Science professors. Answers come **only** "
            "from student reviews on Rate My Professors — if the reviews don't "
            "cover it, the assistant will say so."
        )

        inp = gr.Textbox(
            label="Your question",
            placeholder="e.g. What advice do students give for Dr. Kreinovich's exams?",
        )
        btn = gr.Button("Ask", variant="primary")
        answer = gr.Textbox(label="Answer", lines=8)
        sources = gr.Textbox(label="Retrieved from (sources)", lines=5)

        gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=inp)

        btn.click(handle_query, inputs=inp, outputs=[answer, sources])
        inp.submit(handle_query, inputs=inp, outputs=[answer, sources])

    return demo


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") == "your_key_here":
        # generate.py already loads .env; this is a friendly pre-flight check.
        from dotenv import load_dotenv

        load_dotenv()
        if not os.environ.get("GROQ_API_KEY") or os.environ["GROQ_API_KEY"] == "your_key_here":
            raise SystemExit(
                "GROQ_API_KEY is not set. Copy .env.example to .env and paste "
                "your free key from https://console.groq.com"
            )

    build_demo().launch()
