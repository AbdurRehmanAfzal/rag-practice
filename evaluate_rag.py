"""
RAGAS-style RAG evaluation harness.

Instead of the beginner "does the answer contain these keywords?" check, this
scores retrieval + generation quality with an LLM-as-judge on three metrics,
and runs the SAME question set through two pipelines so you get a real,
defensible before/after number to put on your portfolio:

    NAIVE     = blank-line split + single-shot dense cosine search   (rag.py baseline)
    UPGRADED  = recursive chunking + hybrid retrieval + reranking     (retrieval.py)

Metrics (all 0.0 - 1.0, higher is better):
    context_precision  -> fraction of retrieved chunks that are actually relevant
    faithfulness       -> is the generated answer grounded in the retrieved context
    answer_relevance   -> does the answer actually address the question

Run:
    python evaluate_rag.py

Cost: a handful of gpt-4o-mini calls per question per pipeline. With the default
5-question set that's ~60 cheap calls. Needs OPENAI_API_KEY in your .env.
"""

import os
from statistics import mean

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from retrieval import (
    load_text,
    recursive_chunk,
    Retriever,
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
JUDGE_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Golden question set (add more as your knowledge base grows)
# ---------------------------------------------------------------------------

TEST_QUESTIONS = [
    "What is Abdur Rehman's tech stack?",
    "Where did he study and what is his degree?",
    "What is his experience with RAG and vector databases?",
    "Tell me about the b1properties project.",
    "Is he open to remote work?",
]


# ---------------------------------------------------------------------------
# LLM-as-judge structured verdicts (reuses the project's parse() pattern)
# ---------------------------------------------------------------------------

class RelevanceVerdict(BaseModel):
    relevant: bool = Field(description="True if the text chunk helps answer the question")


class ScoreVerdict(BaseModel):
    score: float = Field(description="A quality score from 0.0 (worst) to 1.0 (best)")


def _parse(system, user, schema):
    resp = client.beta.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=schema,
    )
    return resp.choices[0].message.parsed


def judge_chunk_relevance(question, chunk):
    v = _parse(
        "You judge whether a retrieved text chunk is relevant to answering a question. "
        "Answer strictly about relevance, not correctness.",
        f"Question: {question}\n\nChunk:\n{chunk}\n\nIs this chunk relevant to the question?",
        RelevanceVerdict,
    )
    return 1.0 if v.relevant else 0.0


def judge_faithfulness(context, answer):
    v = _parse(
        "You judge FAITHFULNESS: how well an answer is grounded ONLY in the provided "
        "context. 1.0 = every claim is supported by the context; 0.0 = the answer is "
        "fabricated or contradicts the context.",
        f"Context:\n{context}\n\nAnswer:\n{answer}\n\nScore the answer's faithfulness to the context.",
        ScoreVerdict,
    )
    return max(0.0, min(1.0, v.score))


def judge_answer_relevance(question, answer):
    v = _parse(
        "You judge ANSWER RELEVANCE: how directly an answer addresses the question. "
        "1.0 = fully on point; 0.0 = off-topic or evasive.",
        f"Question: {question}\n\nAnswer:\n{answer}\n\nScore how well the answer addresses the question.",
        ScoreVerdict,
    )
    return max(0.0, min(1.0, v.score))


# ---------------------------------------------------------------------------
# Generation (same grounded prompt for both pipelines, so retrieval is the variable)
# ---------------------------------------------------------------------------

GEN_SYSTEM = (
    "You are a portfolio assistant. Answer the question using ONLY the provided context. "
    "If the context does not contain the answer, say you don't have that information. "
    "Do not invent facts."
)


def generate_answer(question, chunks):
    context = "\n\n".join(chunks)
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content, context


# ---------------------------------------------------------------------------
# Evaluate one pipeline over the whole question set
# ---------------------------------------------------------------------------

def evaluate_pipeline(name, retrieve_fn, k=3):
    print(f"\n>>> Evaluating: {name}")
    precisions, faithfulnesses, relevances = [], [], []

    for q in TEST_QUESTIONS:
        chunks = retrieve_fn(q, k)
        answer, context = generate_answer(q, chunks)

        # context precision: how many retrieved chunks were relevant
        chunk_scores = [judge_chunk_relevance(q, c) for c in chunks]
        precision = mean(chunk_scores) if chunk_scores else 0.0

        faithfulness = judge_faithfulness(context, answer)
        relevance = judge_answer_relevance(q, answer)

        precisions.append(precision)
        faithfulnesses.append(faithfulness)
        relevances.append(relevance)

        print(f"  - {q[:45]:45s}  P={precision:.2f}  F={faithfulness:.2f}  R={relevance:.2f}")

    return {
        "context_precision": mean(precisions),
        "faithfulness": mean(faithfulnesses),
        "answer_relevance": mean(relevances),
    }


def _overall(metrics):
    return mean(metrics.values())


def print_comparison(naive, upgraded):
    rows = ["context_precision", "faithfulness", "answer_relevance"]
    print("\n" + "=" * 60)
    print(f"{'metric':22s} {'naive':>10s} {'upgraded':>10s} {'delta':>10s}")
    print("-" * 60)
    for r in rows:
        d = upgraded[r] - naive[r]
        print(f"{r:22s} {naive[r]:>10.2f} {upgraded[r]:>10.2f} {d:>+10.2f}")
    print("-" * 60)
    no, uo = _overall(naive), _overall(upgraded)
    print(f"{'OVERALL':22s} {no:>10.2f} {uo:>10.2f} {uo - no:>+10.2f}")
    print("=" * 60)
    print(f"\nProof point: overall RAG quality {no:.2f} -> {uo:.2f} "
          f"(+{(uo - no):.2f}) after adding recursive chunking, hybrid retrieval, and reranking.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    text = load_text()

    # NAIVE pipeline: blank-line chunks, dense-only search.
    naive_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    naive_retriever = Retriever(naive_chunks)  # shares one embed model load

    # UPGRADED pipeline: recursive chunks, hybrid + rerank (reuse same embed model).
    smart_chunks = recursive_chunk(text)
    upgraded_retriever = Retriever(smart_chunks, embed_model=naive_retriever.embed_model)

    naive_metrics = evaluate_pipeline(
        "NAIVE (blank-line split + dense cosine)",
        lambda q, k: naive_retriever.naive_search(q, k),
    )
    upgraded_metrics = evaluate_pipeline(
        "UPGRADED (recursive + hybrid + rerank)",
        lambda q, k: upgraded_retriever.search(q, k),
    )

    print_comparison(naive_metrics, upgraded_metrics)


if __name__ == "__main__":
    main()
