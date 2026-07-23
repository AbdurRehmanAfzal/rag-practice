"""
Upgraded retrieval engine for the RAG assistant.

This turns the beginner pipeline (naive blank-line split + single-shot cosine
search, see rag.py) into a production-shaped one:

    1. Recursive chunking with overlap  -> better chunk boundaries, no lost context
    2. Hybrid retrieval                  -> dense vectors + sparse TF-IDF (BM25-style)
    3. Cross-encoder reranking           -> high precision on the final top-k

Dependency-light on purpose: sentence-transformers + scikit-learn + numpy,
all of which the project already uses. Nothing new to install.

Why each step matters (interview-ready talking points):
    - Naive "\n\n" splitting produces uneven chunks that cut sentences in half and
      lose surrounding context. Recursive chunking packs sentences up to a target
      size and carries an overlap window between neighbours.
    - Dense (embedding) search is great for meaning but weak on exact keywords
      (names, IDs, acronyms). Sparse (TF-IDF) search is the opposite. Hybrid gets both.
    - Bi-encoder similarity is fast but approximate. A cross-encoder reads the
      (query, chunk) pair jointly and reranks far more accurately — we only pay
      that cost on the ~10 candidates the hybrid stage already narrowed down to.
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# 1. Recursive chunking with overlap
# ---------------------------------------------------------------------------

def recursive_chunk(text, chunk_size=500, overlap=80):
    """Split text into overlapping chunks that respect natural boundaries.

    We break the text into atomic units (sentences / lines), then greedily pack
    them into windows of up to ~chunk_size characters. Each new chunk starts with
    an `overlap`-character tail of the previous one so context isn't cut mid-thought.

    Args:
        text:       raw source text
        chunk_size: soft maximum characters per chunk
        overlap:    characters of the previous chunk to prepend to the next

    Returns:
        list[str] of chunks
    """
    # Atomic units: split on sentence enders or newlines, drop empties.
    units = re.split(r"(?<=[.!?])\s+|\n+", text)
    units = [u.strip() for u in units if u.strip()]

    chunks = []
    current = ""
    for unit in units:
        # A single unit longer than chunk_size becomes its own chunk.
        if len(unit) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(unit)
            continue

        if len(current) + len(unit) + 1 <= chunk_size:
            current = (current + " " + unit).strip()
        else:
            chunks.append(current)
            # Start the next chunk with an overlap tail from the one we just closed.
            tail = current[-overlap:] if overlap > 0 else ""
            current = (tail + " " + unit).strip()

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# 2 + 3. Hybrid retrieval + cross-encoder reranking
# ---------------------------------------------------------------------------

class Retriever:
    """Hybrid + reranking retriever over a fixed set of chunks.

    Exposes two methods so you can measure the upgrade head-to-head:
        - naive_search : the old single-shot dense cosine search (baseline)
        - search       : hybrid retrieve -> cross-encoder rerank (upgraded)
    """

    def __init__(self, chunks, embed_model=None, alpha=0.5,
                 reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            chunks:      list of text chunks to search over
            embed_model: a loaded SentenceTransformer (loaded here if None)
            alpha:       hybrid weight — 1.0 = pure dense, 0.0 = pure sparse
            reranker_name: cross-encoder model for reranking
        """
        self.chunks = chunks
        self.alpha = alpha
        self.embed_model = embed_model or SentenceTransformer("all-MiniLM-L6-v2")

        # Dense index: one embedding per chunk (computed once).
        self.chunk_embeddings = self.embed_model.encode(chunks, show_progress_bar=False)

        # Sparse index: TF-IDF over the same chunks (the BM25-style keyword signal).
        self.tfidf = TfidfVectorizer().fit(chunks)
        self.chunk_tfidf = self.tfidf.transform(chunks)

        # Reranker: downloads (~90 MB) on first run, then cached locally.
        self.reranker = CrossEncoder(reranker_name)

    # -- baseline -----------------------------------------------------------

    def naive_search(self, query, k=3):
        """Old behavior: single-shot dense cosine similarity (the rag.py baseline)."""
        q = self.embed_model.encode([query])
        sims = cosine_similarity(q, self.chunk_embeddings)[0]
        idx = np.argsort(sims)[::-1][:k]
        return [self.chunks[i] for i in idx]

    # -- upgraded -----------------------------------------------------------

    @staticmethod
    def _minmax(x):
        """Scale scores to [0, 1] so dense and sparse are comparable before blending."""
        x = np.asarray(x, dtype=float)
        span = x.max() - x.min()
        if span < 1e-9:
            return np.zeros_like(x)
        return (x - x.min()) / span

    def _hybrid_candidates(self, query, top_n=10):
        """Blend dense + sparse similarity and return the top_n candidate indices."""
        q_emb = self.embed_model.encode([query])
        dense = cosine_similarity(q_emb, self.chunk_embeddings)[0]

        q_tfidf = self.tfidf.transform([query])
        sparse = cosine_similarity(q_tfidf, self.chunk_tfidf)[0]

        score = self.alpha * self._minmax(dense) + (1 - self.alpha) * self._minmax(sparse)
        idx = np.argsort(score)[::-1][:top_n]
        return [int(i) for i in idx]

    def search(self, query, k=3, top_n=10):
        """Upgraded: hybrid retrieve top_n candidates, then cross-encoder rerank to top k."""
        cand_idx = self._hybrid_candidates(query, top_n=top_n)
        if not cand_idx:
            return []
        pairs = [[query, self.chunks[i]] for i in cand_idx]
        rerank_scores = self.reranker.predict(pairs)
        order = np.argsort(rerank_scores)[::-1][:k]
        return [self.chunks[cand_idx[i]] for i in order]


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def load_text(path="knowledge_base.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_upgraded_retriever(path="knowledge_base.txt", embed_model=None):
    """Recursive chunking + hybrid + rerank over the knowledge base."""
    chunks = recursive_chunk(load_text(path))
    return Retriever(chunks, embed_model=embed_model)


def build_naive_retriever(path="knowledge_base.txt", embed_model=None):
    """Blank-line split + dense-only search — the original baseline, for comparison."""
    raw = load_text(path).split("\n\n")
    chunks = [c.strip() for c in raw if c.strip()]
    return Retriever(chunks, embed_model=embed_model)


if __name__ == "__main__":
    # Quick smoke test: show the chunking difference and a sample query.
    text = load_text()
    naive_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    smart_chunks = recursive_chunk(text)

    print(f"Naive blank-line chunks : {len(naive_chunks)}")
    print(f"Recursive chunks (~500c): {len(smart_chunks)}")
    print()

    print("Loading models (reranker downloads on first run)...")
    retriever = Retriever(smart_chunks)

    q = "What is his experience with RAG and vector databases?"
    print(f"\nQuery: {q}\n")
    print("--- Upgraded (hybrid + rerank) top-3 ---")
    for i, chunk in enumerate(retriever.search(q, k=3), 1):
        print(f"[{i}] {chunk[:160]}...")
