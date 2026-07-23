# RAG Practice — Simple RAG Chatbot

A small practice project for learning how **Retrieval-Augmented Generation (RAG)** works end to end: turning text into embeddings, retrieving the most relevant chunks for a question, and using an LLM (OpenAI) to generate a grounded answer — with multi-turn conversation memory and a simple chat UI.

## How it works

1. **Knowledge base** (`knowledge_base.txt`) is loaded and split into chunks (on blank lines).
2. Each chunk is converted into a vector **embedding** using the local `sentence-transformers` model `all-MiniLM-L6-v2`.
3. When a user asks a question, the question is embedded too, and **cosine similarity** is used to find the top-3 most relevant chunks (semantic search — no database needed, everything is in memory).
4. Those chunks are injected as context into a prompt sent to OpenAI (`gpt-4o-mini`), along with the conversation history for that session, so the model answers using only the retrieved information.
5. The answer, the source chunks used, and a `session_id` are returned to the frontend.

## Project structure

```
rag-practice/
├── rag.py               # Main FastAPI app — the RAG chatbot API (run this)
├── final-rag-bot.py      # Fuller bot: ChromaDB, tool calling, lead capture, guardrails, /evaluate
├── retrieval.py          # Upgraded retrieval engine — recursive chunking + hybrid search + reranking
├── evaluate_rag.py       # RAGAS-style eval harness — scores naive vs upgraded retrieval
├── rag_pipeline.py       # Standalone CLI script — same RAG logic, single question via terminal input
├── test_embeddings.py    # Small script to sanity-check embeddings & cosine similarity
├── knowledge_base.txt    # The source text the chatbot retrieves answers from
├── notes.txt              # Simpler/older sample text used by rag_pipeline.py
├── static/
│   └── index.html         # Frontend chat UI (fetches /ask)
├── .env                   # OPENAI_API_KEY (not committed — see .gitignore)
└── .gitignore
```

### `rag.py` — the FastAPI app

This is the actual chatbot server. Key pieces:

- **Startup**: loads the embedding model once, reads and chunks `knowledge_base.txt`, and pre-computes embeddings for every chunk — so retrieval at request time is just a similarity lookup, not re-embedding the whole knowledge base.
- **`conversation_store`**: an in-memory dict (`{session_id: [messages...]}`) that keeps chat history per session, so follow-up questions have context. This resets whenever the server restarts (it's not persisted to a database).
- **`POST /ask`**: the main endpoint.
  - Accepts `{ "question": "...", "session_id": "..." }` (`session_id` is optional — a new one is generated if missing or unknown).
  - Retrieves the top-3 relevant chunks via cosine similarity.
  - Builds the OpenAI message list: system prompt + prior history + new question with retrieved context.
  - Calls OpenAI and returns `{ question, answer, sources, session_id }`.
- **Static serving**: mounts `static/` and serves `static/index.html` at `/`.

### `rag_pipeline.py`

A standalone, non-server script — same embed → retrieve → generate flow, but runs once from the terminal against `notes.txt` and takes one question via `input()`. Useful for understanding the pipeline step by step without FastAPI in the way.

### `test_embeddings.py`

A minimal script to check that embeddings behave as expected — encodes a few sentences and prints cosine similarity between them (e.g. confirming "software engineer" and "programmer" are more similar than "software engineer" and "a cat sleeping").

### `static/index.html`

A simple chat interface (vanilla HTML/CSS/JS, no framework). It:
- Sends messages to `POST /ask` with the current `session_id`.
- Stores the `session_id` returned by the first response and reuses it for follow-up messages so conversation memory works.
- Resets `session_id` on page reload (not persisted in localStorage).

## Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn sentence-transformers scikit-learn numpy python-dotenv openai
   ```

3. **Add your OpenAI API key** in a `.env` file at the project root:
   ```
   OPENAI_API_KEY=your_key_here
   ```

4. **Run the server**:
   ```bash
   uvicorn rag:app --reload
   ```
   Then open `http://127.0.0.1:8000` in your browser.

## Upgraded retrieval pipeline (`retrieval.py` + `evaluate_rag.py`)

The base `rag.py` uses the beginner retrieval path: a naive blank-line chunk split
and a single-shot dense cosine search. `retrieval.py` upgrades that into a
production-shaped pipeline, and `evaluate_rag.py` proves the improvement with numbers.

### `retrieval.py` — production-shaped retrieval

Three upgrades over the baseline, all dependency-light (`sentence-transformers` +
`scikit-learn` + `numpy`, nothing new to install):

1. **Recursive chunking with overlap** — packs sentences up to a target size and
   carries an overlap window between chunks, instead of splitting on blank lines
   (which produced uneven, context-losing chunks — one was 1132 chars vs a clean
   ~560 cap after the upgrade).
2. **Hybrid retrieval** — blends dense (embedding) similarity with sparse (TF-IDF)
   similarity, so exact keywords (names, acronyms) *and* semantic meaning both count.
3. **Cross-encoder reranking** — a `cross-encoder/ms-marco-MiniLM-L-6-v2` model
   rereads each `(query, chunk)` pair jointly and reorders the top candidates for
   precision. This runs only on the ~10 candidates the hybrid stage already narrowed.

The `Retriever` class exposes both `naive_search()` (baseline) and `search()`
(upgraded) so the two can be measured head-to-head.

### `evaluate_rag.py` — RAGAS-style evaluation

Instead of a keyword-match check, this scores retrieval + generation with an
**LLM-as-judge** on three metrics (0.0–1.0): **context precision**, **faithfulness**,
and **answer relevance**. It runs the same question set through the naive and the
upgraded pipeline and prints a before/after table.

```bash
python evaluate_rag.py
```

Measured result on the current knowledge base:

| metric            | naive | upgraded | delta  |
|-------------------|-------|----------|--------|
| context_precision | 0.53  | 0.80     | +0.27  |
| faithfulness      | 1.00  | 1.00     |  0.00  |
| answer_relevance  | 0.60  | 0.90     | +0.30  |
| **OVERALL**       | **0.71** | **0.90** | **+0.19** |

Overall RAG quality went **0.71 → 0.90** after adding recursive chunking, hybrid
retrieval, and reranking — with faithfulness already at 1.00, confirming the
bottleneck was *retrieval*, not generation.

> Note: `evaluate_rag.py` makes ~60 `gpt-4o-mini` calls (cheap) and needs
> `OPENAI_API_KEY` in `.env`. Expand `TEST_QUESTIONS` for a more stable score.

## Notes

- The embedding model (`all-MiniLM-L6-v2`) and the reranker (`ms-marco-MiniLM-L-6-v2`)
  download automatically on first run and are cached locally.
- `rag.py` is the simple in-memory learning version; `retrieval.py` is the upgraded,
  measured pipeline built on top of the same knowledge base.
