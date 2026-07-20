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

## Notes

- The embedding model (`all-MiniLM-L6-v2`) downloads automatically on first run and is cached locally.
- This is a learning project — chunking is naive (blank-line split), and there's no vector database; everything is held in memory for simplicity.
