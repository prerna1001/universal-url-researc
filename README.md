# Universal URL Research Tool

Live demo: https://universal-url-researc-7jj88zvtofjkvegmykp447.streamlit.app/

Deployed as a live app on **Streamlit Cloud**, backed by **Supabase PostgreSQL + pgvector** and a **Cloudflare Worker AI** LLM endpoint.

## What This Project Is For

Universal URL Research Tool is built for researchers, analysts, and students who want to quickly study information from the open web without manually copy‑pasting content.

At a high level, it lets you:

- Add multiple URLs (articles, blog posts, reports, documentation, etc.).
- Automatically fetch, clean, and index the content from those pages.
- Ask natural‑language questions based **only** on the indexed pages.
- Get concise, grounded answers with links back to the original sources.


-Typical use cases include:

- Literature and background research across many web articles.
- Competitive/market research over product pages and blogs.
- Trading and quantitative research using financial blogs, exchange docs, and research posts.
- Policy, legal, or technical deep‑dives using documentation URLs.

You bring the URLs; the tool does the crawling, chunking, embedding, and retrieval so you can focus on asking questions and interpreting results.

## Technology Stack

| Layer        | Tools & Libraries |
|--------------|-------------------|
| Frontend     | Streamlit |
| Orchestration | LangChain |
| Backend      | Python (Streamlit app on Streamlit Cloud) |
| Database     | PostgreSQL + pgvector (Supabase Session Pooler) |
| Embeddings   | sentence-transformers |
| LLM          | Cloudflare Workers AI (LLaMA 3) |
| Data Fetch   | Requests, BeautifulSoup |
| Chunking     | LangChain Text Splitters |

This project is a small MVP-style Retrieval-Augmented Generation (RAG) app that lets you:

- Paste one or more URLs.
- Fetch and chunk the page content into semantically meaningful segments.
- Store those segments as embeddings in PostgreSQL with PGVector.
- Ask questions and get grounded answers plus source links, powered by a Cloudflare Worker LLM.

The stack is designed around a simple **Model–View–Presenter (MVP)** separation:

- **View**: Streamlit UI (`app.py`).
- **Presenter / Orchestration**: The high-level flows inside `app.py` that call ingestion, vector store, and RAG chain.
- **Model**: Ingestion, embeddings, vector store, database and LLM/RAG chain (`ingestion.py`, `vector_store.py`, `rag_chain.py`, PostgreSQL + PGVector, Cloudflare Worker).

---

## Technologies and What They Are Used For

- **Python + Streamlit**: Front-end UI and orchestration logic (`app.py`).
- **LangChain**:
  - `RetrievalQA` for the RAG pipeline.
  - Text splitters (`RecursiveCharacterTextSplitter`) for chunking.
  - LLM abstractions (`BaseLLM`, `LLMResult`, `Generation`) for the custom Worker-backed LLM.
- **PostgreSQL + PGVector**:
  - Stores raw cleaned page text (`indexed_urls` table).
  - Stores embeddings and metadata for semantic search (`url_embeddings` via `PGVector`).
- **HuggingFace / sentence-transformers**:
  - `sentence-transformers/all-MiniLM-L6-v2` for turning text chunks into embedding vectors.
- **Cloudflare Worker AI**:
  - Hosts the LLM (`@cf/meta/llama-3-8b-instruct`) behind a simple HTTP endpoint used by the custom LangChain LLM wrapper.
- **Requests + BeautifulSoup**:
  - Fetch page HTML and clean it into plain text suitable for embeddings.

---

## Components and Their Roles

### 1. `app.py` – Streamlit View + Presenter

**What it uses**

- **Streamlit**: to build the UI.
- **psycopg2 / Postgres**: to store full page text in `indexed_urls`.
- **`vector_store.get_vector_store`**: to get a PGVector-backed vector store.
- **`ingestion.index_url_into_vector_store`**: to fetch, clean, chunk, and embed URLs.
- **`rag_chain.create_rag_chain`**: to build the RetrievalQA chain on top of the retriever.
- **Session state (`st.session_state`)**: to cache the RAG chain between interactions.

**What it does**

- Reads DB config from environment (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`) and builds:
  - A psycopg2 connection for relational data.
  - A PG connection string for the vector store.
- On **“Index Sources”**:
  1. Creates a `PGVector` store: `get_vector_store(connection_string, table_name="url_embeddings")`.
  2. Wraps it in a retriever: `vector_store.as_retriever()`.
  3. Builds a RAG chain: `create_rag_chain(retriever)`.
  4. Stores the chain in `st.session_state["rag_chain"]`.
  5. Ensures the `indexed_urls` table exists.
  6. For each URL:
     - Calls `index_url_into_vector_store(url, vector_store)` to fetch + embed all segments.
     - Inserts `(url, full_cleaned_text)` into `indexed_urls`.
- On **question submit**:
  - Retrieves `rag_chain` from session.
  - Calls `rag_chain({"query": question})`.
  - Displays the formatted answer.
  - Extracts `doc.metadata["url"]` from `source_documents`, deduplicates, and shows each unique source URL once.

---

### 2. `ingestion.py` – Fetching, Cleaning, and Chunking

**What it uses**

- `requests` with custom headers (polite `User-Agent`, `Accept`).
- `BeautifulSoup` from `beautifulsoup4` for HTML parsing and cleaning.
- `RecursiveCharacterTextSplitter` from `langchain-text-splitters` for segmenting text.

**What it does**

- `fetch_url_text(url)`:
  - Downloads the web page HTML.
  - Handles HTTP errors cleanly (e.g., clear message for 403/blocked URLs).
  - Strips scripts/styles and normalizes whitespace.
  - Returns plain cleaned text.
- `split_into_chunks(text, chunk_size=1000, chunk_overlap=200)`:
  - Breaks long text into overlapping segments suitable for retrieval.
- `index_url_into_vector_store(url, vector_store)`:
  - Fetches and chunks text.
  - Calls `vector_store.add_texts` with:
    - `texts = [chunk1, chunk2, …]`
    - `metadatas = [{"url": url, "chunk_index": i}, …]`
  - Returns the full cleaned text so `app.py` can also store it in `indexed_urls`.

---

### 3. `vector_store.py` – PGVector Store

**What it uses**

- LangChain’s `PGVector` integration from `langchain_community`.
- `HuggingFaceEmbeddings` wrapping `sentence-transformers/all-MiniLM-L6-v2`.

**What it does**

- `get_vector_store(connection_string, table_name="url_embeddings")`:
  - Instantiates `HuggingFaceEmbeddings` once.
  - Calls `PGVector.from_texts(texts=[], embedding=embeddings, collection_name=table_name, connection_string=connection_string)` to configure the collection.
  - Returns a vector store object that:
    - Accepts `add_texts` during ingestion.
    - Exposes `.as_retriever()` for semantic search at question time.

---

### 4. `rag_chain.py` – RAG Chain and Cloudflare Worker LLM

**What it uses**

- `BaseLLM`, `LLMResult`, `Generation` from `langchain_core`.
- `RetrievalQA` from LangChain (classic).
- `PromptTemplate` for a carefully designed RAG prompt.
- `requests` to call the Cloudflare Worker endpoint.

**Custom LLM: `WorkerAILLM`**

- Holds `endpoint: str` (Cloudflare Worker URL).
- `_call(prompt, stop=None)`:
  - POSTs `{"prompt": prompt}` to the Worker.
  - Expects a JSON array and extracts the nested text response.
- `_generate(prompts, stop=None, **kwargs)`:
  - Calls `_call` for each prompt and wraps results in a proper `LLMResult`.

**RAG chain: `create_rag_chain(retriever, ...)`**

- Builds a prompt that:
  - Treats the model as a **serious, concise research assistant**.
  - Uses **only** the provided context.
  - Avoids jokes and hallucinations.
  - Structures the answer as:
    - Short Answer
    - Key Points
    - Evidence from Sources
    - Limitations
- Creates a `RetrievalQA` chain with:
  - `llm = WorkerAILLM(endpoint=...)`
  - `retriever = vector_store.as_retriever()`
  - `chain_type = "stuff"`
  - `return_source_documents = True`

At question time, this chain:

1. Uses the retriever to pull the most relevant chunks from PGVector.
2. Fills the prompt template with `CONTEXT` and `QUESTION`.
3. Sends the final prompt to the Cloudflare Worker LLM.
4. Returns the answer text plus the source documents for UI display.

---

### 5. PostgreSQL + PGVector

**What it uses**

- A PostgreSQL database (e.g. `universal_url_research`).
- PGVector extension enabled in that database.

**Tables**

- `indexed_urls` (managed by `app.py`):
  - `id SERIAL PRIMARY KEY`
  - `url TEXT NOT NULL`
  - `indexed_content TEXT NOT NULL`
  - `created_at TIMESTAMP DEFAULT NOW()`
- `url_embeddings` (managed by the LangChain PGVector integration):
  - Stores:
    - Chunk text.
    - Embedding vector.
    - JSON metadata such as `{"url": "...", "chunk_index": ...}`.

---

### 6. `worker.js` – Cloudflare Worker LLM Endpoint

**What it uses**

- Cloudflare Workers runtime.
- `env.AI.run("@cf/meta/llama-3-8b-instruct", { prompt })` to access the LLM.

**What it does**

- Exposes an HTTP `fetch` handler that:
  - Reads JSON from the request body and extracts `prompt`.
  - Validates that `prompt` is a non-empty string.
  - Calls `env.AI.run` with that prompt.
  - Wraps the model output into a JSON response that `WorkerAILLM` expects.

This ensures that **the RAG prompt constructed in Python is exactly what the model sees**, instead of any hard-coded joke prompt.

---

## End-to-End Flow (How It Works)

1. **User opens the Streamlit app (`app.py`).**
2. **User enters one or more URLs and clicks “Index Sources”.**
   - `app.py`:
     - Normalizes DB settings from `.env`.
     - Builds the PGVector store and retriever.
     - Builds and stores the RAG chain in session state.
     - Ensures `indexed_urls` table exists.
   - For each URL:
     - `ingestion.index_url_into_vector_store`:
       - Fetches HTML, cleans to text.
       - Splits text into overlapping chunks.
       - Embeds and stores chunks in the `url_embeddings` PGVector collection with URL metadata.
     - `app.py` also inserts full cleaned text into `indexed_urls`.
3. **User asks a question.**
   - `app.py` calls `rag_chain({"query": question})`.
   - `RetrievalQA`:
     - Uses the retriever to select the most relevant chunks from PGVector.
     - Builds a structured RAG prompt with those chunks as `CONTEXT`.
     - Sends that prompt to the Cloudflare Worker LLM via `WorkerAILLM`.
4. **Cloudflare Worker runs the LLM and returns a response.**
   - The LLM generates a structured answer following the prompt’s instructions.
   - `WorkerAILLM` parses the JSON, extracts the answer text, and returns it to the chain.
5. **Streamlit shows the result.**
   - Renders the answer (Short Answer, Key Points, Evidence, Limitations).
   - Extracts URLs from `source_documents`, deduplicates them, and lists each unique source.

---
