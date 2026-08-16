# Universal URL Research Tool

Universal URL Research Tool is a chat-style research workspace where you add URLs, index them, and ask grounded questions against only those sources.

## Live App

- **Frontend (live UI)**: [https://universal-url-researc-ui.onrender.com](https://universal-url-researc-ui.onrender.com)
- **Backend API**: [https://universal-url-researc-api.onrender.com](https://universal-url-researc-api.onrender.com)

## Current Architecture

- **Frontend**: React + Vite in [`frontend/`](/Users/prerna/Downloads/universal-url-researc-main/repo_git/frontend)
- **Backend**: FastAPI in [`backend/main.py`](/Users/prerna/Downloads/universal-url-researc-main/repo_git/backend/main.py)
- **Database**: Supabase PostgreSQL + pgvector
- **Embeddings + answers**: Cloudflare Workers AI via [`worker.js`](/Users/prerna/Downloads/universal-url-researc-main/repo_git/worker.js)
- **Deployment**: Render via [`Dockerfile`](/Users/prerna/Downloads/universal-url-researc-main/repo_git/Dockerfile) and [`render.yaml`](/Users/prerna/Downloads/universal-url-researc-main/repo_git/render.yaml)

## What This Project Is For

Universal URL Research Tool is built for researchers, analysts, and students who want to quickly study information from the open web without manually copy‑pasting content.

At a high level, it lets you:

- Add multiple URLs (articles, blog posts, reports, documentation, etc.).
- Automatically fetch, clean, and index the content from those pages.
- Ask natural‑language questions based **only** on the indexed pages.
- Get concise, grounded answers with links back to the original sources.

Typical use cases include:

- Literature and background research across many web articles.
- Competitive/market research over product pages and blogs.
- Trading and quantitative research using financial blogs, exchange docs, and research posts.
- Policy, legal, or technical deep‑dives using documentation URLs.

You bring the URLs; the tool does the crawling, chunking, embedding, and retrieval so you can focus on asking questions and interpreting results.

## Updated Features

The current version already supports:

- **Chat-style research flow** with a polished React interface.
- **Add, remove, and re-save sources** from a dedicated source manager modal.
- **Session-scoped active sources** so each browser session can work on its own URL set.
- **Grounded answers only from indexed sources** rather than broad open-ended model replies.
- **Visible source attribution** under answers so you can see where an answer came from.
- **Shared research log in the database** for questions, answers, active URLs, and source URLs.
- **Automatic answer cleanup** to reduce leaked prompt text, repeated fragments, and overly long raw output.
- **Shorter UI-friendly answer shaping** so replies stay readable in the chat interface.
- **Vector reuse for already indexed URLs** so the app can avoid unnecessary re-embedding when possible.
- **Playwright-assisted page extraction fallback** for modern pages that rely on JavaScript rendering.
- **Indexing notes per URL** so you can see success, warning, and failure states after saving sources.
- **Split deployment on Render** with separate frontend and backend services.

## Upcoming Features

Planned and high-value next steps include:

- **Role-based access control** for personal, admin, and team-level permissions.
- **Downloadable chat transcripts** so research trails can be saved, shared, or archived.
- **Research history dashboard** to revisit earlier questions and answers in a cleaner timeline.
- **Named research workspaces** so users can maintain multiple source collections instead of one temporary session.
- **User authentication** so each person gets their own saved source sets and chat history.
- **Shared team research rooms** for collaborative analysis across the same source base.
- **Pinned findings and notes** to turn chat outputs into a reusable research brief.
- **Source snapshots / version tracking** so answers can be tied to the exact content seen at indexing time.
- **Scheduled re-indexing** for sources that change often.
- **Stronger source-quality controls** such as duplicate detection, crawl diagnostics, and content completeness warnings.
- **Answer export options** such as PDF, Markdown, or structured research notes.
- **Follow-up aware citations** that show exactly which source supported each major claim.

## Technology Stack

| Layer        | Tools & Libraries |
|--------------|-------------------|
| Frontend     | React, Vite |
| Backend      | FastAPI |
| Orchestration | LangChain |
| Database     | PostgreSQL + pgvector (Supabase Session Pooler) |
| Embeddings   | Cloudflare Workers AI embeddings |
| LLM          | Cloudflare Workers AI (LLaMA 3) |
| Data Fetch   | Requests, BeautifulSoup |
| Chunking     | LangChain Text Splitters |
| JS Rendering | Playwright |
| Hosting      | Render |

This project is a Retrieval-Augmented Generation (RAG) app that lets you:

- Paste one or more URLs.
- Fetch and chunk the page content into semantically meaningful segments.
- Store those segments as embeddings in PostgreSQL with PGVector.
- Ask questions and get grounded answers plus source links, powered by a Cloudflare Worker LLM.

The current stack is split cleanly into:

- **Frontend**: React chat UI in `frontend/`
- **API**: FastAPI endpoints in `backend/main.py`
- **Core research logic**: ingestion, vector store, prompting, and RAG modules in `backend/`

## Render Deployment

This repo is now prepared for a split Render deployment:

- **Frontend**: a static React site
- **Backend**: a separate FastAPI web service

Important files:

- [`Dockerfile`](/Users/prerna/Downloads/universal-url-researc-main/repo_git/Dockerfile): backend-only container for FastAPI
- [`render.yaml`](/Users/prerna/Downloads/universal-url-researc-main/repo_git/render.yaml): declares both Render services

Render services in the blueprint:

- `universal-url-researc-ui`
  - React static site
  - Build command: `cd frontend && npm install && npm run build`
  - Live URL: `https://universal-url-researc-ui.onrender.com`
- `universal-url-researc-api`
  - FastAPI backend
  - Connects to Supabase and the Cloudflare Worker endpoint
  - Live URL: `https://universal-url-researc-api.onrender.com`

Required backend environment variables on Render:

- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`
- `WORKER_ENDPOINT`

Cross-service wiring:

- Frontend uses `VITE_API_BASE_URL=https://universal-url-researc-api.onrender.com`
- Backend allows `FRONTEND_ORIGIN=https://universal-url-researc-ui.onrender.com`

Why this split helps:

- The React app becomes a static site, so it loads faster and does not compete with indexing work.
- The backend keeps its memory for indexing, retrieval, and database work only.
- UI requests stay snappier even when indexing is busy.

---

## Technologies and What They Are Used For

- **React + Vite**: the live chat-style frontend.
- **FastAPI**: the backend API for source indexing, question answering, and persistence.
- **LangChain**:
  - Retrieval flow for question answering.
  - Text splitting for chunking long source pages.
  - LLM wrappers for the Worker-backed model.
- **PostgreSQL + pgvector**:
  - Stores cleaned indexed URL content.
  - Stores vector embeddings and retrieval metadata.
  - Stores the shared chat log table for research history.
- **Cloudflare Workers AI embeddings**:
  - Converts source chunks into vectors for semantic retrieval.
- **Cloudflare Worker AI**:
  - Generates grounded answers from retrieved source context.
- **Requests + BeautifulSoup + Playwright**:
  - Fetch, clean, and extract content from both simple pages and modern JS-rendered pages.

---

## Components and Their Roles

### 1. `frontend/` – React Chat Interface

**What it uses**

- React state for the chat, source modal, and session-level source list.
- `frontend/src/api.*` helpers to talk to the FastAPI backend.
- Session storage so a browser tab can preserve its current active chat state.

**What it does**

- Shows the current active sources as chips.
- Opens the **Add Sources** modal to add, remove, and save URLs.
- Sends questions to the backend with recent chat history.
- Renders replies in a cleaner research-chat format.
- Shows source chips under grounded answers.

---

### 2. `backend/main.py` – FastAPI API and Research Orchestration

**What it uses**

- FastAPI for HTTP routes.
- psycopg2 for PostgreSQL access.
- Ingestion, vector store, and RAG helpers from the repo.

**What it does**

- Exposes `/api/sources`, `/api/sources/index`, `/api/chat`, and `/api/health`.
- Saves indexed page text into `indexed_urls`.
- Logs every chat exchange into `chat_messages`.
- Sanitizes model output before returning it to the frontend.
- Keeps answer formatting shorter and more chat-friendly.

---

### 3. `backend/ingestion.py` – Fetching, Cleaning, and Chunking

**What it uses**

- `requests` with browser-like headers.
- `BeautifulSoup` for HTML cleanup.
- `RecursiveCharacterTextSplitter` for chunking.
- Playwright fallback for modern pages that need client-side rendering.

**What it does**

- Downloads and cleans URL content.
- Detects when a page needs browser rendering.
- Splits long text into retrievable chunks.
- Sends those chunks into the vector store with source metadata.

---

### 4. `backend/vector_store.py` – PGVector Store

**What it uses**

- LangChain’s PGVector integration.
- A custom embedding client backed by the Worker endpoint.

**What it does**

- Stores chunk embeddings in PostgreSQL.
- Preserves URL metadata per chunk for later citation.
- Supports similarity search scoped to the active source URLs.

---

### 5. `backend/rag_chain.py` – Grounded Answer Generation

**What it uses**

- LangChain prompt templates.
- A custom Worker AI LLM wrapper.
- Recent chat history plus retrieved source chunks.

**What it does**

- Builds a grounded prompt that tells the model to answer only from retrieved context.
- Uses chat history only for follow-up continuity, not as factual evidence.
- Encourages shorter, cleaner answers for the chat interface.

---

### 6. PostgreSQL + pgvector

**What it uses**

- Supabase PostgreSQL.
- pgvector collections created through the LangChain integration.

**What it does**

- Stores:
  - `indexed_urls` for cleaned source content
  - vector rows for semantic retrieval
  - `chat_messages` for shared research logging across devices

---

### 7. `worker.js` – Cloudflare Workers AI Endpoint

**What it uses**

- Cloudflare Workers runtime
- Workers AI for embeddings and answer generation

**What it does**

- Accepts prompts from the backend.
- Generates embeddings for chunks during indexing.
- Generates grounded answer text during chat.

---

## End-to-End Flow

1. **A user opens the live React app.**
2. **They add one or more URLs in the source modal and click Save.**
3. **The backend fetches, cleans, chunks, and embeds each valid source.**
4. **Embeddings are stored in PostgreSQL + pgvector with URL metadata.**
5. **The user asks a question in the chat box.**
6. **The backend retrieves the most relevant chunks from only the currently active URLs.**
7. **The RAG prompt is sent to Worker AI for the final grounded answer.**
8. **The backend cleans the reply, logs the exchange, and returns the answer plus source links.**
9. **The frontend renders the answer in the chat view with source chips underneath.**

---
