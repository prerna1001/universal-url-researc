import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ingestion import index_url_into_vector_store
from rag_chain import create_rag_chain
from vector_store import get_vector_store


load_dotenv()

COLLECTION_NAME = "url_embeddings"
ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIST_DIR = ROOT_DIR / "frontend" / "dist"
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"


class IndexSourcesRequest(BaseModel):
    urls: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    active_urls: list[str] = Field(default_factory=list, alias="activeUrls")


class IndexReportItem(BaseModel):
    state: str
    url: str
    message: str


class IndexSourcesResponse(BaseModel):
    active_urls: list[str] = Field(alias="activeUrls")
    report: list[IndexReportItem]


class SourcesResponse(BaseModel):
    active_urls: list[str] = Field(alias="activeUrls")


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    grounded: bool
    active_urls: list[str] = Field(alias="activeUrls")


def get_db_config() -> dict[str, Any]:
    """Read database configuration strictly from environment variables."""

    def _get(name: str, default: str | None = None) -> str | None:
        value = os.getenv(name)
        return value if value else default

    db_host = _get("DB_HOST")
    db_name = _get("DB_NAME")
    db_user = _get("DB_USER")
    db_password = _get("DB_PASSWORD")
    db_port_raw = _get("DB_PORT", "5432") or "5432"

    missing = [
        name
        for name, value in [
            ("DB_HOST", db_host),
            ("DB_NAME", db_name),
            ("DB_USER", db_user),
            ("DB_PASSWORD", db_password),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required database environment variables: " + ", ".join(missing)
        )

    db_port = int(db_port_raw) if db_port_raw.isdigit() else 5432

    return {
        "host": db_host,
        "name": db_name,
        "user": db_user,
        "password": db_password,
        "port": db_port,
    }


def get_db_connection():
    """Return a psycopg2 connection using environment configuration only."""
    cfg = get_db_config()
    return psycopg2.connect(
        dbname=cfg["name"],
        user=cfg["user"],
        password=cfg["password"],
        host=cfg["host"],
        port=cfg["port"],
        sslmode="require",
    )


def ensure_indexed_urls_table(conn) -> None:
    """Create the indexed_urls table if it does not exist."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS indexed_urls (
                id SERIAL PRIMARY KEY,
                url TEXT NOT NULL,
                indexed_content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )
        conn.commit()


def normalize_urls(raw_urls: list[str]) -> list[str]:
    """Normalize URL inputs and drop duplicates while preserving order."""
    normalized: list[str] = []
    seen: set[str] = set()

    for raw_url in raw_urls:
        url = (raw_url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        normalized.append(url)

    return normalized


def delete_indexed_urls_except(conn, keep_urls: list[str]) -> None:
    """Remove relational URL rows that are no longer part of the active run."""
    with conn.cursor() as cur:
        if keep_urls:
            cur.execute(
                "DELETE FROM indexed_urls WHERE NOT (url = ANY(%s))",
                (keep_urls,),
            )
        else:
            cur.execute("DELETE FROM indexed_urls")
        conn.commit()


def delete_vector_rows_except(conn, collection_name: str, keep_urls: list[str]) -> None:
    """Remove vector rows whose URL metadata is outside the active URL list."""
    with conn.cursor() as cur:
        if keep_urls:
            cur.execute(
                """
                DELETE FROM langchain_pg_embedding AS e
                USING langchain_pg_collection AS c
                WHERE e.collection_id = c.uuid
                  AND c.name = %s
                  AND COALESCE(e.cmetadata->>'url', '') <> ALL(%s)
                """,
                (collection_name, keep_urls),
            )
        else:
            cur.execute(
                """
                DELETE FROM langchain_pg_embedding AS e
                USING langchain_pg_collection AS c
                WHERE e.collection_id = c.uuid
                  AND c.name = %s
                """,
                (collection_name,),
            )
        conn.commit()


def get_vector_indexed_urls(
    conn, collection_name: str, candidate_urls: list[str] | None = None
) -> list[str]:
    """Return URLs that still have real stored vector rows."""
    with conn.cursor() as cur:
        if candidate_urls is None:
            cur.execute(
                """
                SELECT DISTINCT COALESCE(e.cmetadata->>'url', '')
                FROM langchain_pg_embedding AS e
                JOIN langchain_pg_collection AS c ON e.collection_id = c.uuid
                WHERE c.name = %s
                ORDER BY COALESCE(e.cmetadata->>'url', '')
                """,
                (collection_name,),
            )
            return [row[0] for row in cur.fetchall() if row[0]]

        if not candidate_urls:
            return []

        cur.execute(
            """
            SELECT DISTINCT e.cmetadata->>'url'
            FROM langchain_pg_embedding AS e
            JOIN langchain_pg_collection AS c ON e.collection_id = c.uuid
            WHERE c.name = %s
              AND COALESCE(e.cmetadata->>'url', '') = ANY(%s)
            ORDER BY e.cmetadata->>'url'
            """,
            (collection_name, candidate_urls),
        )
        return [row[0] for row in cur.fetchall() if row[0]]


@lru_cache(maxsize=1)
def build_rag_stack():
    """Instantiate the vector store, retriever, and RAG chain."""
    db_cfg = get_db_config()
    password_encoded = quote_plus(db_cfg["password"])
    connection_string = (
        f"postgresql://{db_cfg['user']}:{password_encoded}"
        f"@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['name']}?sslmode=require"
    )
    vector_store = get_vector_store(connection_string, table_name=COLLECTION_NAME)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    rag_chain = create_rag_chain(retriever)
    return vector_store, retriever, rag_chain


def extract_source_urls(source_docs) -> list[str]:
    """Return unique source URLs from retrieved documents."""
    seen_urls: set[str] = set()
    unique_urls: list[str] = []

    for doc in source_docs:
        meta = getattr(doc, "metadata", {}) or {}
        src_url = meta.get("url")
        if src_url and src_url not in seen_urls:
            seen_urls.add(src_url)
            unique_urls.append(src_url)

    return unique_urls


def list_active_sources() -> list[str]:
    """Return the currently active source URLs from the vector store."""
    conn = get_db_connection()
    try:
        return get_vector_indexed_urls(conn, COLLECTION_NAME, None)
    finally:
        conn.close()


def index_sources(urls: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    """Index the current URL set and keep only this run active."""
    normalized_urls = normalize_urls(urls)
    if not normalized_urls:
        raise HTTPException(status_code=400, detail="Please provide at least one URL.")

    vector_store, _, _ = build_rag_stack()
    report: list[dict[str, str]] = []
    active_urls: list[str] = []

    conn = get_db_connection()
    ensure_indexed_urls_table(conn)

    try:
        delete_indexed_urls_except(conn, normalized_urls)
        delete_vector_rows_except(conn, COLLECTION_NAME, normalized_urls)
        existing_vector_urls = set(
            get_vector_indexed_urls(conn, COLLECTION_NAME, normalized_urls)
        )

        for url in normalized_urls:
            if url in existing_vector_urls:
                active_urls.append(url)
                report.append(
                    {
                        "state": "note",
                        "url": url,
                        "message": "Kept the existing vectors for this URL.",
                    }
                )
                continue

            try:
                page_text, chunk_count = index_url_into_vector_store(url, vector_store)
            except Exception as ingest_err:  # pragma: no cover - network dependent
                report.append(
                    {
                        "state": "error",
                        "url": url,
                        "message": str(ingest_err),
                    }
                )
                continue

            if chunk_count == 0:
                report.append(
                    {
                        "state": "warning",
                        "url": url,
                        "message": (
                            "Fetched the page, but no usable text chunks were produced. "
                            "This page may rely on JavaScript or heavily restrict scraping."
                        ),
                    }
                )
                continue

            with conn.cursor() as cur:
                cur.execute("DELETE FROM indexed_urls WHERE url = %s", (url,))
                cur.execute(
                    """
                    INSERT INTO indexed_urls (url, indexed_content)
                    VALUES (%s, %s)
                    """,
                    (url, page_text),
                )
                conn.commit()

            active_urls.append(url)
            report.append(
                {
                    "state": "success",
                    "url": url,
                    "message": f"Stored {chunk_count} retrievable chunks.",
                }
            )
    finally:
        conn.close()

    return active_urls, report


def answer_question(question: str, expected_active_urls: list[str]) -> dict[str, Any]:
    """Answer a question only when the current run is grounded."""
    normalized_expected = normalize_urls(expected_active_urls)
    db_active_urls = list_active_sources()

    if not db_active_urls:
        return {
            "answer": "I do not have any active indexed sources yet. Add and index URLs first.",
            "sources": [],
            "grounded": False,
            "activeUrls": [],
        }

    if normalized_expected and set(normalized_expected) != set(db_active_urls):
        return {
            "answer": (
                "The source list in the browser does not match the backend's current indexed "
                "set. Refresh sources and try again."
            ),
            "sources": [],
            "grounded": False,
            "activeUrls": db_active_urls,
        }

    _, retriever, rag_chain = build_rag_stack()
    relevant_docs = retriever.invoke(question)
    relevant_docs = [
        doc for doc in relevant_docs if getattr(doc, "page_content", "").strip()
    ]

    if not relevant_docs:
        return {
            "answer": (
                "I could not retrieve any grounded passages for that question. Try a clearer "
                "question, re-index the page, or switch to a source with richer visible text."
            ),
            "sources": [],
            "grounded": False,
            "activeUrls": db_active_urls,
        }

    result = rag_chain({"query": question})
    answer = result.get("result", "No answer returned.").strip()
    source_docs = result.get("source_documents") or relevant_docs
    source_urls = extract_source_urls(source_docs)

    if not source_urls:
        return {
            "answer": (
                "I retrieved context, but the result came back without usable source metadata, "
                "so I am not treating the answer as grounded. Re-indexing the URL is the safest next step."
            ),
            "sources": [],
            "grounded": False,
            "activeUrls": db_active_urls,
        }

    return {
        "answer": answer,
        "sources": source_urls,
        "grounded": True,
        "activeUrls": db_active_urls,
    }


def get_allowed_origins() -> list[str]:
    """Return configured CORS origins."""
    raw = os.getenv("FRONTEND_ORIGIN", "").strip()
    if not raw:
        return ["http://localhost:5173", "http://127.0.0.1:5173"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(title="Universal URL Research Tool API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/sources", response_model=SourcesResponse)
def get_sources():
    active_urls = list_active_sources()
    return {"activeUrls": active_urls}


@app.post("/api/sources/index", response_model=IndexSourcesResponse)
def post_sources_index(payload: IndexSourcesRequest):
    active_urls, report = index_sources(payload.urls)
    return {"activeUrls": active_urls, "report": report}


@app.post("/api/chat", response_model=ChatResponse)
def post_chat(payload: ChatRequest):
    result = answer_question(payload.question.strip(), payload.active_urls)
    return result


if FRONTEND_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS_DIR), name="assets")


@app.get("/{full_path:path}")
def serve_spa(full_path: str):
    if full_path.startswith("api/"):
        return JSONResponse({"detail": "Not found"}, status_code=404)

    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)

    return JSONResponse(
        {
            "message": "Frontend build not found yet. Build the React app or deploy through Render."
        },
        status_code=503,
    )
