import os
import json
from functools import lru_cache
from typing import Any
from urllib.parse import quote_plus
import re

import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ingestion import index_url_into_vector_store
from rag_chain import generate_grounded_answer, get_rag_prompt_template, get_worker_llm
from vector_store import get_vector_store


load_dotenv()

COLLECTION_NAME = "url_embeddings"


class IndexSourcesRequest(BaseModel):
    urls: list[str] = Field(default_factory=list)


class ChatHistoryTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    active_urls: list[str] = Field(default_factory=list, alias="activeUrls")
    chat_history: list[ChatHistoryTurn] = Field(default_factory=list, alias="chatHistory")


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


def ensure_chat_messages_table(conn) -> None:
    """Create the chat_messages table if it does not exist."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                grounded BOOLEAN NOT NULL DEFAULT FALSE,
                active_urls JSONB NOT NULL DEFAULT '[]'::jsonb,
                source_urls JSONB NOT NULL DEFAULT '[]'::jsonb,
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
def build_vector_store():
    """Instantiate and cache the shared vector store."""
    db_cfg = get_db_config()
    password_encoded = quote_plus(db_cfg["password"])
    connection_string = (
        f"postgresql://{db_cfg['user']}:{password_encoded}"
        f"@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['name']}?sslmode=require"
    )
    return get_vector_store(connection_string, table_name=COLLECTION_NAME)


@lru_cache(maxsize=1)
def build_rag_stack():
    """Instantiate the retriever plus the shared model and prompt."""
    vector_store = build_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = get_worker_llm()
    prompt_template = get_rag_prompt_template()
    return vector_store, retriever, llm, prompt_template


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


PROMPT_ECHO_PATTERNS = [
    "You are a grounded research assistant inside a chat app.",
    "Answer the user's QUESTION using ONLY the CONTEXT.",
    "Rules:",
    "CONTEXT:",
    "QUESTION:",
    "Answer:",
]


def dedupe_repeated_paragraphs(text: str) -> str:
    """Drop repeated paragraphs while preserving the original order."""
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    seen_normalized: set[str] = set()
    unique_paragraphs: list[str] = []

    for paragraph in paragraphs:
        normalized = re.sub(r"\s+", " ", paragraph).strip().lower()
        if normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        unique_paragraphs.append(paragraph)

    return "\n\n".join(unique_paragraphs)


def trim_meta_tail(text: str) -> str:
    """Remove note-like or fallback-like tails appended after a valid answer."""
    patterns = [
        r"\bNote:\s.*$",
        r"\bSources?\s+used\s*$",
        r"I couldn't find that in the indexed sources\..*$",
    ]

    trimmed = text
    for pattern in patterns:
        trimmed = re.sub(pattern, "", trimmed, flags=re.IGNORECASE | re.DOTALL).strip()

    return trimmed


def sanitize_model_answer(answer: str) -> str:
    """Strip leaked prompt instructions and obvious duplicated fallback text."""
    cleaned = (answer or "").strip()
    fallback = "I couldn't find that in the indexed sources."

    if not cleaned:
        return fallback

    # If the model echoed the prompt, keep only the tail after the last explicit answer marker.
    if "Answer:" in cleaned:
        cleaned = cleaned.split("Answer:")[-1].strip()

    lines = [line.rstrip() for line in cleaned.splitlines()]
    filtered_lines: list[str] = []
    skipping_prompt_block = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if filtered_lines and filtered_lines[-1] != "":
                filtered_lines.append("")
            continue

        if any(line.startswith(pattern) for pattern in PROMPT_ECHO_PATTERNS):
            skipping_prompt_block = True
            continue

        if skipping_prompt_block and (
            line.startswith("- ")
            or line.startswith("* ")
            or re.match(r"^\d+\.", line)
        ):
            continue

        skipping_prompt_block = False
        filtered_lines.append(line)

    cleaned = "\n".join(filtered_lines).strip()
    cleaned = trim_meta_tail(cleaned)
    cleaned = dedupe_repeated_paragraphs(cleaned)

    # Remove repeated fallback fragments if the model loops them.
    fallback_matches = re.findall(re.escape(fallback), cleaned)
    if len(fallback_matches) > 1:
        cleaned = re.sub(
            rf"(?:\)?\s*{re.escape(fallback)})+",
            fallback,
            cleaned,
        ).strip()

    partial_fallback_pattern = re.compile(
        r"^"
        + re.escape(fallback)
        + r"(?:\s+I couldn't find that in the(?:\s+indexed(?:\s+sources?)?)?\.?)?$",
        flags=re.IGNORECASE,
    )
    if partial_fallback_pattern.match(cleaned):
        cleaned = fallback

    # Remove trailing note fragments that often appear after an echoed instruction block.
    cleaned = re.sub(r"\(\s*Note:.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip() if "\n" not in cleaned else cleaned.strip()

    return cleaned or fallback


def list_active_sources() -> list[str]:
    """Return an empty list because visible active sources are session-local."""
    return []


def index_sources(urls: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    """Index the requested URLs without changing other browsers' source sets."""
    normalized_urls = normalize_urls(urls)
    report: list[dict[str, str]] = []
    active_urls: list[str] = []

    conn = get_db_connection()
    ensure_indexed_urls_table(conn)
    ensure_chat_messages_table(conn)

    try:
        if not normalized_urls:
            report.append(
                {
                    "state": "success",
                    "url": "",
                    "message": "Cleared the active sources for this chat session.",
                }
            )
            return active_urls, report

        vector_store = build_vector_store()
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
                page_text, chunk_count, extraction_mode = index_url_into_vector_store(
                    url, vector_store
                )
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
                message = (
                    "Fetched the page, but no usable text chunks were produced. "
                    "This page may rely on JavaScript or heavily restrict scraping."
                )
                if extraction_mode == "browser":
                    message = (
                        "Fetched the page with browser rendering, but still could not produce usable text chunks. "
                        "This page may be access-controlled, too sparse, or intentionally difficult to scrape."
                    )

                report.append(
                    {
                        "state": "warning",
                        "url": url,
                        "message": message,
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
                    (url, page_text[:2000]),
                )
                conn.commit()

            success_message = f"Stored {chunk_count} retrievable chunks."
            if extraction_mode == "browser":
                success_message = (
                    f"Stored {chunk_count} retrievable chunks using browser-rendered extraction."
                )

            active_urls.append(url)
            report.append(
                {
                    "state": "success",
                    "url": url,
                    "message": success_message,
                }
            )
    finally:
        conn.close()

    return active_urls, report


def retrieve_docs_for_active_urls(question: str, active_urls: list[str]):
    """Retrieve relevant docs only from the URLs active in this browser session."""
    vector_store, _, _, _ = build_rag_stack()
    docs = []
    seen_keys: set[tuple[str, int | None, str]] = set()

    for url in active_urls:
        matches = vector_store.similarity_search(question, k=4, filter={"url": url})
        for doc in matches:
            metadata = getattr(doc, "metadata", {}) or {}
            key = (
                metadata.get("url", ""),
                metadata.get("chunk_index"),
                getattr(doc, "page_content", ""),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            docs.append(doc)

    return docs


def log_chat_message(
    question: str,
    answer: str,
    grounded: bool,
    active_urls: list[str],
    source_urls: list[str],
) -> None:
    """Persist each chat exchange in the shared database log."""
    conn = get_db_connection()
    ensure_chat_messages_table(conn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages (question, answer, grounded, active_urls, source_urls)
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
                """,
                (
                    question,
                    answer,
                    grounded,
                    json.dumps(active_urls),
                    json.dumps(source_urls),
                ),
            )
            conn.commit()
    finally:
        conn.close()


def answer_question(
    question: str,
    expected_active_urls: list[str],
    chat_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Answer a question using only the URLs active in this browser session."""
    normalized_expected = normalize_urls(expected_active_urls)

    if not normalized_expected:
        result = {
            "answer": "I do not have any active indexed sources yet. Add and index URLs first.",
            "sources": [],
            "grounded": False,
            "activeUrls": [],
        }
        log_chat_message(question, result["answer"], False, [], [])
        return result

    _, _, llm, prompt_template = build_rag_stack()
    relevant_docs = retrieve_docs_for_active_urls(question, normalized_expected)
    relevant_docs = [
        doc for doc in relevant_docs if getattr(doc, "page_content", "").strip()
    ]

    if not relevant_docs:
        result = {
            "answer": (
                "I could not retrieve any grounded passages for that question. Try a clearer "
                "question, re-index the page, or switch to a source with richer visible text."
            ),
            "sources": [],
            "grounded": False,
            "activeUrls": normalized_expected,
        }
        log_chat_message(question, result["answer"], False, normalized_expected, [])
        return result

    answer = sanitize_model_answer(
        generate_grounded_answer(
            llm=llm,
            prompt_template=prompt_template,
            question=question,
            source_docs=relevant_docs,
            chat_history=chat_history,
        )
    )
    source_docs = relevant_docs
    source_urls = extract_source_urls(source_docs)

    if not source_urls:
        result = {
            "answer": (
                "I retrieved context, but the result came back without usable source metadata, "
                "so I am not treating the answer as grounded. Re-indexing the URL is the safest next step."
            ),
            "sources": [],
            "grounded": False,
            "activeUrls": normalized_expected,
        }
        log_chat_message(question, result["answer"], False, normalized_expected, [])
        return result

    result = {
        "answer": answer,
        "sources": source_urls,
        "grounded": True,
        "activeUrls": normalized_expected,
    }
    log_chat_message(question, answer, True, normalized_expected, source_urls)
    return result


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
    result = answer_question(
        payload.question.strip(),
        payload.active_urls,
        [
            {"role": turn.role, "content": turn.content}
            for turn in payload.chat_history
            if turn.content.strip()
        ],
    )
    return result


@app.get("/{full_path:path}")
def serve_spa(full_path: str):
    if full_path.startswith("api/"):
        return JSONResponse({"detail": "Not found"}, status_code=404)

    return JSONResponse(
        {
            "message": "Universal URL Research Tool API is running.",
            "health": "/api/health",
            "sources": "/api/sources",
            "chat": "/api/chat",
        }
    )
