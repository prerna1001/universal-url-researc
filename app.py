import os
from urllib.parse import quote_plus

import psycopg2
import streamlit as st
from dotenv import load_dotenv

from ingestion import index_url_into_vector_store
from rag_chain import create_rag_chain
from vector_store import get_vector_store


load_dotenv()

st.set_page_config(
    page_title="Universal URL Research Tool",
    page_icon=":link:",
    layout="wide",
)


def inject_styles():
    """Apply a more intentional, chat-first visual design."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
          --bg: #07111f;
          --panel: rgba(11, 22, 38, 0.78);
          --panel-strong: rgba(12, 25, 43, 0.94);
          --line: rgba(157, 188, 255, 0.18);
          --accent: #4ecdc4;
          --accent-2: #ffd166;
          --text: #f5f7fb;
          --muted: #a8b5cc;
          --danger: #ff7b7b;
          --success: #52d273;
        }

        .stApp {
          background:
            radial-gradient(circle at top left, rgba(78, 205, 196, 0.16), transparent 30%),
            radial-gradient(circle at top right, rgba(255, 209, 102, 0.14), transparent 26%),
            linear-gradient(160deg, #07111f 0%, #0b1730 52%, #09101a 100%);
          color: var(--text);
        }

        html, body, [class*="css"]  {
          font-family: "Space Grotesk", sans-serif;
        }

        .block-container {
          padding-top: 2.2rem;
          padding-bottom: 2rem;
          max-width: 1280px;
        }

        h1, h2, h3 {
          color: var(--text);
          letter-spacing: -0.02em;
        }

        .hero-card, .control-card, .chat-shell, .status-card {
          background: var(--panel);
          backdrop-filter: blur(18px);
          border: 1px solid var(--line);
          border-radius: 26px;
          box-shadow: 0 24px 80px rgba(0, 0, 0, 0.24);
        }

        .hero-card {
          padding: 1.5rem 1.6rem;
          margin-bottom: 1rem;
        }

        .hero-kicker {
          color: var(--accent);
          text-transform: uppercase;
          letter-spacing: 0.16em;
          font-size: 0.78rem;
          font-weight: 700;
        }

        .hero-title {
          font-size: 3.35rem;
          line-height: 1;
          margin: 0.35rem 0 0.75rem 0;
          font-weight: 700;
        }

        .hero-copy {
          color: var(--muted);
          font-size: 1.03rem;
          line-height: 1.65;
          max-width: 58rem;
        }

        .control-card, .chat-shell, .status-card {
          padding: 1.2rem 1.2rem 1.1rem 1.2rem;
        }

        .panel-title {
          font-size: 1.15rem;
          font-weight: 700;
          margin-bottom: 0.15rem;
        }

        .panel-copy {
          color: var(--muted);
          font-size: 0.95rem;
          line-height: 1.5;
          margin-bottom: 1rem;
        }

        .source-chip {
          display: inline-block;
          padding: 0.38rem 0.72rem;
          margin: 0 0.45rem 0.45rem 0;
          border-radius: 999px;
          border: 1px solid rgba(78, 205, 196, 0.25);
          background: rgba(78, 205, 196, 0.08);
          color: #d5fff8;
          font-size: 0.88rem;
          line-height: 1.2;
          word-break: break-all;
        }

        .report-item {
          padding: 0.82rem 0.9rem;
          border-radius: 16px;
          margin-bottom: 0.7rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.06);
        }

        .report-label {
          font-family: "IBM Plex Mono", monospace;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          font-size: 0.76rem;
          margin-bottom: 0.28rem;
        }

        .report-success .report-label { color: var(--success); }
        .report-warning .report-label { color: var(--accent-2); }
        .report-error .report-label { color: var(--danger); }
        .report-note .report-label { color: var(--accent); }

        .report-url {
          color: var(--text);
          font-size: 0.92rem;
          margin-bottom: 0.25rem;
          word-break: break-all;
        }

        .report-message {
          color: var(--muted);
          font-size: 0.9rem;
          line-height: 1.45;
        }

        .chat-shell {
          min-height: 650px;
        }

        .chat-heading {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: 0.8rem;
          margin-bottom: 0.35rem;
        }

        .chat-subcopy {
          color: var(--muted);
          font-size: 0.93rem;
          margin-bottom: 0.8rem;
        }

        .stChatMessage {
          background: transparent !important;
        }

        .stChatMessage [data-testid="stMarkdownContainer"] p,
        .stChatMessage [data-testid="stMarkdownContainer"] li {
          font-size: 1rem;
          line-height: 1.72;
        }

        .stChatMessage [data-testid="stMarkdownContainer"] ul {
          padding-left: 1.3rem;
        }

        .source-block {
          margin-top: 0.7rem;
          padding: 0.8rem 0.95rem;
          border-radius: 16px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .source-block-title {
          color: var(--accent);
          font-family: "IBM Plex Mono", monospace;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          font-size: 0.78rem;
          margin-bottom: 0.45rem;
        }

        .source-link {
          color: #b9d7ff;
          text-decoration: none;
        }

        .source-link:hover {
          text-decoration: underline;
        }

        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input,
        div[data-testid="stChatInput"] textarea {
          background: rgba(255, 255, 255, 0.04) !important;
          border-radius: 18px !important;
          border: 1px solid rgba(157, 188, 255, 0.14) !important;
          color: var(--text) !important;
        }

        div[data-testid="stNumberInput"] label,
        div[data-testid="stTextInput"] label {
          color: var(--muted) !important;
          font-weight: 500;
        }

        .stButton > button {
          border-radius: 16px;
          border: 1px solid rgba(78, 205, 196, 0.28);
          background: linear-gradient(135deg, rgba(78, 205, 196, 0.22), rgba(255, 209, 102, 0.18));
          color: var(--text);
          font-weight: 700;
          min-height: 3rem;
        }

        .stButton > button:hover {
          border-color: rgba(78, 205, 196, 0.5);
          color: white;
        }

        div[data-testid="stAlert"] {
          border-radius: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state():
    """Initialize the session keys used by the app."""
    st.session_state.setdefault("rag_chain", None)
    st.session_state.setdefault("retriever", None)
    st.session_state.setdefault("active_urls", [])
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_index_report", [])

    if not st.session_state["chat_history"]:
        st.session_state["chat_history"] = [
            {
                "role": "assistant",
                "content": (
                    "Bring me one or more URLs, index them, and I will answer only from "
                    "retrieved source passages."
                ),
                "sources": [],
            }
        ]


def get_db_config():
    """Read database configuration strictly from environment variables."""

    def _get(name: str, default: str | None = None) -> str | None:
        value = os.getenv(name)
        if value:
            return value

        try:
            if name in st.secrets:
                return str(st.secrets[name])
        except Exception:
            pass

        return default

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
        raise ValueError(
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


def ensure_indexed_urls_table(conn):
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


def normalize_urls(raw_urls):
    """Normalize URL inputs and drop duplicates while preserving order."""
    normalized = []
    seen = set()

    for raw_url in raw_urls:
        url = (raw_url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        normalized.append(url)

    return normalized


def delete_indexed_urls_except(conn, keep_urls):
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


def delete_vector_rows_except(conn, collection_name, keep_urls):
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


def get_vector_indexed_urls(conn, collection_name, candidate_urls):
    """Return URLs that still have real stored vector rows."""
    if not candidate_urls:
        return set()

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT e.cmetadata->>'url'
            FROM langchain_pg_embedding AS e
            JOIN langchain_pg_collection AS c ON e.collection_id = c.uuid
            WHERE c.name = %s
              AND COALESCE(e.cmetadata->>'url', '') = ANY(%s)
            """,
            (collection_name, candidate_urls),
        )
        return {row[0] for row in cur.fetchall() if row[0]}


def reset_chat_history(active_urls):
    """Reset the chat transcript after a successful indexing run."""
    if active_urls:
        source_list = "\n".join(f"- `{url}`" for url in active_urls)
        intro = (
            "Your indexed source set is ready. I will stay grounded to these URLs:\n\n"
            f"{source_list}"
        )
    else:
        intro = (
            "I do not have any active indexed sources yet. Add a URL and run indexing first."
        )

    st.session_state["chat_history"] = [
        {"role": "assistant", "content": intro, "sources": []}
    ]


def record_assistant_message(content, sources=None):
    """Append an assistant message to chat history."""
    st.session_state["chat_history"].append(
        {"role": "assistant", "content": content, "sources": sources or []}
    )


def same_url_set(current_urls, active_urls):
    """Return True when both URL lists represent the same active source set."""
    return len(current_urls) == len(active_urls) and set(current_urls) == set(active_urls)


def build_connection_string():
    """Create the Postgres connection string for PGVector."""
    db_cfg = get_db_config()
    password_encoded = quote_plus(db_cfg["password"])
    return (
        f"postgresql://{db_cfg['user']}:{password_encoded}"
        f"@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['name']}?sslmode=require"
    )


def build_rag_stack():
    """Instantiate the vector store, retriever, and RAG chain."""
    connection_string = build_connection_string()
    vector_store = get_vector_store(connection_string, table_name="url_embeddings")
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    rag_chain = create_rag_chain(retriever)
    return vector_store, retriever, rag_chain


def index_active_urls(current_urls):
    """Index the current URL set and keep only this run active."""
    if not current_urls:
        return [], [], None, None

    vector_store, retriever, rag_chain = build_rag_stack()

    report = []
    active_urls = []

    conn = get_db_connection()
    ensure_indexed_urls_table(conn)

    try:
        delete_indexed_urls_except(conn, current_urls)
        delete_vector_rows_except(conn, "url_embeddings", current_urls)
        existing_vector_urls = get_vector_indexed_urls(conn, "url_embeddings", current_urls)

        for url in current_urls:
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
            except Exception as ingest_err:
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

    if active_urls:
        st.session_state["rag_chain"] = rag_chain
        st.session_state["retriever"] = retriever
        st.session_state["active_urls"] = active_urls
        reset_chat_history(active_urls)
    else:
        st.session_state["rag_chain"] = None
        st.session_state["retriever"] = None
        st.session_state["active_urls"] = []
        reset_chat_history([])

    st.session_state["last_index_report"] = report
    return active_urls, report, retriever, rag_chain


def render_source_chips(active_urls):
    """Render the current source list as chips."""
    if not active_urls:
        st.markdown(
            "<div class='panel-copy'>No active sources yet.</div>",
            unsafe_allow_html=True,
        )
        return

    chips = "".join(
        f"<span class='source-chip'>{url}</span>" for url in active_urls
    )
    st.markdown(chips, unsafe_allow_html=True)


def render_index_report(report):
    """Render the latest indexing report."""
    if not report:
        st.markdown(
            "<div class='panel-copy'>Index a URL set to see retrieval readiness here.</div>",
            unsafe_allow_html=True,
        )
        return

    for item in report:
        st.markdown(
            f"""
            <div class='report-item report-{item["state"]}'>
              <div class='report-label'>{item["state"]}</div>
              <div class='report-url'>{item["url"]}</div>
              <div class='report-message'>{item["message"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chat_history():
    """Render the grounded Q&A transcript."""
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                source_items = "".join(
                    f"<li><a class='source-link' href='{url}' target='_blank'>{url}</a></li>"
                    for url in message["sources"]
                )
                st.markdown(
                    f"""
                    <div class='source-block'>
                      <div class='source-block-title'>Sources</div>
                      <ul>{source_items}</ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def extract_source_urls(source_docs):
    """Return unique source URLs from retrieved documents."""
    seen_urls = set()
    unique_urls = []

    for doc in source_docs:
        meta = getattr(doc, "metadata", {}) or {}
        src_url = meta.get("url")
        if src_url and src_url not in seen_urls:
            seen_urls.add(src_url)
            unique_urls.append(src_url)

    return unique_urls


def answer_question(question, current_urls):
    """Answer a user question only when the current run is grounded."""
    rag_chain = st.session_state.get("rag_chain")
    retriever = st.session_state.get("retriever")
    active_urls = st.session_state.get("active_urls", [])

    st.session_state["chat_history"].append(
        {"role": "user", "content": question, "sources": []}
    )

    if not rag_chain or not retriever:
        record_assistant_message(
            "I do not have an active retrieval stack yet. Index at least one URL first."
        )
        return

    if not same_url_set(current_urls, active_urls):
        record_assistant_message(
            "The visible URL list no longer matches the last successful indexing run. "
            "Click `Index Sources` again before asking another question."
        )
        return

    relevant_docs = retriever.invoke(question)
    relevant_docs = [
        doc for doc in relevant_docs if getattr(doc, "page_content", "").strip()
    ]

    if not relevant_docs:
        record_assistant_message(
            "I could not retrieve any grounded passages for that question. Try a clearer "
            "question, re-index the page, or switch to a source with richer visible text."
        )
        return

    result = rag_chain({"query": question})
    answer = result.get("result", "No answer returned.").strip()
    source_docs = result.get("source_documents") or relevant_docs
    source_urls = extract_source_urls(source_docs)

    if not source_urls:
        record_assistant_message(
            "I retrieved context, but the result came back without usable source metadata, "
            "so I am not treating the answer as grounded. Re-indexing the URL is the safest next step."
        )
        return

    record_assistant_message(answer, sources=source_urls)


inject_styles()
init_session_state()

st.markdown(
    """
    <div class='hero-card'>
      <div class='hero-kicker'>Grounded Research Workspace</div>
      <div class='hero-title'>Universal URL Research Tool</div>
      <div class='hero-copy'>
        Build a focused source set, prune stale context automatically, and ask grounded
        questions in a chat flow that keeps citations visible.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([0.9, 1.55], gap="large")

with left_col:
    st.markdown(
        """
        <div class='control-card'>
          <div class='panel-title'>Source Set</div>
          <div class='panel-copy'>
            Index only the URLs you want active right now. Existing vectors for matching URLs stay.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("index_form", clear_on_submit=False):
        num_urls = st.number_input(
            "How many URLs do you want to index?",
            min_value=1,
            max_value=20,
            step=1,
            value=1,
        )

        raw_urls = []
        for i in range(num_urls):
            raw_urls.append(st.text_input(f"URL {i + 1}", key=f"url_{i}"))

        index_clicked = st.form_submit_button("Index Sources")

    current_urls = normalize_urls(raw_urls)

    if index_clicked:
        if not current_urls:
            st.warning("Please enter at least one URL before indexing.")
        else:
            try:
                active_urls, report, _, _ = index_active_urls(current_urls)
                if active_urls and all(item["state"] != "error" for item in report):
                    st.success("Active source set refreshed.")
                elif active_urls:
                    st.warning("Some URLs are ready, but others need attention.")
                else:
                    st.warning("No URLs became retrievable in this run.")
            except Exception as err:
                st.error(f"An error occurred while indexing: {err}")

    st.markdown(
        """
        <div class='status-card' style='margin-top: 1rem;'>
          <div class='panel-title'>Currently Indexed</div>
          <div class='panel-copy'>
            These are the only URLs this chat should answer from.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_source_chips(st.session_state.get("active_urls", []))

    st.markdown(
        """
        <div class='status-card' style='margin-top: 1rem;'>
          <div class='panel-title'>Index Report</div>
          <div class='panel-copy'>
            We only trust runs that produced retrievable chunks with source metadata.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_index_report(st.session_state.get("last_index_report", []))

with right_col:
    st.markdown(
        """
        <div class='chat-shell'>
          <div class='chat-heading'>
            <div class='panel-title'>Grounded Q&A</div>
          </div>
          <div class='chat-subcopy'>
            Ask questions in plain language. Answers only count as valid when retriever-backed source URLs come back.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_chat_history()

question = st.chat_input("Ask a question about the active source set")

if question:
    current_urls = normalize_urls(
        [st.session_state.get(f"url_{i}", "") for i in range(num_urls)]
    )
    try:
        answer_question(question, current_urls)
        st.rerun()
    except Exception as err:
        st.error(f"An error occurred while answering your question: {err}")
