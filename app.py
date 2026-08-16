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
    """Apply a simple, chat-first visual design."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
          --bg: #f5f7fb;
          --panel: #ffffff;
          --line: #dfe5ef;
          --line-strong: #cbd5e1;
          --text: #0f172a;
          --muted: #5b6474;
          --accent: #2563eb;
          --accent-soft: #eff6ff;
          --success: #0f9f6e;
          --warning: #b7791f;
          --danger: #c24141;
        }

        .stApp {
          background: var(--bg);
          color: var(--text);
        }

        html, body, [class*="css"]  {
          font-family: "Inter", sans-serif;
        }

        .block-container {
          padding-top: 1.4rem;
          padding-bottom: 1.6rem;
          max-width: 980px;
        }

        .app-frame {
          max-width: 860px;
          margin: 0 auto;
        }

        .app-header {
          margin-bottom: 1rem;
        }

        .app-title {
          font-size: 1.85rem;
          font-weight: 700;
          letter-spacing: -0.03em;
          color: var(--text);
          margin-bottom: 0.25rem;
        }

        .app-subtitle {
          color: var(--muted);
          font-size: 0.98rem;
          line-height: 1.5;
        }

        .source-strip, .source-panel, .chat-card, .composer-card {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 20px;
          box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        }

        .source-strip {
          padding: 0.9rem 1rem;
          margin-bottom: 0.9rem;
        }

        .source-strip-title {
          font-size: 0.92rem;
          font-weight: 600;
          margin-bottom: 0.25rem;
          color: var(--text);
        }

        .source-strip-copy {
          color: var(--muted);
          font-size: 0.9rem;
        }

        .source-panel {
          padding: 1rem;
          margin-bottom: 0.9rem;
        }

        .panel-title {
          font-size: 1rem;
          font-weight: 600;
          margin-bottom: 0.2rem;
          color: var(--text);
        }

        .panel-copy {
          color: var(--muted);
          font-size: 0.92rem;
          line-height: 1.5;
          margin-bottom: 0.8rem;
        }

        .source-chip {
          display: inline-block;
          padding: 0.35rem 0.7rem;
          margin: 0 0.42rem 0.42rem 0;
          border-radius: 999px;
          border: 1px solid var(--line-strong);
          background: #f8fafc;
          color: var(--text);
          font-size: 0.88rem;
          word-break: break-all;
        }

        .chat-card {
          min-height: 520px;
          padding: 1rem;
          margin-bottom: 0.85rem;
        }

        .chat-title {
          font-size: 0.96rem;
          font-weight: 600;
          margin-bottom: 0.2rem;
          color: var(--text);
        }

        .chat-copy {
          color: var(--muted);
          font-size: 0.9rem;
          margin-bottom: 0.85rem;
        }

        .composer-card {
          padding: 0.85rem;
          position: sticky;
          bottom: 0.75rem;
        }

        .stChatMessage {
          background: transparent !important;
          padding-left: 0 !important;
          padding-right: 0 !important;
        }

        .stChatMessage [data-testid="stMarkdownContainer"] p,
        .stChatMessage [data-testid="stMarkdownContainer"] li {
          font-size: 0.98rem;
          line-height: 1.68;
          color: var(--text);
        }

        .report-item {
          padding: 0.8rem 0.85rem;
          border-radius: 16px;
          margin-bottom: 0.65rem;
          background: #f8fafc;
          border: 1px solid var(--line);
        }

        .report-label {
          text-transform: uppercase;
          letter-spacing: 0.08em;
          font-size: 0.76rem;
          margin-bottom: 0.28rem;
        }

        .report-note .report-label { color: var(--accent); }
        .report-success .report-label { color: var(--success); }
        .report-warning .report-label { color: var(--warning); }
        .report-error .report-label { color: var(--danger); }

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

        .source-block {
          margin-top: 0.7rem;
          padding: 0.8rem 0.95rem;
          border-radius: 16px;
          background: #f8fafc;
          border: 1px solid var(--line);
        }

        .source-block-title {
          color: var(--muted);
          text-transform: uppercase;
          letter-spacing: 0.08em;
          font-size: 0.78rem;
          margin-bottom: 0.45rem;
        }

        .source-link {
          color: var(--accent);
          text-decoration: none;
        }

        .source-link:hover {
          text-decoration: underline;
        }

        div[data-testid="stTextArea"] textarea,
        div[data-testid="stTextInput"] input {
          background: #ffffff !important;
          border-radius: 16px !important;
          border: 1px solid var(--line-strong) !important;
          color: var(--text) !important;
        }

        div[data-testid="stTextArea"] label,
        div[data-testid="stTextInput"] label {
          color: var(--muted) !important;
          font-weight: 500;
        }

        .stButton > button {
          border-radius: 14px;
          border: 1px solid var(--line-strong);
          background: #ffffff;
          color: var(--text);
          font-weight: 600;
          min-height: 2.8rem;
        }

        .stButton > button:hover {
          border-color: var(--accent);
          color: var(--accent);
        }

        div[data-testid="stAlert"] {
          border-radius: 18px;
        }

        div[data-testid="stForm"] {
          border: 0 !important;
          padding: 0 !important;
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
    st.session_state.setdefault("source_urls_text", "")
    st.session_state.setdefault("sources_panel_open", False)

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


def render_index_summary(report):
    """Render a compact indexing summary inside the source drawer."""
    if not report:
        st.caption("Paste one URL per line, then refresh the source set.")
        return

    success_count = sum(1 for item in report if item["state"] == "success")
    note_count = sum(1 for item in report if item["state"] == "note")
    warning_count = sum(1 for item in report if item["state"] == "warning")
    error_count = sum(1 for item in report if item["state"] == "error")

    if success_count or note_count:
        st.success(
            f"Ready: {success_count} newly indexed, {note_count} reused from the current source set."
        )
    if warning_count:
        st.warning(f"Needs review: {warning_count} URL(s) fetched but did not produce usable chunks.")
    if error_count:
        st.error(f"Failed: {error_count} URL(s) could not be indexed.")

    with st.expander("See indexing details"):
        render_index_report(report)


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
    <div class='app-frame'>
      <div class='app-header'>
        <div class='app-title'>Universal URL Research Tool</div>
        <div class='app-subtitle'>
          Add sources only when you need them, then ask questions in one simple chat.
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='app-frame'>
      <div class='source-strip'>
        <div class='source-strip-title'>Active Sources</div>
        <div class='source-strip-copy'>
          Answers should stay grounded only in these indexed URLs.
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
render_source_chips(st.session_state.get("active_urls", []))

if st.session_state.get("sources_panel_open"):
    st.markdown(
        """
        <div class='app-frame'>
        <div class='source-panel'>
          <div class='panel-title'>Manage Sources</div>
          <div class='panel-copy'>
            Paste one URL per line, then refresh the source set.
          </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("source_drawer_form", clear_on_submit=False):
        source_text = st.text_area(
            "URLs",
            key="source_urls_text",
            placeholder="https://example.com/article-1\nhttps://example.com/article-2",
            height=160,
            label_visibility="collapsed",
        )
        index_clicked = st.form_submit_button("Refresh Sources")

    current_urls = normalize_urls(source_text.splitlines())

    if index_clicked:
        if not current_urls:
            st.warning("Please add at least one URL before indexing.")
        else:
            try:
                active_urls, report, _, _ = index_active_urls(current_urls)
                st.session_state["source_urls_text"] = "\n".join(active_urls or current_urls)
                if active_urls:
                    st.session_state["sources_panel_open"] = False
                if active_urls and all(item["state"] != "error" for item in report):
                    st.success("Source set refreshed.")
                elif active_urls:
                    st.warning("Some URLs are ready, but others need attention.")
                else:
                    st.warning("No URLs became retrievable in this run.")
            except Exception as err:
                st.error(f"An error occurred while indexing: {err}")

    render_index_summary(st.session_state.get("last_index_report", []))

st.markdown(
    """
    <div class='app-frame'>
      <div class='chat-card'>
        <div class='chat-title'>Chat</div>
        <div class='chat-copy'>
          Ask naturally. Grounded replies will show their sources right underneath.
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
render_chat_history()

st.markdown("<div class='app-frame'><div class='composer-card'>", unsafe_allow_html=True)
composer_col, sources_col, send_col = st.columns([6.5, 1.8, 1.2], gap="small")

with composer_col:
    question = st.text_input(
        "Question",
        key="composer_input",
        placeholder="Ask anything about your sources...",
        label_visibility="collapsed",
    )

with sources_col:
    if st.button("Add Sources", use_container_width=True):
        st.session_state["sources_panel_open"] = not st.session_state["sources_panel_open"]
        st.rerun()

with send_col:
    send_clicked = st.button("Send", use_container_width=True)

st.markdown("</div></div>", unsafe_allow_html=True)

if send_clicked and question.strip():
    current_urls = normalize_urls(
        st.session_state.get("source_urls_text", "").splitlines()
    )
    try:
        answer_question(question.strip(), current_urls)
        st.session_state["composer_input"] = ""
        st.rerun()
    except Exception as err:
        st.error(f"An error occurred while answering your question: {err}")
