import streamlit as st
import psycopg2
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from rag_chain import create_rag_chain
from vector_store import get_vector_store
from ingestion import index_url_into_vector_store


# Load environment variables from a local .env file 
load_dotenv()


def get_db_config():
    """Read database configuration strictly from environment variables.

    Required: DB_HOST, DB_NAME, DB_USER, DB_PASSWORD.
    Optional: DB_PORT (defaults to 5432 if missing/invalid).
    """
    # Helper that prefers real environment variables, but also falls back to
    # Streamlit Cloud secrets (st.secrets) when running in the hosted app.
    def _get(name: str, default: str | None = None) -> str | None:
        # 1) Standard OS env (local .env via load_dotenv or platform-level env)
        value = os.getenv(name)
        if value:
            return value

        # 2) Streamlit secrets (used on Streamlit Cloud)
        try:
            if name in st.secrets:
                return str(st.secrets[name])
        except Exception:
            # If st.secrets is not available or behaves unexpectedly, ignore
            # and fall through to default.
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


# Database connection setup
def get_db_connection():
    """Return a psycopg2 connection using environment configuration only."""
    cfg = get_db_config()

    return psycopg2.connect(
        dbname=cfg["name"],
        user=cfg["user"],
        password=cfg["password"],
        host=cfg["host"],
        port=cfg["port"],
        sslmode="require",  # Supabase and most hosted Postgres instances expect SSL
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
    """Remove relational URL records that are no longer part of the active run."""
    with conn.cursor() as cur:
        if keep_urls:
            cur.execute(
                "DELETE FROM indexed_urls WHERE NOT (url = ANY(%s))",
                (keep_urls,),
            )
        else:
            cur.execute("DELETE FROM indexed_urls")
        conn.commit()


def get_existing_indexed_urls(conn, candidate_urls):
    """Return the subset of requested URLs already indexed in the relational table."""
    if not candidate_urls:
        return set()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT url FROM indexed_urls WHERE url = ANY(%s)",
            (candidate_urls,),
        )
        return {row[0] for row in cur.fetchall()}


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

# Title
st.title("Universal URL Research Tool")

# Step 1: Input number of URLs
num_urls = st.number_input("How many URLs do you want to index?", min_value=1, max_value=20, step=1)

# Step 2: Dynamic URL input fields
urls = []
for i in range(num_urls):
    url = st.text_input(f"URL {i + 1}")
    urls.append(url)

current_urls = normalize_urls(urls)

# Step 3: Index Sources Button
if st.button("Index Sources"):
    try:
        if not current_urls:
            st.warning("Please enter at least one URL before indexing.")
            st.stop()

        # Initialize vector store and RAG chain using DB settings from environment
        db_cfg = get_db_config()

        # URL-encode password so special characters like '@' don't break the URI
        password_encoded = quote_plus(db_cfg["password"])

        connection_string = (
            f"postgresql://{db_cfg['user']}:{password_encoded}"
            f"@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['name']}?sslmode=require"
        )

        vector_store = get_vector_store(connection_string, table_name="url_embeddings")
        retriever = vector_store.as_retriever()
        rag_chain = create_rag_chain(retriever)

        # Connect to the database and ensure the table exists
        conn = get_db_connection()
        ensure_indexed_urls_table(conn)
        cursor = conn.cursor()

        # Keep only the URLs for the active indexing run.
        delete_indexed_urls_except(conn, current_urls)
        delete_vector_rows_except(conn, "url_embeddings", current_urls)
        existing_urls = get_existing_indexed_urls(conn, current_urls)

        active_urls = []
        new_index_count = 0
        failed_urls = []

        # Index the URLs
        for url in current_urls:
            if url in existing_urls:
                st.write(f"Keeping existing index: {url}")
                active_urls.append(url)
                continue

            st.write(f"Indexing: {url}")

            try:
                # Fetch, chunk, and store this URL into the vector store
                page_text = index_url_into_vector_store(url, vector_store)
            except Exception as ingest_err:
                st.error(f"Failed to index {url}: {ingest_err}")
                failed_urls.append(url)
                continue

            # Also store the full cleaned content in the relational table
            cursor.execute(
                "INSERT INTO indexed_urls (url, indexed_content) VALUES (%s, %s)",
                (url, page_text)
            )
            conn.commit()
            active_urls.append(url)
            new_index_count += 1

        cursor.close()
        conn.close()

        # Cache the active retrieval state only for the URLs that survived this run.
        st.session_state["rag_chain"] = rag_chain
        st.session_state["active_urls"] = active_urls

        if active_urls and not failed_urls:
            if new_index_count == 0:
                st.success("Selected URLs were already indexed and are ready to use.")
            else:
                st.success("Sources indexed and stored successfully!")
        elif active_urls:
            st.warning(
                "Some URLs are ready to use, but others failed to index. "
                "Please review the errors above."
            )
        else:
            st.warning("No URLs were successfully indexed in this run.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Step 4: Question Input
question = st.text_input("Ask a question based on the indexed URLs")

# Step 5: Results Display
if question:
    rag_chain = st.session_state.get("rag_chain")
    active_urls = st.session_state.get("active_urls", [])

    if not rag_chain:
        st.warning("Please index some URLs first using 'Index Sources'.")
    elif current_urls != active_urls:
        st.warning(
            "The current URL list does not match the last successful indexing run. "
            "Click 'Index Sources' again before asking a question."
        )
    else:
        try:
            result = rag_chain({"query": question})
            answer = result.get("result", "No answer returned.")
            source_docs = result.get("source_documents", [])

            st.write("Answer:", answer)

            if source_docs:
                st.write("Sources:")

                # Deduplicate URLs so each source is shown only once
                seen_urls = set()
                unique_urls = []
                for doc in source_docs:
                    meta = getattr(doc, "metadata", {}) or {}
                    src_url = meta.get("url", "(no URL)")
                    if src_url not in seen_urls:
                        seen_urls.add(src_url)
                        unique_urls.append(src_url)

                for i, src_url in enumerate(unique_urls, start=1):
                    st.write(f"{i}. {src_url}")
            else:
                st.write("Sources: (no source documents returned)")
        except Exception as e:
            st.error(f"An error occurred while answering your question: {e}")
