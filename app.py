import streamlit as st
import psycopg2
import os
from rag_chain import create_rag_chain  # Import your RAG chain function
from vector_store import get_vector_store  # Import vector store setup
from ingestion import index_url_into_vector_store


# Database connection setup
def get_db_connection():
    """Return a psycopg2 connection using .env configuration.

    Handles missing or non-numeric DB_PORT values gracefully.
    """
    db_port_raw = os.getenv("DB_PORT")
    db_port = int(db_port_raw) if db_port_raw and db_port_raw.isdigit() else 5432

    db_host_raw = os.getenv("DB_HOST")
    db_host = db_host_raw if db_host_raw and db_host_raw.lower() != "none" else "localhost"

    db_name_raw = os.getenv("DB_NAME")
    db_name = db_name_raw if db_name_raw and db_name_raw.lower() != "none" else "universal_url_research"

    db_user_raw = os.getenv("DB_USER")
    db_user = db_user_raw if db_user_raw and db_user_raw.lower() != "none" else "postgres"

    db_password_raw = os.getenv("DB_PASSWORD")
    db_password = db_password_raw if db_password_raw and db_password_raw.lower() != "none" else "password"

    return psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port,
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

# Title
st.title("Universal URL Research Tool")

# Step 1: Input number of URLs
num_urls = st.number_input("How many URLs do you want to index?", min_value=1, max_value=20, step=1)

# Step 2: Dynamic URL input fields
urls = []
for i in range(num_urls):
    url = st.text_input(f"URL {i + 1}")
    urls.append(url)

# Step 3: Index Sources Button
if st.button("Index Sources"):
    try:
        # Initialize vector store and RAG chain using the same DB settings from .env
        db_port_raw = os.getenv("DB_PORT")
        db_port_str = db_port_raw if db_port_raw and db_port_raw.isdigit() else "5432"

        db_host_raw = os.getenv("DB_HOST")
        db_host_str = db_host_raw if db_host_raw and db_host_raw.lower() != "none" else "localhost"

        db_name_raw = os.getenv("DB_NAME")
        db_name = db_name_raw if db_name_raw and db_name_raw.lower() != "none" else "universal_url_research"

        db_user_raw = os.getenv("DB_USER")
        db_user = db_user_raw if db_user_raw and db_user_raw.lower() != "none" else "postgres"

        db_password_raw = os.getenv("DB_PASSWORD")
        db_password = db_password_raw if db_password_raw and db_password_raw.lower() != "none" else "password"

        connection_string = (
            f"postgresql://{db_user}:{db_password}"\
            f"@{db_host_str}:{db_port_str}/{db_name}"\
        )

        vector_store = get_vector_store(connection_string, table_name="url_embeddings")
        retriever = vector_store.as_retriever()
        rag_chain = create_rag_chain(retriever)

        # Cache the RAG chain in session state for later question answering
        st.session_state["rag_chain"] = rag_chain

        # Connect to the database and ensure the table exists
        conn = get_db_connection()
        ensure_indexed_urls_table(conn)
        cursor = conn.cursor()

        # Index the URLs
        for url in urls:
            if not url:
                continue

            st.write(f"Indexing: {url}")

            try:
                # Fetch, chunk, and store this URL into the vector store
                page_text = index_url_into_vector_store(url, vector_store)
            except Exception as ingest_err:
                st.error(f"Failed to index {url}: {ingest_err}")
                continue

            # Also store the full cleaned content in the relational table
            cursor.execute(
                "INSERT INTO indexed_urls (url, indexed_content) VALUES (%s, %s)",
                (url, page_text)
            )
            conn.commit()

        cursor.close()
        conn.close()

        st.success("Sources indexed and stored successfully!")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Step 4: Question Input
question = st.text_input("Ask a question based on the indexed URLs")

# Step 5: Results Display
if question:
    rag_chain = st.session_state.get("rag_chain")

    if not rag_chain:
        st.warning("Please index some URLs first using 'Index Sources'.")
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
