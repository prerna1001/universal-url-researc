from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_vector_store(connection_string, table_name="url_embeddings"):
    """
    Initialize a PGVector vector store for storing embeddings.

    Args:
        connection_string (str): PostgreSQL connection string.
        table_name (str): Name of the table to store embeddings.

    Returns:
        PGVector: Initialized PGVector instance.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create (or load) an empty PGVector collection so we don't call
    # the low-level PGVector __init__ with unsupported args like `table_name`.
    # `collection_name` is effectively the table/collection name in Postgres.
    vector_store = PGVector.from_texts(
        texts=[],
        embedding=embeddings,
        collection_name=table_name,
        connection_string=connection_string,
    )
    return vector_store