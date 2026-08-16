import os
from functools import lru_cache

import requests
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.embeddings import Embeddings


class WorkerEmbeddings(Embeddings):
    """Use the existing Cloudflare Worker to generate embeddings remotely."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            self.endpoint,
            json={"task": "embed", "texts": texts},
            headers={"Content-Type": "application/json"},
            timeout=120,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Embedding request failed with status {response.status_code}: {response.text}"
            ) from exc

        payload = response.json()
        vectors = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(vectors, list) or not vectors:
            raise RuntimeError("Embedding worker returned an unexpected response.")

        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batch_size = 24
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(self._request_embeddings(texts[start : start + batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self._request_embeddings([text])
        return vectors[0]


@lru_cache(maxsize=1)
def get_embeddings():
    endpoint = (os.getenv("WORKER_ENDPOINT") or "").strip()
    if not endpoint:
        raise RuntimeError(
            "WORKER_ENDPOINT is required for remote embeddings and question answering."
        )
    return WorkerEmbeddings(endpoint)


def get_vector_store(connection_string, table_name="url_embeddings"):
    """
    Initialize a PGVector vector store for storing embeddings.

    Args:
        connection_string (str): PostgreSQL connection string.
        table_name (str): Name of the table to store embeddings.

    Returns:
        PGVector: Initialized PGVector instance.
    """
    embeddings = get_embeddings()

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
