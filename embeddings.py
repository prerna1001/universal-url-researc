from langchain.embeddings import HuggingFaceEmbeddings

def generate_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generate embeddings for a list of texts using LangChain's HuggingFaceEmbeddings.

    Args:
        texts (list): List of text strings to generate embeddings for.
        model_name (str): Name of the HuggingFace model to use for embedding generation.

    Returns:
        list: List of embeddings.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings.embed_documents(texts)