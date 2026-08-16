from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(documents, chunk_size=500, chunk_overlap=50):
    """
    Chunk text from documents using LangChain's RecursiveCharacterTextSplitter.

    Args:
        documents (list): List of LangChain documents to chunk.
        chunk_size (int): Maximum size of each chunk in characters.
        chunk_overlap (int): Overlap between chunks in characters.

    Returns:
        list: List of chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs