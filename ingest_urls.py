from langchain.document_loaders import WebBaseLoader

def load_urls(urls):
    """
    Load and clean content from a list of URLs using LangChain's WebBaseLoader.

    Args:
        urls (list): List of URLs to load.

    Returns:
        list: List of LangChain documents with cleaned content and metadata.
    """
    documents = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs = loader.load()
        documents.extend(docs)
    return documents