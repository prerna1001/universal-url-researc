import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter


def fetch_url_text(url: str) -> str:
    """Fetch a URL and return cleaned visible text.

    This is a simple MVP-style fetcher; it strips HTML tags and
    returns the page's main textual content as plain text.

    Adds a polite User-Agent so sites like Wikipedia are less
    likely to block the request.
    """

    headers = {
        "User-Agent": "UniversalURLResearchTool/0.1 (contact: example@example.com)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    resp = requests.get(url, timeout=20, headers=headers)

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:  # pragma: no cover - network dependent
        # Provide a clearer error message for 403s and similar cases
        status = resp.status_code
        if status == 403:
            raise RuntimeError(f"HTTP 403 Forbidden while fetching {url}. The site is blocking automated access.") from exc
        raise

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    text = soup.get_text(separator="\n")
    # Normalize whitespace
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text


def split_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Split raw text into overlapping chunks suitable for embedding."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def index_url_into_vector_store(url: str, vector_store) -> str:
    """Fetch, chunk, and store a single URL into the vector store.

    Returns the full cleaned page text for optional relational storage.
    """

    page_text = fetch_url_text(url)
    chunks = split_into_chunks(page_text)

    if not chunks:
        return page_text

    metadatas = [{"url": url, "chunk_index": i} for i in range(len(chunks))]
    vector_store.add_texts(chunks, metadatas=metadatas)

    return page_text
