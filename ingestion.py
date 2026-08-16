import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_HTML_BYTES = 1_500_000
MAX_VISIBLE_TEXT_CHARS = 120_000


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

    resp = requests.get(url, timeout=20, headers=headers, stream=True)

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:  # pragma: no cover - network dependent
        # Provide a clearer error message for 403s and similar cases
        status = resp.status_code
        if status == 403:
            raise RuntimeError(f"HTTP 403 Forbidden while fetching {url}. The site is blocking automated access.") from exc
        raise

    content_type = (resp.headers.get("Content-Type") or "").lower()
    if content_type and "html" not in content_type and "xml" not in content_type:
        raise RuntimeError(
            f"Unsupported content type for {url}: {content_type}. "
            "Try a normal article page instead of a PDF or download page."
        )

    content_length = resp.headers.get("Content-Length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_HTML_BYTES:
        raise RuntimeError(
            f"{url} is too large to index on the current deployment limit. "
            "Try a lighter article page."
        )

    body = bytearray()
    for chunk in resp.iter_content(chunk_size=16_384):
        if not chunk:
            continue
        body.extend(chunk)
        if len(body) > MAX_HTML_BYTES:
            raise RuntimeError(
                f"{url} is too large to index on the current deployment limit. "
                "Try a lighter article page."
            )

    html = body.decode(resp.encoding or "utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    text = soup.get_text(separator="\n")
    # Normalize whitespace
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    if len(text) > MAX_VISIBLE_TEXT_CHARS:
        text = text[:MAX_VISIBLE_TEXT_CHARS]

    return text


def split_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Split raw text into overlapping chunks suitable for embedding."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def index_url_into_vector_store(url: str, vector_store) -> tuple[str, int]:
    """Fetch, chunk, and store a single URL into the vector store.

    Returns the full cleaned page text and the number of stored chunks.
    """

    page_text = fetch_url_text(url)
    chunks = split_into_chunks(page_text)

    if not chunks:
        return page_text, 0

    metadatas = [{"url": url, "chunk_index": i} for i in range(len(chunks))]
    vector_store.add_texts(chunks, metadatas=metadatas)

    return page_text, len(chunks)
