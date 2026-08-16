from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

MAX_HTML_BYTES = 1_500_000
MAX_VISIBLE_TEXT_CHARS = 120_000
MIN_STATIC_TEXT_CHARS = 1_200
REQUEST_TIMEOUT_SECONDS = 20
PLAYWRIGHT_NAVIGATION_TIMEOUT_MS = 25_000
PLAYWRIGHT_RENDER_WAIT_MS = 1_500


@dataclass
class FetchResult:
    text: str
    mode: str


def build_headers() -> dict[str, str]:
    return {
        "User-Agent": "UniversalURLResearchTool/0.1 (contact: example@example.com)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


def normalize_visible_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    normalized = "\n".join(line for line in lines if line)
    return normalized[:MAX_VISIBLE_TEXT_CHARS]


def html_to_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    return normalize_visible_text(soup.get_text(separator="\n"))


def validate_fetch_response(url: str, resp: requests.Response) -> None:
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:  # pragma: no cover - network dependent
        status = resp.status_code
        if status == 403:
            raise RuntimeError(
                f"HTTP 403 Forbidden while fetching {url}. The site is blocking automated access."
            ) from exc
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


def read_limited_response(url: str, resp: requests.Response) -> str:
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

    return body.decode(resp.encoding or "utf-8", errors="ignore")


def looks_like_js_shell(html: str, text: str) -> bool:
    if len(text) < MIN_STATIC_TEXT_CHARS:
        return True

    html_lower = html.lower()
    markers = [
        "__next",
        "__nuxt",
        'id="root"',
        "data-reactroot",
        "window.__initial_state__",
        "window.__nuxt__",
        "webpack",
        "hydration",
    ]
    return any(marker in html_lower for marker in markers) and len(text) < (MIN_STATIC_TEXT_CHARS * 2)


def fetch_url_text_with_requests(url: str) -> tuple[str, str]:
    resp = requests.get(
        url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers=build_headers(),
        stream=True,
    )

    validate_fetch_response(url, resp)
    html = read_limited_response(url, resp)
    text = html_to_visible_text(html)
    return html, text


def fetch_url_text_with_playwright(url: str) -> str:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--no-sandbox",
            ],
        )
        try:
            page = browser.new_page(user_agent=build_headers()["User-Agent"])
            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=PLAYWRIGHT_NAVIGATION_TIMEOUT_MS,
            )
            page.wait_for_timeout(PLAYWRIGHT_RENDER_WAIT_MS)
            html = page.content()
            return html_to_visible_text(html)
        except PlaywrightTimeoutError as exc:
            raise RuntimeError(
                f"Timed out while browser-rendering {url}. The page may be too heavy or blocking automation."
            ) from exc
        finally:
            browser.close()


def fetch_url_text(url: str) -> FetchResult:
    """Fetch a URL and return cleaned visible text.

    First try a lightweight request/HTML parse. If the page looks like a JS-rendered
    shell or returns too little visible text, retry with Playwright so modern pages
    can fully render before extraction.
    """

    html, text = fetch_url_text_with_requests(url)

    if not looks_like_js_shell(html, text):
        return FetchResult(text=text, mode="static")

    try:
        browser_text = fetch_url_text_with_playwright(url)
    except Exception as exc:  # pragma: no cover - browser/runtime dependent
        if text.strip():
            return FetchResult(text=text, mode="static")
        raise RuntimeError(
            f"{url} appears to need browser rendering, but Playwright could not extract usable text: {exc}"
        ) from exc

    if len(browser_text.strip()) > len(text.strip()):
        return FetchResult(text=browser_text, mode="browser")

    return FetchResult(text=text, mode="static")


def split_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Split raw text into overlapping chunks suitable for embedding."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def index_url_into_vector_store(url: str, vector_store) -> tuple[str, int, str]:
    """Fetch, chunk, and store a single URL into the vector store.

    Returns the full cleaned page text, the number of stored chunks,
    and the extraction mode used.
    """

    fetch_result = fetch_url_text(url)
    page_text = fetch_result.text
    chunks = split_into_chunks(page_text)

    if not chunks:
        return page_text, 0, fetch_result.mode

    metadatas = [{"url": url, "chunk_index": i} for i in range(len(chunks))]
    vector_store.add_texts(chunks, metadatas=metadatas)

    return page_text, len(chunks), fetch_result.mode
