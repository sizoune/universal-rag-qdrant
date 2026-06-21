import os
import re
import logging
import ipaddress
import socket
import importlib.util
from collections.abc import Callable
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils import get_file_hash, is_file_allowed
from src.cache_store import load_cache, save_cache, get_content_hash
from src.code_parser import parse_code_file
from src.config import config
from src.layout_parser import chunk_elements, parse_docx, parse_pdf

logger = logging.getLogger(__name__)


def validate_public_http_url(url: str) -> None:
    """Validate URL to reduce SSRF risk (public http/https only)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are allowed")

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise ValueError("URL hostname is required")
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise ValueError("Localhost URLs are not allowed")

    # Direct IP validation
    try:
        ip = ipaddress.ip_address(hostname)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError("Private or non-routable IPs are not allowed")
        return
    except ValueError:
        # hostname is not an IP literal; continue DNS resolution checks
        pass

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve hostname: {hostname}") from exc

    for _, _, _, _, sockaddr in addresses:
        resolved_ip = ipaddress.ip_address(sockaddr[0])
        if (
            resolved_ip.is_private
            or resolved_ip.is_loopback
            or resolved_ip.is_link_local
            or resolved_ip.is_multicast
            or resolved_ip.is_reserved
            or resolved_ip.is_unspecified
        ):
            raise ValueError("Resolved to private or non-routable IP")


def _fetch_web_content_with_limits(url: str, headers: dict) -> str:
    max_redirects = 5
    max_bytes = config.WEB_MAX_CONTENT_BYTES if config.WEB_MAX_CONTENT_BYTES > 0 else 2097152
    current_url = url

    for _ in range(max_redirects + 1):
        validate_public_http_url(current_url)
        with requests.get(
            current_url,
            headers=headers,
            timeout=30,
            allow_redirects=False,
            stream=True,
        ) as response:
            if 300 <= response.status_code < 400 and response.headers.get("Location"):
                current_url = urljoin(current_url, response.headers["Location"])
                continue

            response.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(
                        f"Web content exceeded max size limit ({max_bytes} bytes)"
                    )
                chunks.append(chunk)
            encoding = response.encoding or "utf-8"
            return b"".join(chunks).decode(encoding, errors="replace")

    raise ValueError("Too many redirects while fetching URL")


# --- Boilerplate filtering -------------------------------------------------
# Drop chunks that carry no retrievable meaning: bare page-footer URLs (the BPS
# watermark repeats on every page), "Sumber:/Source:" stamps, page numbers, and
# tiny fragments. Conservative on purpose — real content chunks run ~500+ chars,
# so a short floor never touches statistical table rows. Validated against the
# live corpus: drops ~11%, all confirmed boilerplate.
_URL_RE = re.compile(r"https?://\S+")
_BOILERPLATE_RE = re.compile(
    r"^(sumber|source)\s*[:/].{0,40}$"
    r"|^https?://\S+$"
    r"|^(halaman|page|hal\.?)\s*\d+$",
    re.IGNORECASE,
)


def is_low_value_chunk(text: str) -> bool:
    """True if a chunk is boilerplate/noise not worth embedding or retrieving."""
    s = (text or "").strip()
    if len(s) < config.MIN_CHUNK_CHARS:
        return True
    one_line = " ".join(s.split())
    if _BOILERPLATE_RE.match(one_line):
        return True
    # essentially just a URL with a scrap of text around it
    if _URL_RE.search(s) and len(_URL_RE.sub("", s).strip()) < config.MIN_CHUNK_CHARS:
        return True
    return False


def drop_low_value_chunks(docs: list[Document]) -> list[Document]:
    """Return a new list with boilerplate chunks removed (no mutation)."""
    return [d for d in docs if not is_low_value_chunk(d.page_content)]


def get_text_splitter():
    """Returns a general RecursiveCharacterTextSplitter for chunking."""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )


def parse_web_url(url: str) -> tuple[list[Document], bool]:
    """Scrapes clean article text from a web URL.

    Returns:
        (docs, changed): list of chunked Documents and whether content changed.
        If content is unchanged from cache, returns ([], False).
    """
    logger.info(f"Scraping Web URL: {url}")
    try:
        headers = {
            "User-Agent": os.getenv(
                "USER_AGENT",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )
        }
        try:
            html = _fetch_web_content_with_limits(url, headers=headers)
        except ValueError as exc:
            raise ValueError(f"invalid web URL: {exc}") from exc
        soup = BeautifulSoup(html, "html.parser")

        # Remove noise elements
        for tag in soup.find_all(
            ["script", "style", "nav", "footer", "header", "aside", "noscript"]
        ):
            tag.decompose()

        # Try to find main content area (ordered by specificity)
        content = (
            soup.find("div", class_="mw-parser-output")  # Wikipedia
            or soup.find("article")  # Semantic HTML5
            or soup.find("main")  # Semantic HTML5
            or soup.find("div", id="content")  # Common pattern
            or soup.find("div", class_="content")  # Common pattern
            or soup.find("div", id="bodyContent")  # MediaWiki
            or soup.body  # Fallback to entire body
            or soup
        )

        # Extract clean text
        text = content.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        if not clean_text:
            logger.warning(f"No content extracted from {url}")
            return [], False

        # Check content hash against cache
        content_hash = get_content_hash(clean_text)
        cache = load_cache()

        if cache.get(url) == content_hash:
            logger.info(f"Content unchanged for {url} (hash match). Skipping.")
            return [], False

        logger.info(f"Extracted {len(clean_text)} characters from {url}")

        doc = Document(
            page_content=clean_text,
            metadata={"source": url, "source_type": "web"},
        )

        splitter = get_text_splitter()
        split_docs = splitter.split_documents([doc])

        # Save hash to cache
        cache[url] = content_hash
        save_cache(cache)

        return split_docs, True
    except Exception as e:
        logger.error(f"Failed to parse web URL '{url}': {e}")
        return [], False


def load_local_document(filepath: str) -> list[Document]:
    """Load a document and return final, ready-to-embed chunks.

    No further splitting is needed by the caller — chunks already respect
    the layout boundaries / max_chunk_size policy.

    Strategy:
    - .py / .js: Tree-sitter semantic chunks (legacy parser, already chunked).
    - .pdf / .docx: layout-aware parser (`src.layout_parser`); falls back to
      legacy text-only loaders when the layout parser raises.
    - .csv / .txt / .md / others: legacy loader + character-based splitter.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".py", ".js"):
        docs = parse_code_file(filepath)
        if docs is not None:
            for doc in docs:
                doc.metadata["file_hash"] = get_file_hash(filepath)
            return docs
        # Tree-sitter unavailable — fall through to legacy text path

    if ext == ".pdf":
        try:
            elements = parse_pdf(filepath)
            chunks = chunk_elements(elements)
            return _enrich_local_chunks(chunks, filepath)
        except Exception as exc:
            logger.warning(
                "Layout parser failed for PDF '%s': %s. Falling back to PyPDFLoader.",
                filepath,
                exc,
            )
            return _legacy_load_and_split(filepath)

    if ext == ".docx":
        try:
            elements = parse_docx(filepath)
            chunks = chunk_elements(elements)
            return _enrich_local_chunks(chunks, filepath)
        except Exception as exc:
            logger.warning(
                "Layout parser failed for DOCX '%s': %s. Falling back to Docx2txtLoader.",
                filepath,
                exc,
            )
            return _legacy_load_and_split(filepath)

    return _legacy_load_and_split(filepath)


def _enrich_local_chunks(chunks: list[Document], filepath: str) -> list[Document]:
    """Add source/source_type/file_hash metadata to layout-parsed chunks."""
    file_hash = get_file_hash(filepath)
    for doc in chunks:
        doc.metadata.setdefault("source", filepath)
        doc.metadata.setdefault("source_type", "local")
        doc.metadata["file_hash"] = file_hash
    return chunks


def _legacy_load_and_split(filepath: str) -> list[Document]:
    """Legacy single-shot loader + character splitter (parser_version=1).

    Used as a fallback when the layout parser fails, and as the primary path
    for formats not covered by Tree-sitter or layout-aware parsing
    (.csv, .txt, .md, .html, ...).
    """
    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == ".pdf":
            if importlib.util.find_spec("pypdf") is None:
                raise RuntimeError(
                    "PDF ingestion requires `pypdf`. Install dependency and restart service."
                )
            loader = PyPDFLoader(filepath)
        elif ext == ".csv":
            loader = CSVLoader(filepath)
        elif ext == ".docx":
            loader = Docx2txtLoader(filepath)
        else:
            loader = TextLoader(filepath, encoding="utf-8")

        raw_docs = loader.load()
    except RuntimeError:
        raise
    except Exception as exc:
        logger.error("Failed to load document '%s': %s", filepath, exc)
        return []

    splitter = get_text_splitter()
    chunks = splitter.split_documents(raw_docs)

    file_hash = get_file_hash(filepath)
    for doc in chunks:
        doc.metadata["source_type"] = "local"
        doc.metadata["file_hash"] = file_hash
        doc.metadata.setdefault("parser_version", 1)
    return chunks


def process_directory(
    dir_path: str,
    on_file_start: Callable[[str], None] | None = None,
) -> tuple[list[Document], list[str]]:
    """Scans directory, filters files, checks hashes for incremental updates,
    and returns chunked LangChain documents for Vector Store ingestion.

    Returns:
        (chunks, changed_sources): list of Documents and list of source paths that changed.
    """
    logger.info(f"Scanning directory: {dir_path}")
    all_chunks = []
    changed_sources = []
    skipped_count = 0

    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        logger.error(f"Directory '{dir_path}' does not exist.")
        return [], []

    cache = load_cache()
    max_size_mb = max(1, config.UPLOAD_MAX_BYTES // (1024 * 1024))

    for root, _, files in os.walk(dir_path):
        for file in files:
            filepath = os.path.join(root, file)
            filepath = os.path.abspath(filepath)

            # File Filtering
            if not is_file_allowed(filepath, max_size_mb=max_size_mb):
                continue

            # Incremental Hash Checking (persistent cache)
            current_hash = get_file_hash(filepath)
            if cache.get(filepath) == current_hash:
                skipped_count += 1
                continue

            # Process new/changed file. load_local_document now returns
            # FINAL chunks (no further splitting needed).
            logger.info(f"Loading '{filepath}'...")
            if on_file_start is not None:
                on_file_start(filepath)
            chunks = load_local_document(filepath)

            if chunks:
                all_chunks.extend(chunks)
                changed_sources.append(filepath)
                cache[filepath] = current_hash

    # Save cache after processing all files
    save_cache(cache)

    logger.info(
        f"Found {len(all_chunks)} chunks from {len(changed_sources)} changed files. "
        f"Skipped {skipped_count} unchanged files."
    )
    return all_chunks, changed_sources
