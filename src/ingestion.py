import os
import logging
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

logger = logging.getLogger(__name__)


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
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

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
    """Loads a single document based on its extension.
    Uses Tree-sitter for code files (.py, .js), standard loaders for others."""
    ext = os.path.splitext(filepath)[1].lower()

    # Try Tree-sitter for code files
    if ext in (".py", ".js"):
        docs = parse_code_file(filepath)
        if docs is not None:
            for doc in docs:
                doc.metadata["file_hash"] = get_file_hash(filepath)
            return docs
        # Fallback to TextLoader if Tree-sitter failed

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(filepath)
        elif ext == ".csv":
            loader = CSVLoader(filepath)
        elif ext == ".docx":
            loader = Docx2txtLoader(filepath)
        else:  # Default for .txt, .md, .py, etc
            loader = TextLoader(filepath, encoding="utf-8")

        docs = loader.load()
        for doc in docs:
            doc.metadata["source_type"] = "local"
            doc.metadata["file_hash"] = get_file_hash(filepath)

        return docs
    except Exception as e:
        logger.error(f"Failed to load document '{filepath}': {e}")
        return []


def process_directory(dir_path: str) -> tuple[list[Document], list[str]]:
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
    splitter = get_text_splitter()

    for root, _, files in os.walk(dir_path):
        for file in files:
            filepath = os.path.join(root, file)
            filepath = os.path.abspath(filepath)

            # File Filtering
            if not is_file_allowed(filepath):
                continue

            # Incremental Hash Checking (persistent cache)
            current_hash = get_file_hash(filepath)
            if cache.get(filepath) == current_hash:
                skipped_count += 1
                continue

            # Process new/changed file
            logger.info(f"Loading '{filepath}'...")
            docs = load_local_document(filepath)

            if docs:
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
                changed_sources.append(filepath)
                # Update cache
                cache[filepath] = current_hash

    # Save cache after processing all files
    save_cache(cache)

    logger.info(
        f"Found {len(all_chunks)} chunks from {len(changed_sources)} changed files. "
        f"Skipped {skipped_count} unchanged files."
    )
    return all_chunks, changed_sources
