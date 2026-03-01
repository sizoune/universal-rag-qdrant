import os
import logging
from bs4 import BeautifulSoup
import requests
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils import get_file_hash, is_file_allowed

logger = logging.getLogger(__name__)

# Very basic dictionary cache for Incremental Updates (in-memory for MVP)
# In production, this would be a local SQLite DB or Redis
FILE_HASH_CACHE = {}


def get_text_splitter():
    """Returns a general RecursiveCharacterTextSplitter for chunking."""
    # Chunking limits as per PRD (100-1000 characters)
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )


def parse_web_url(url: str) -> list[Document]:
    """Scrapes clean text from a web URL using WebBaseLoader and BeautifulSoup."""
    logger.info(f"Scraping Web URL: {url}")
    try:
        # Adding user-agent to bypass simple blocking
        loader = WebBaseLoader(
            url,
            header_template={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        docs = loader.load()

        # Clean up text
        for doc in docs:
            soup = BeautifulSoup(doc.page_content, "html.parser")
            doc.page_content = soup.get_text(separator=" ", strip=True)
            doc.metadata["source_type"] = "web"

        splitter = get_text_splitter()
        split_docs = splitter.split_documents(docs)
        return split_docs
    except Exception as e:
        logger.error(f"Failed to parse web URL '{url}': {e}")
        return []


def load_local_document(filepath: str) -> list[Document]:
    """Loads a single document based on its extension."""
    ext = os.path.splitext(filepath)[1].lower()

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


def process_directory(dir_path: str) -> list[Document]:
    """
    Scans directory, filters files, checks hashes for incremental updates,
    and returns chunked LangChain documents for Vector Store ingestion.
    """
    logger.info(f"Scanning directory: {dir_path}")
    all_chunks = []
    skipped_count = 0

    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        logger.error(f"Directory '{dir_path}' does not exist.")
        return []

    splitter = get_text_splitter()

    for root, _, files in os.walk(dir_path):
        for file in files:
            filepath = os.path.join(root, file)
            filepath = os.path.abspath(filepath)

            # File Filtering
            if not is_file_allowed(filepath):
                continue

            # Incremental Hash Checking
            current_hash = get_file_hash(filepath)
            if (
                filepath in FILE_HASH_CACHE
                and FILE_HASH_CACHE[filepath] == current_hash
            ):
                # File is unchanged
                skipped_count += 1
                continue

            # Process new/changed file
            logger.info(f"Loading '{filepath}'...")
            docs = load_local_document(filepath)

            if docs:
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
                # Update Cache
                FILE_HASH_CACHE[filepath] = current_hash

    logger.info(
        f"Found {len(all_chunks)} chunks to embed. Skipped {skipped_count} unmodified files."
    )
    return all_chunks
