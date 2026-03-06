# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) system using Qdrant vector database with multi-provider LLM/embedding support. Three interfaces: CLI (`main.py`), FastAPI REST API (`api.py`), and Telegram Bot (`src/telegram_bot.py`).

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run CLI
python main.py                          # Interactive menu
python main.py chat "question"          # Single-shot Q&A
python main.py ingest-web <url>         # Ingest webpage
python main.py ingest-file <path>       # Ingest file/directory

# Run API server
python -m uvicorn api:app --host 0.0.0.0 --port 8000

# Run Telegram bot
python main.py gateway

# Tests
python -m pytest tests/
python -m pytest tests/test_api.py      # Single test file
python -m pytest tests/test_api.py::test_name -v  # Single test

# Docker (runs qdrant + api + telegram bot)
docker compose up --build

# Qdrant locally
./run_qdrant.sh   # Start
./stop_qdrant.sh  # Stop
```

## Architecture

**Three entry points** share the same core logic in `src/`:

- `main.py` — CLI with argparse subcommands
- `api.py` — FastAPI app; all `/api/v1/*` endpoints require Bearer token auth (`src/api_auth.py`)
- `src/telegram_bot.py` — Bot gateway with chat, file upload, and web ingest commands

**Ingestion pipeline** (`src/ingestion.py`):
1. Load content (web scraping via BeautifulSoup, or file loaders for PDF/DOCX/CSV/TXT/MD/code)
2. SHA256 hash check against `.cache/ingestion_hashes.json` (`src/cache_store.py`) — skip if unchanged
3. Chunk text (1000 chars, 100 overlap) or parse code via Tree-sitter AST (`src/code_parser.py`)
4. Delete old chunks for the same source (`src/file_index.py` deduplication)
5. Generate embeddings + optional sparse BM25 vectors, upsert to Qdrant in batches of 100

**Retrieval & chat** (`src/chat.py`):
- LangChain retrieval chain with 5-turn conversation memory
- Dense search with score threshold filtering (default 0.7)
- Optional hybrid mode: dense + sparse BM25 with Reciprocal Rank Fusion (`src/hybrid_retriever.py`)
- Optional cross-encoder reranking (`src/reranker.py`)

**Multi-provider support** (`src/embedding_manager.py`, `src/chat.py`):
- Embeddings: OpenAI, Gemini, Ollama, Azure OpenAI — selected by `EMBEDDER_BASE_URL` pattern
- LLM: OpenAI, Gemini, Ollama, OpenAI-compatible — selected by `LLM_BASE_URL` pattern

## Configuration

All config via environment variables loaded in `src/config.py` (uses `python-dotenv`). See `.env.example` for the full list. Key settings:

- `EMBEDDER_BASE_URL`, `EMBEDDER_API_KEY`, `EMBEDDER_MODEL`, `EMBEDDER_DIMENSION` — embedding provider
- `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` — chat LLM provider
- `QDRANT_URL`, `QDRANT_COLLECTION_NAME` — vector database
- `SEARCH_MODE` ("dense" or "hybrid"), `SEARCH_SCORE_THRESHOLD`, `MAX_SEARCH_RESULTS`
- `API_BEARER_TOKEN` — required for all API endpoints except `/health`
- `UPLOAD_MAX_BYTES` (default 100MB), `WEB_MAX_CONTENT_BYTES` (default 2MB)

## Key Patterns

- **Provider detection**: Both embedding and LLM providers are auto-detected from the base URL (e.g., "generativelanguage.googleapis.com" → Gemini). See `get_embeddings()` in `src/embedding_manager.py` and `get_llm()` in `src/chat.py`.
- **Vector store initialization** (`src/vector_store.py`): Enforces strict dimension checking. Creates collections with both dense and sparse vector configs when in hybrid mode.
- **Pydantic models** for API request/response schemas live in `src/api_models.py`.
- **Security**: SSRF protection and path traversal validation in ingestion. Tests for these in `tests/test_ingestion_security.py`.
- **Ingest status tracking**: Background ingestion tasks expose status via `/api/v1/ingest/status`.
