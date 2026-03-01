# Universal RAG System

Universal RAG adalah sistem Retrieval-Augmented Generation berbasis Qdrant + LangChain untuk ingest dokumen/URL lalu tanya-jawab dengan LLM pilihan (OpenAI, Gemini, Ollama, atau OpenAI-compatible endpoint).

## Dokumentasi

- [Arsitektur RAG](docs/architecture.md)
- [Advanced RAG & Deployment](docs/advanced-rag-deployment.md)
- [Token Usage](docs/token-usage.md)
- [Telegram Bot Gateway](docs/telegram-bot.md)

## Fitur Utama

- Multi-provider embedding dan chat model
- Strict dimension checking untuk keamanan perubahan model embedding
- Smart ingestion untuk web/file + dedup berbasis hash
- Hybrid search mode (dense + sparse) dengan fallback aman
- Optional cross-encoder reranking
- Telegram bot gateway
- Docker deployment (Qdrant + bot)

## Struktur Proyek

```text
rag-project/
|-- docs/
|   |-- architecture.md
|   |-- advanced-rag-deployment.md
|   |-- token-usage.md
|   `-- telegram-bot.md
|-- src/
|   |-- chat.py
|   |-- config.py
|   |-- hybrid_retriever.py
|   |-- ingestion.py
|   |-- reranker.py
|   |-- sparse_encoder.py
|   `-- vector_store.py
|-- tests/
|-- Dockerfile
|-- docker-compose.yml
|-- main.py
`-- requirements.txt
```

## Setup Local

### 1. Buat virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Siapkan environment

Copy `.env.example` ke `.env`, lalu isi variabel utama:

```env
# Embedding
EMBEDDER_BASE_URL="https://api.openai.com/v1"
EMBEDDER_API_KEY="your-api-key"
EMBEDDER_MODEL="text-embedding-3-small"
EMBEDDER_DIMENSION=1536

# Qdrant
QDRANT_URL="http://localhost:6333"
QDRANT_COLLECTION_NAME="universal_rag_collection"

# LLM
LLM_BASE_URL="https://api.openai.com/v1"
LLM_API_KEY="your-api-key"
LLM_MODEL="gpt-3.5-turbo"

# Advanced RAG
SEARCH_MODE="dense"              # dense | hybrid
RERANKER_ENABLED=false
RERANKER_MODEL="Xenova/ms-marco-MiniLM-L-6-v2"
```

### 3. Jalankan perintah

```bash
# status
venv\Scripts\python main.py status

# ingest web
venv\Scripts\python main.py ingest-web https://id.wikipedia.org/wiki/SaaS

# ingest folder/file
venv\Scripts\python main.py ingest-file ./documents

# chat interaktif
venv\Scripts\python main.py chat

# single-shot question
venv\Scripts\python main.py chat "apa itu SaaS?"

# telegram gateway
venv\Scripts\python main.py gateway
```

## Advanced RAG (Ringkas)

- `SEARCH_MODE="dense"`: retrieval semantic biasa
- `SEARCH_MODE="hybrid"`: dense + sparse/BM25 retrieval
- `RERANKER_ENABLED=true`: aktifkan cross-encoder reranking setelah retrieval

Penjelasan lengkap ada di [docs/advanced-rag-deployment.md](docs/advanced-rag-deployment.md).

## Docker Deployment

```bash
docker compose up --build
```

Service default:
- `qdrant` (port 6333)
- `rag-bot` (menjalankan `python main.py gateway`)

## Testing

```bash
venv\Scripts\python -m pytest tests/
```

## Catatan

- Jika mengganti model embedding dengan dimensi berbeda, lakukan re-index/clear collection.
- Untuk mode hybrid, disarankan ingest ulang agar sparse vector tersedia untuk semua chunk.
