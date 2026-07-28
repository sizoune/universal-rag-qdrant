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
- Web search fallback via 9Router (opt-in per request) saat RAG tak menemukan jawaban
- Telegram bot gateway
- FastAPI HTTP API + CRUD dokumen terindeks
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
|-- api.py
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

# Optional: folder khusus untuk ingest upload
UPLOADS_DIR="uploads"
INGEST_BASE_DIR="uploads"
UPLOAD_MAX_BYTES=104857600
WEB_MAX_CONTENT_BYTES=2097152

# LLM
LLM_BASE_URL="https://api.openai.com/v1"
LLM_API_KEY="your-api-key"
LLM_MODEL="gpt-3.5-turbo"

# Advanced RAG
SEARCH_MODE="dense"              # dense | hybrid
RERANKER_ENABLED=false
RERANKER_MODEL="Xenova/ms-marco-MiniLM-L-6-v2"

# API
API_BEARER_TOKEN="change-me"
API_HOST="0.0.0.0"
API_PORT=8000
API_CORS_ORIGINS="*"

# Web Search Fallback (9Router) — lihat seksi khusus di bawah
WEB_SEARCH_ENABLED=false          # kill-switch global; tetap perlu enable_web_search=true per request
WEB_SEARCH_URL=""                 # kosong -> derive LLM_BASE_URL + "/search"
WEB_SEARCH_API_KEY=""             # kosong -> fallback ke LLM_API_KEY
WEB_SEARCH_PROVIDER="exa"         # neural search, sumber terkini; butuh FETCH_CONTENT=true
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_TIMEOUT=10
WEB_SEARCH_FETCH_CONTENT="true"   # fetch isi halaman penuh (wajib utk exa, snippet kosong)
```

### 3. Jalankan perintah

```bash
# status
venv\Scripts\python main.py status

# ingest web
venv\Scripts\python main.py ingest-web https://id.wikipedia.org/wiki/SaaS

# ingest folder/file
venv\Scripts\python main.py ingest-file ./documents

# ingest dari folder upload (default: ./uploads atau env UPLOADS_DIR)
venv\Scripts\python main.py ingest-uploads

# chat interaktif
venv\Scripts\python main.py chat

# single-shot question
venv\Scripts\python main.py chat "apa itu SaaS?"

# telegram gateway
venv\Scripts\python main.py gateway

# API server (frontend integration)
venv\Scripts\python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

## API Integration (Frontend)

Semua endpoint API berada di prefix `/api/v1/*` dan butuh header:

```http
Authorization: Bearer <API_BEARER_TOKEN>
```

OpenAPI docs:

```text
http://localhost:8000/docs
```

Endpoint utama:
- `GET /health` (tanpa auth)
- `GET /api/v1/status`
- `POST /api/v1/chat`
- `POST /api/v1/chat/stream` (SSE streaming)
- `POST /api/v1/ingest/web`
- `POST /api/v1/ingest/file-path`
- `POST /api/v1/ingest/uploads`
- `GET /api/v1/files`
- `GET /api/v1/files/{source_id}`
- `POST /api/v1/files/upload`
- `PUT /api/v1/files/{source_id}`
- `POST /api/v1/files/reingest-all`
- `DELETE /api/v1/files/{source_id}`

Contoh request chat:

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"question":"apa itu SaaS?","session_id":"fe-user-1"}'
```

Field `system_prompt` (opsional, maks. 8000 karakter) menambahkan instruksi
sistem per-request — ditambahkan ke prompt dasar, bukan menggantinya. Berlaku
untuk `POST /api/v1/chat` dan `POST /api/v1/chat/stream`:

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"question":"apa itu SaaS?","system_prompt":"Jawab singkat dalam 1 kalimat."}'
```

## Web Search Fallback (9Router)

Saat RAG tidak menemukan jawaban di dokumen terindeks, sistem dapat fallback ke
**web search lewat 9Router** (`POST /v1/search`) lalu menjawab ulang dari hasil
web. Fitur ini **opt-in per request** dan punya **kill-switch global**.

**Cara kerja.** LLM mencoba menjawab dari konteks RAG terlebih dulu. Bila model
menilai jawaban tidak ada di konteks (mengeluarkan sentinel `NO_ANSWER`), barulah
web search dipanggil dan LLM menjawab ulang dari hasil web. Jadi biaya LLM hanya
menjadi 2x saat fallback benar-benar terpicu — bila RAG sudah bisa menjawab, tetap
1x panggilan. Hasil web muncul sebagai sumber `source_type="web"` lewat pipeline
sitasi yang sama. Pada streaming, sentinel ditahan (buffered) agar tidak bocor ke
klien.

**Aktivasi.** Butuh dua hal:
1. Global: `WEB_SEARCH_ENABLED=true` (default `false`).
2. Per request: kirim `"enable_web_search": true`. Jika global `false`, flag
   request diabaikan diam-diam (`web_search_used=false`).

Respons `POST /api/v1/chat` menambah field `web_search_used: bool`. Pada
`POST /api/v1/chat/stream` ada event SSE `{"type":"web_search","used":<bool>}`
sebelum event `sources`.

**Konfigurasi (env).**

| Env | Default | Fungsi |
|---|---|---|
| `WEB_SEARCH_ENABLED` | `false` | kill-switch global admin |
| `WEB_SEARCH_URL` | *(derive `LLM_BASE_URL + "/search"`)* | endpoint POST 9Router |
| `WEB_SEARCH_API_KEY` | *(fallback `LLM_API_KEY`)* | Bearer token |
| `WEB_SEARCH_PROVIDER` | `exa` | field `model`/`provider`. **`exa`** (neural search) paling jitu menemukan sumber terkini → butuh `WEB_SEARCH_FETCH_CONTENT=true` (snippet exa kosong). `tavily` mengisi snippet sendiri tapi sering menarik artikel usang. **Hindari `search-combo`** |
| `WEB_SEARCH_MAX_RESULTS` | `5` | jumlah hasil |
| `WEB_SEARCH_TIMEOUT` | `10` | timeout detik |
| `WEB_SEARCH_FETCH_CONTENT` | `true` | ambil isi halaman penuh top-3 (via `/v1/web/fetch`) → konteks bertanggal, LLM tak mengutip snippet usang. Wajib untuk `exa` |

> **Catatan URL.** Karena LLM biasanya sudah lewat 9Router, `WEB_SEARCH_URL`
> dibiarkan kosong dan diturunkan otomatis dari `LLM_BASE_URL + "/search"`. Ini
> **hanya benar bila `LLM_BASE_URL` berakhiran `/v1`** (mis. `http://host:port/v1`
> → `http://host:port/v1/search`). Bila berbeda, set `WEB_SEARCH_URL` eksplisit ke
> URL `/v1/search` penuh. Pastikan juga `WEB_SEARCH_PROVIDER` benar-benar ada di
> instance 9Router-mu: `curl $NINEROUTER_URL/v1/models/web`.

Contoh request dengan fallback aktif:

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"question":"siapa presiden Prancis saat ini?","enable_web_search":true}'
# -> {"answer":"...","web_search_used":true, "sources":[{"source_type":"web",...}], ...}
```

> Fallback hanya terpicu saat RAG tidak punya jawaban. Untuk pertanyaan yang
> jawabannya ada di dokumen, `web_search_used` tetap `false` meski toggle aktif.

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
- `qdrant` (internal; shared by all API instances)
- `rag-bot` (menjalankan `python main.py gateway`)
- `rag-api` — retrieve / chat / status (host `13121` → container `8000`)
- `rag-api-ingest` — ingest / file writes (host `13123` → container `8000`)

Routing yang disarankan saat indexing berat (mis. n8n reconcile) berjalan bersamaan dengan Q&A:
- Klien baca (PPID Tanya Dokumen, UI chat): `http://<host>:13121`
- Klien tulis (n8n ingest, admin reindex): `http://<host>:13123`

Keduanya memakai Qdrant + volume cache/uploads yang sama, jadi indeks tetap satu; proses HTTP-nya terpisah agar retrieve tidak timeout saat ingest.

## Testing

```bash
venv\Scripts\python -m pytest tests/
```

## Catatan

- Jika mengganti model embedding dengan dimensi berbeda, lakukan re-index/clear collection.
- Untuk mode hybrid, disarankan ingest ulang agar sparse vector tersedia untuk semua chunk.
- Endpoint `POST /api/v1/ingest/file-path` dibatasi hanya untuk path di dalam `INGEST_BASE_DIR`.
- Upload API dibatasi oleh `UPLOAD_MAX_BYTES`, dan ingest web dibatasi oleh `WEB_MAX_CONTENT_BYTES`.
