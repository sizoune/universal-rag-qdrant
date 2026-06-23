# Spec: Web Search Fallback (9router) untuk RAG

- **Tanggal:** 2026-06-24
- **Status:** Disetujui (siap implementasi)
- **Cakupan:** REST API saja (`/api/v1/chat` + `/api/v1/chat/stream`)

## 1. Tujuan

Saat RAG tidak menemukan jawaban dari embedding yang ada, sistem melakukan
**web search via 9router** (`POST /v1/search`) lalu menjawab ulang berdasarkan
hasil web. Fitur **opt-in per request** (`enable_web_search`) dan punya
**kill-switch global** untuk admin.

## 2. Prinsip & blast radius

- Fitur **aditif & opt-in**. Jalur default (`enable_web_search=false`) **tidak
  berubah sama sekali** — perilaku byte-for-byte seperti sekarang.
- Logika sentinel + web hanya aktif bila request meminta **DAN** admin
  mengizinkan (`web_active = request.enable_web_search and WEB_SEARCH_ENABLED`).
- Biaya LLM hanya menjadi 2x **saat fallback benar-benar terpicu** (RAG tak
  punya jawaban). Jika RAG punya jawaban, tetap 1x panggilan.

## 3. Deteksi "RAG tidak punya jawaban" (sentinel)

Pendekatan: **LLM-judged** lewat sentinel, bukan pencocokan frasa bebas.

Saat web search aktif untuk sebuah request, prompt QA (tetap **satu system
message** demi kompatibilitas backend my-combo — lihat catatan di
`build_qa_prompt`) diinstruksikan: *jika jawaban tidak ada di konteks, balas
PERSIS `NO_ANSWER` dan tidak ada teks lain*; selain itu jawab normal.

- **Sync:** deteksi `answer.strip() == "NO_ANSWER"`.
- **Stream:** buffer output selama `accumulated.strip()` masih merupakan prefiks
  dari `"NO_ANSWER"`. Begitu menyimpang → flush buffer & stream normal. Jika
  stream selesai dan == `"NO_ANSWER"` → fallback. Ini mencegah sentinel bocor ke
  pengguna.

Sentinel const: `NO_ANSWER_SENTINEL = "NO_ANSWER"`.

## 4. Komponen

### 4.1 Modul baru `src/web_search.py`

Klien tipis untuk 9router `/v1/search`. Tanpa state, config-driven.

```python
@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str
    score: float | None
```

- `search_web(query: str) -> list[WebResult]`
  - `POST {WEB_SEARCH_URL}` dengan body `{"model": WEB_SEARCH_PROVIDER,
    "query": query, "max_results": WEB_SEARCH_MAX_RESULTS}` dan header
    `Authorization: Bearer {WEB_SEARCH_API_KEY}` (dilewati bila key kosong),
    `Content-Type: application/json`, `timeout=WEB_SEARCH_TIMEOUT`.
  - Parse `results[]` → `WebResult` (title, url, snippet, score).
  - **Gagal/timeout/non-200/`errors[]` tidak kosong → return `[]`** dan log
    warning. Fallback yang gagal cukup berujung pesan "tidak ditemukan", tidak
    melempar exception ke pengguna.
- `web_results_to_documents(results: list[WebResult]) -> list[Document]`
  - Map ke LangChain `Document`: `page_content=snippet`, `metadata={"source":
    url, "source_type": "web", "title": title}`. Mengalir apa adanya ke
    `build_source_items` (yang sudah mendukung `source_type="web"`).
- `web_context_text(results: list[WebResult]) -> str`
  - Blok teks konteks untuk LLM call #2 (judul + url + snippet per hasil).

### 4.2 Orkestrasi (`src/chat.py`)

- Konstanta `NO_ANSWER_SENTINEL` dan helper prompt varian-sentinel (satu system
  message, memuat instruksi sentinel + `{context}` + `{extra_system}` +
  `{date_guidance}`).
- Helper sync baru:
  `answer_with_web_fallback(question, history, vector_store, extra_system, web_active) -> (answer: str, sources: list[SourceItem], web_used: bool)`
  1. Retrieve (history-aware) → LLM call #1 (prompt sentinel).
  2. Jika sentinel **dan** `web_active`: `search_web(question)`.
     - Ada hasil → LLM call #2 dengan konteks web (prompt normal) →
       `web_used=True`, `sources=build_source_items(web_docs)`.
     - Tak ada hasil → pesan ramah "tidak ditemukan di dokumen maupun web",
       `web_used=False`.
  3. Jika sentinel **dan** tidak `web_active`: pesan ramah "tidak ditemukan di
     dokumen", `web_used=False`.
  4. Selain itu: jawaban RAG normal, `sources` dari `context_docs`,
     `web_used=False`.
  - `ponytail:` query web memakai `question` mentah — follow-up anaforik bisa
    kurang presisi; upgrade = condense memakai history (3 panggilan LLM).
- `stream_chat_response(..., enable_web_search: bool = False)`:
  - Tidak aktif → persis perilaku sekarang (tanpa sentinel).
  - Aktif → call #1 streamed dengan prompt sentinel + buffered-sentinel
    suppression; bila sentinel terdeteksi → `search_web` → call #2 streamed atas
    konteks web. Emit hasil web sebagai event `sources`.

### 4.3 Konfigurasi (`src/config.py` + `.env.example`)

| Env | Default | Fungsi |
|---|---|---|
| `WEB_SEARCH_ENABLED` | `false` | kill-switch global admin |
| `WEB_SEARCH_URL` | *(derive `LLM_BASE_URL.rstrip('/') + "/search"`)* | endpoint POST 9router |
| `WEB_SEARCH_API_KEY` | *(fallback `LLM_API_KEY`)* | Bearer token |
| `WEB_SEARCH_PROVIDER` | `search-combo` | field `model`/`provider` |
| `WEB_SEARCH_MAX_RESULTS` | `5` | jumlah hasil |
| `WEB_SEARCH_TIMEOUT` | `10` | timeout detik |

`WEB_SEARCH_ENABLED` di-parse boolean (`"true"/"1"/"yes"` → True). `WEB_SEARCH_URL`
kosong → derive dari `LLM_BASE_URL`; bila `LLM_BASE_URL` juga kosong → tetap
kosong (search efektif dimatikan, `search_web` return `[]`).

### 4.4 API (`api.py`, `src/api_models.py`)

- `ChatRequest` + `enable_web_search: bool = False`.
- `ChatResponse` + `web_search_used: bool = False`.
- `/chat`: hitung `web_active = payload.enable_web_search and config.WEB_SEARCH_ENABLED`.
  - `web_active` true → `answer_with_web_fallback(...)`.
  - `web_active` false → jalur chain lama **tidak tersentuh**.
- `/chat/stream`: teruskan `web_active` ke `stream_chat_response`; tambah event
  `{"type": "web_search", "used": bool}` sebelum event `sources`.
- Global OFF saat request minta web → param diabaikan diam-diam,
  `web_search_used=false`.

## 5. Pengujian

- `tests/test_web_search.py` — mock `requests.post`: parse sukses → `WebResult`;
  timeout/`errors[]`/non-200 → `[]`; mapping `web_results_to_documents` punya
  metadata `source_type="web"`.
- `tests/test_chat_web_fallback.py` — mock retriever + LLM (kasus sentinel vs
  jawaban) + `search_web`: fallback hanya terpicu saat sentinel; `web_used`
  benar; `sources` berisi sumber web saat fallback.
- `tests/test_api.py` (tambahan) — `enable_web_search=true` (dependensi
  di-mock) menghasilkan `web_search_used=true`; kasus `WEB_SEARCH_ENABLED=false`
  → param diabaikan, `web_search_used=false`.
- `requirements.txt` — pin `requests` eksplisit (sudah dipakai
  `src/ingestion.py`, kini dependency kelas satu).

## 6. Di luar cakupan (YAGNI)

Caching hasil search; dedup web vs RAG; condense query untuk web search;
toggle/command di Telegram & CLI; streaming-merge gabungan RAG+web dalam satu
jawaban.
