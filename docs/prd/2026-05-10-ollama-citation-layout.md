## Problem Statement

Sebagai operator RAG backend, saat ini saya kehilangan akses Gemini embedding (subscription habis), sehingga harus migrasi ke Ollama bge-m3 yang sudah ter-host di server internal. Selain itu:

1. Saat ini ketika RAG menjawab pertanyaan, sumber yang ditampilkan hanya berupa path file (`"sources": ["/uploads/laporan.pdf"]`). Saya tidak tahu **halaman berapa** atau **bab mana** dari dokumen yang menjadi sumber jawaban — sulit untuk audit, verifikasi fakta, atau navigasi langsung ke bagian dokumen.

2. PDF dan DOCX dengan tabel di-chunk dengan `RecursiveCharacterTextSplitter(1000, 100)` yang generic. Tabel sering terpotong di tengah baris, header tabel hilang dari konteks chunk berikutnya, dan retrieval untuk pertanyaan "berapa nilai X di tabel Y" menjadi tidak akurat.

3. Heading dokumen (Bab, Pasal, Section) tidak dipakai sebagai sinyal retrieval, sehingga pertanyaan "apa isi Bab 2?" tidak match secara natural ke chunks yang berasal dari Bab 2.

## Solution

Migrasi backend RAG dengan tiga perubahan terkoordinasi yang akan di-deploy dalam satu PR (3 commit) ke main branch (production belum live, jadi cycle deploy aman):

1. **Swap embedding ke Ollama bge-m3** (1024-dim) menggantikan Gemini (768-dim). Tidak ada perubahan fungsionalitas — semua format dokumen yang sudah jalan (PDF text-based, DOCX, TXT, MD, CSV, code, web) tetap jalan. Multimodal/image support tetap **tidak ada** karena baik Gemini maupun bge-m3 sama-sama text-only di hilirnya, dan use-case operator hanya dokumen digital.

2. **Citation dengan page numbers + heading hierarchy** yang ter-struktur. Response `/chat` dan `/chat/stream` mengirim `sources` sebagai object array (breaking change dari `string[]`), berisi `display` label terformat (Bahasa Indonesia), `chunk_preview` (200 karakter), dan structured fields (`page`, `url_fragment`, `line_range`) untuk deep-linking. Frontend, Telegram bot, dan CLI semua di-update untuk render format baru.

3. **Layout-aware parsing untuk PDF dan DOCX** menggunakan `pdfplumber` + `python-docx` sebagai pengganti generic loader. Tabel jadi chunk terpisah dengan caption preserved (tabel besar di-split per row dengan header diulang). Heading detection: DOCX via style names, PDF via hybrid (font-size primary + Indonesian regex fallback untuk format BAB/Pasal/Bagian). Heading di-prepend ke chunk content untuk boost retrieval, plus disimpan di metadata untuk citation display. Fallback ke loader lama jika parser baru gagal.

## User Stories

1. As an RAG operator, I want migrasi embedding dari Gemini ke Ollama bge-m3, so that backend tetap jalan tanpa subscription Gemini yang sudah expired.

2. As an RAG operator, I want konfigurasi embedding via env vars (`EMBEDDER_MODEL=bge-m3:latest`, `EMBEDDER_DIMENSION=1024`), so that swap provider gak butuh code change.

3. As an RAG operator, I want collection Qdrant baru `dokumen_v2` (1024-dim), so that tidak konflik dengan dimension lama (768).

4. As an RAG operator, I want endpoint `POST /api/v1/files/reingest-all` tetap jalan, so that semua dokumen lama bisa di-reingest dengan parser + embedding baru.

5. As an end user via API, I want response `/chat` mengirim sources sebagai struct array (bukan string array), so that frontend bisa render rich UI dengan page navigation.

6. As an end user via API, I want setiap source menyertakan `locations` array, so that 1 file bisa berkontribusi banyak chunk dari halaman/bab berbeda dengan preview masing-masing.

7. As an end user via API, I want `display` label sudah di-format backend dalam Bahasa Indonesia, so that frontend cukup render string tanpa formatting logic.

8. As an end user via API, I want `chunk_preview` 200 karakter per location, so that saya tahu relevansi chunk tanpa download dokumen.

9. As an end user via API, I want structured fields (`page`, `url_fragment`, `line_range`), so that frontend bisa bikin deep link (`file.pdf#page=5`, `page.html#section`, `code.py#L45-L67`).

10. As an end user via API, I want endpoint `/chat/stream` (SSE) mengirim event `sources` dengan struct yang sama dengan `/chat`, so that streaming dan non-streaming consistent.

11. As an end user via Telegram bot, I want pesan jawaban menyertakan sumber dengan format yang readable di Telegram (file → list halaman dengan preview), so that saya tahu sumber jawaban tanpa buka aplikasi lain.

12. As an end user via CLI, I want output sumber menggunakan format Indonesian yang sama dengan API/Telegram, so that experience konsisten across interface.

13. As an RAG operator, I want PDF dengan tabel di-parse layout-aware, so that pertanyaan "berapa nilai X di tabel Y" bisa terjawab akurat.

14. As an RAG operator, I want tabel kecil (≤1000 chars) menjadi 1 chunk markdown utuh, so that struktur tabel preserved untuk embedding.

15. As an RAG operator, I want tabel besar di-split per row dengan header diulang, so that setiap chunk tetap punya konteks header tabel.

16. As an RAG operator, I want caption tabel disimpan di metadata `table_caption`, so that citation display bisa menunjukkan "Halaman 5 — Tabel: Anggaran 2024".

17. As an RAG operator, I want heading dokumen (BAB I, Pasal 2, dll) terdeteksi dari PDF Indonesian formal, so that hierarchy heading tampil di citation.

18. As an RAG operator, I want heading detection PDF pakai hybrid (font-size + regex Indonesian), so that dokumen yang formatting-nya beda (font weight only vs section numbering) tetap ter-detect.

19. As an RAG operator, I want heading dari DOCX ter-detect via Word style names ("Heading 1"), so that dokumen Word native ter-handle akurat.

20. As an RAG operator, I want heading_path di-prepend ke chunk content (`# Bab 2 / 2.1\n\n[content]`), so that BM25 dan dense retrieval bisa match keyword heading.

21. As an RAG operator, I want fallback ke loader lama (PyPDFLoader/Docx2txtLoader) kalau parser baru gagal, so that ingestion tidak pernah hard-fail karena parser bug.

22. As an RAG operator, I want metadata `parser_version` di setiap chunk, so that bisa monitor berapa % dokumen pakai parser baru vs fallback.

23. As an RAG operator, I want code parser (Tree-sitter `.py`/`.js`) tetap jalan tanpa perubahan, so that fitur code-aware chunking yang sudah ada tidak regress.

24. As an end user via API, I want existing endpoints (`/api/v1/ingest/web`, `/api/v1/ingest/file-path`, `/api/v1/ingest/uploads`, `/api/v1/files`, `/api/v1/files/upload`, `/api/v1/files/reingest-all`, `/health`, `/metrics`) tetap berfungsi, so that integrasi yang ada tidak break selain shape `sources` di `/chat`.

25. As an RAG operator, I want all changes coverage minimum 80% (95%+ untuk pure functions citation/chunker), so that regression risk minimum.

26. As an RAG operator, I want test fixtures (sample PDF/DOCX dengan tabel + heading) di repo, so that test deterministic dan bisa di-run di CI.

27. As an RAG operator, I want migration plan: deploy → reingest-all → verify, so that downtime minimal dan rollback path jelas.

## Implementation Decisions

### Modul baru

- **`src/citation.py`** — Pure module untuk format citation. Interface utama: `format_display(metadata, source_type) -> str`, `build_source_items(context_docs) -> list[SourceItem]`, `truncate_preview(text, max_chars) -> str`. Tidak ada I/O, mudah di-test tanpa mock.

- **`src/layout_parser.py`** — Modul parsing dan chunking layout-aware. Interface: `parse_pdf(filepath) -> list[Element]`, `parse_docx(filepath) -> list[Element]`, `chunk_elements(elements, max_size) -> list[Document]`, `detect_heading(text, font_size, body_size) -> tuple[bool, int]`. `Element` adalah frozen dataclass dengan kind (heading/paragraph/table/list_item), level, text, page, table_caption. Parser file melakukan I/O, tetapi `chunk_elements` adalah pure function — bisa di-test dengan synthetic Element list.

### Modul yang dimodifikasi

- **`src/ingestion.py`** — Refactor `load_local_document` agar SELALU return final chunks (bukan unsplit docs). Add `_legacy_load_and_split()` sebagai fallback path. `process_directory` tidak lagi memanggil `splitter.split_documents()` — chunks dari `load_local_document` di-extend langsung. Konsekuensi: code chunks via Tree-sitter tidak di-resplit lagi (sebelumnya re-split kalau >1000 chars).

- **`src/api_models.py`** — Schema baru: `LocationItem` (display, chunk_preview, page, url_fragment, line_range), `SourceItem` (source, source_type, filename, locations). `ChatResponse.sources` berubah dari `list[str]` jadi `list[SourceItem]` (breaking change).

- **`src/chat.py`** — `stream_chat_response` yield event `sources` dengan struct baru. Build dilakukan via `citation.build_source_items()`.

- **`api.py`** — Endpoint `/chat` build `SourceItem[]` dari `context_docs` via `citation` module.

- **`src/telegram_bot.py`** — Update render handler: dari flat list jadi format collapsible per file → list halaman dengan preview.

- **`main.py`** — Update CLI render serupa Telegram.

- **`src/config.py`** — Default `EMBEDDER_MODEL=bge-m3:latest`, `EMBEDDER_DIMENSION=1024`, `QDRANT_COLLECTION_NAME=dokumen_v2`. `.env.example` di-update juga.

### Schema metadata chunk baru

- `source` — path / URL (tetap)
- `source_type` — local / web / telegram_upload (tetap)
- `page` — 1-indexed page number (NEW, untuk PDF)
- `heading_path` — list[str] hierarchy (NEW)
- `table_caption` — caption tabel (NEW, untuk chunk tabel)
- `chunk_kind` — paragraph / table / list (NEW)
- `parser_version` — 2 untuk parser baru, 1 untuk legacy fallback (NEW)
- `ingested_at` — ISO timestamp (tetap)
- `file_hash` — SHA256 (tetap)

### API contract changes

**Breaking:** `ChatResponse.sources` shape:
- Sebelum: `list[str]` (flat path strings)
- Sesudah: `list[SourceItem]` dengan nested `locations[]`

**Breaking:** SSE event `sources` di `/chat/stream`:
- Sebelum: `{"type": "sources", "sources": ["path1", "path2"]}`
- Sesudah: `{"type": "sources", "sources": [{...SourceItem}]}`

**Non-breaking:** semua endpoint lain tidak berubah.

### Library additions

- `pdfplumber>=0.11` — PDF table + text extraction dengan font metadata
- `python-docx>=1.1` — DOCX paragraph + table + style introspection
- `docx2txt` tetap dipertahankan sebagai dependency `Docx2txtLoader` untuk fallback path

### Indonesian heading regex patterns (untuk PDF hybrid detection)

Pattern set yang dipakai (extended Indonesian formal):
- `^BAB\s+[IVXLCDM]+\b` (level 1) — BAB I, BAB II
- `^Bab\s+\d+` (level 1) — Bab 1
- `^BAGIAN\s+(?:KESATU|KEDUA|KETIGA|KEEMPAT|KELIMA)` (level 1)
- `^Bagian\s+\d+` (level 1)
- `^Lampiran\s+` (level 1)
- `^Pasal\s+\d+` (level 2)
- `^[IVXLCDM]+\.\s+[A-Z]` (level 2) — I. Pendahuluan
- `^\d+\.\d+\s` (level 2) — 1.1
- `^[A-Z]\.\s+[A-Z]` (level 3) — A. Latar belakang
- `^\d+\.\d+\.\d+\s` (level 3) — 1.1.1

### Display formatting per source_type (Bahasa Indonesia)

- **PDF**: `"Halaman {page} — {table_caption_prefix}{heading_path}"` dengan fallback hierarchy
- **DOCX**: `"{table_caption_prefix}{heading_path}"` dengan fallback `"Dokumen"`
- **Web**: `"Bagian: {heading}"` dengan fallback `"Halaman web"`
- **Code**: `"Function {name} (line X-Y)"` / `"Class {name}"` dengan fallback `"Kode"`
- **CSV**: `"Baris {row_number}"`
- **TXT/MD**: `"Bagian {chunk_index}"` atau heading-aware untuk MD

### Migration / deployment plan

1. Merge PR backend (3 commit) ke main
2. Update `.env` di Dokploy dengan nilai bge-m3 (1024-dim) dan collection name baru
3. Restart container backend
4. Trigger `POST /api/v1/files/reingest-all` (production belum live, low risk)
5. Verify dengan beberapa pertanyaan + check `/api/v1/files` untuk confirm metadata baru
6. Merge PR frontend (di repo `rag-qdrant-frontend`)
7. Trigger deploy frontend

### Commit structure dalam PR

- **Commit 1**: `chore(embedding): swap to Ollama bge-m3 (1024-dim)` — env defaults + tests
- **Commit 2**: `feat(citation): structured sources with page numbers + ID display` — citation module + API/SSE/Telegram/CLI updates + tests
- **Commit 3**: `feat(parser): layout-aware PDF/DOCX with table support` — layout_parser module + ingestion refactor + dependencies + tests

## Testing Decisions

### Filosofi test

Test eksternal behavior (input → output) untuk pure functions, dan smoke test untuk integration boundaries. Hindari test internal implementation (private helpers, intermediate state) supaya test gak rusak setiap kali refactor internal. Mock di boundary (Qdrant, LLM, network) bukan di internal function.

### Test scope per modul

- **`src/citation.py`** (HIGH priority, target 95%+ coverage): Pure function tests. Cover semua kombinasi metadata fields, fallback hierarchy, semua source_type, edge cases (empty heading_path, missing page, very long preview). No mocks needed.

- **`src/layout_parser.py` chunker** (HIGH priority, target 95%+ coverage): Pure function tests untuk `chunk_elements` dengan synthetic Element lists. Cover: tabel kecil tetap satu chunk, tabel besar split per row dengan header repeat, paragraph grouping sampai max_size, heading prepended ke content, page metadata preserved, mixed elements.

- **`src/layout_parser.py` parsers** (HIGH priority, target 90%+ coverage): Test dengan sample fixture files di `tests/fixtures/`. Cover: PDF dengan tabel + heading, DOCX dengan style heading + tabel, PDF dengan font-size heuristic, PDF dengan regex pattern heading Indonesian, fallback ketika file rusak.

- **`src/ingestion.py` refactored** (HIGH priority, target 85%+ coverage): Integration test untuk `load_local_document` dengan fixture files; test fallback path dengan file yang sengaja dibuat error; verify chunks final tidak double-split di `process_directory`.

- **`api.py /chat`** (MEDIUM priority, target 80%+): Integration test response shape baru (mock vector_store + LLM). Verify SourceItem fields, locations[], display formatting end-to-end.

- **`api.py /chat/stream`** (MEDIUM priority): Integration test SSE event shape baru.

- **`src/telegram_bot.py`** (LOW priority): Smoke test render output (snapshot test text format).

- **`main.py` CLI** (LOW priority): Smoke test CLI render output.

### Prior art

Existing tests di `tests/` directory:
- `tests/test_api.py` — pattern untuk integration test FastAPI endpoint
- `tests/test_ingestion_security.py` — pattern untuk SSRF/path traversal tests, bisa dipakai pola unit + fixture
- `tests/conftest.py` — fixtures untuk test setup

Pattern test baru akan mengikuti konvensi yang ada: pytest, `pytest.mark.unit` / `pytest.mark.integration`, fixture sample files di `tests/fixtures/`.

### Test fixture additions

`tests/fixtures/layout/` (NEW direktori):
- `simple_paragraph.pdf` — PDF tanpa heading/tabel
- `with_table.pdf` — PDF dengan 1 tabel medium
- `with_large_table.pdf` — PDF dengan tabel besar (>1000 chars)
- `formal_indonesian.pdf` — PDF dengan BAB I, Pasal X structure
- `font_size_headings.pdf` — PDF tanpa regex heading, hanya font-size
- `simple.docx` — DOCX dengan style headings + tabel
- `corrupt.pdf` — PDF rusak untuk test fallback

## Out of Scope

1. **OCR / scanned PDF support**: Confirmed dengan operator bahwa semua dokumen bersifat digital (text-based PDF, DOCX native). OCR pipeline (Tesseract / vision LLM) tidak dibangun.

2. **Image ingestion (.jpg, .png, .webp)**: `is_file_allowed` tetap reject tipe file gambar. Future enhancement jika butuh.

3. **Multi-knowledge-base / multi-tenant**: Tetap single global collection. Future enhancement.

4. **Knowledge graph / GraphRAG / entity extraction**: Future enhancement.

5. **Query rewriting / self-RAG / iterative retrieval**: Future enhancement.

6. **Agent framework (multi-step tools, function calling)**: Future enhancement.

7. **Page bbox highlighting (visual annotation pada PDF preview)**: `display` label cukup untuk MVP. Future enhancement kalau butuh.

8. **Heavy ML layout parser (`marker`, unstructured `hi_res` strategy)**: Tradeoff resource vs quality tidak worthwhile untuk dokumen digital sederhana. Tetap pakai `pdfplumber` light path.

9. **Backwards compatibility shim untuk `ChatResponse.sources`** (ngirim string[] AND object[]): Production belum live, breaking change diterima. Frontend di-update sekalian.

10. **Migration tool / migrate.py update**: Existing `migrate.py` tetap dipertahankan tapi tidak di-update karena migration path = drop+reingest.

## Further Notes

- Backend repo: `https://github.com/sizoune/universal-rag-qdrant`
- Frontend repo: lokal `~/Projects/Typescript/rag-qdrant-frontend` (perlu confirm GitHub remote-nya)
- Production deploy via Dokploy (https://peluncur.tabalongkab.go.id), Docker registry internal Tabalong
- Ollama server bge-m3:latest sudah di-host di `30.10.10.20:11434` (model sudah di-pull)
- Operator: muhammadwildaniskandar@gmail.com, context use case = dokumen pemerintah kabupaten Tabalong (peraturan, SK, laporan dinas — banyak format Indonesian formal yang relevan dengan regex pattern detection)
- Test coverage target overall: 80%+, namun pure function modules (citation, chunker) target 95%+
- Indonesian display: konsisten dengan UI Telegram bot existing (📚 Sumber prefix)
- Effort estimate: 1.5-2 minggu (3-4 hari per commit, plus frontend)
