# Advanced RAG & Deployment

Dokumen ini menjelaskan implementasi fase 3 pada project ini:

1. Hybrid Search (Dense + Sparse vectors)
2. Optional Re-ranking (Cross-Encoder)
3. Docker deployment (Qdrant + bot)

## 1. Tujuan Fase 3

Dense embedding bagus untuk semantic similarity, tetapi lemah untuk exact keyword seperti kode error, akronim, atau nama produk. Fase ini menambahkan jalur sparse BM25 agar pencarian lebih seimbang.

Target akhir:
- Dense-only tetap didukung
- Hybrid bisa diaktifkan via config tanpa ubah kode
- Hasil retrieval bisa ditingkatkan dengan reranker
- Deployment lokal via Docker Compose

## 2. Arsitektur Ringkas

Alur query saat `SEARCH_MODE=hybrid`:

1. Query di-embed jadi dense vector
2. Query juga di-encode jadi sparse BM25 vector
3. Qdrant mengambil kandidat dense + sparse
4. Hasil difusi/fusion diambil sebagai candidate docs
5. Jika reranker aktif, candidate diurutkan ulang oleh cross-encoder
6. Top document dikirim ke chain LLM

Alur ingest dokumen:

1. Dokumen di-chunk
2. Setiap chunk menghasilkan dense + sparse vector
3. Chunk di-upsert ke Qdrant sebagai named vectors

## 3. Komponen Utama

### 3.1 Vector Store

File: `src/vector_store.py`

Fungsi utama:
- Inisialisasi Qdrant collection
- Strict dimension checking untuk dense vector
- Kompatibel dengan collection lama (unnamed dense vector) dan collection baru (named `dense`)
- Ingest custom yang menulis dense + sparse vector dalam satu point

Catatan mode collection:
- `SEARCH_MODE=hybrid` -> create collection with named dense vector + sparse vectors config
- mode lain -> create collection dense biasa (unnamed)

### 3.2 Sparse Encoder

File: `src/sparse_encoder.py`

Fungsi:
- BM25 sparse encoding dengan `fastembed.SparseTextEmbedding`
- Model default: `Qdrant/bm25`
- Output: list dictionary `{indices, values}` yang kompatibel dengan Qdrant sparse vector

### 3.3 Hybrid Retriever

File: `src/hybrid_retriever.py`

Fungsi:
- Menjalankan retrieval hybrid saat mode hybrid aktif
- Fallback ke dense retrieval saat hybrid gagal
- Menyaring hasil dengan `score_threshold`
- Menjalankan re-ranking jika diaktifkan

### 3.4 Reranker

File: `src/reranker.py`

Fungsi:
- Optional cross-encoder re-ranking
- Model dikontrol env `RERANKER_MODEL`
- Aktif jika `RERANKER_ENABLED=true`

## 4. Konfigurasi .env

Tambahkan atau pastikan variabel berikut ada:

```env
# Search mode
SEARCH_MODE="dense"              # dense | hybrid

# Retrieval tuning
SEARCH_SCORE_THRESHOLD=0.7
MAX_SEARCH_RESULTS=4

# Reranker
RERANKER_ENABLED=false
RERANKER_MODEL="Xenova/ms-marco-MiniLM-L-6-v2"
```

Rekomendasi awal:
- Mulai dengan `SEARCH_MODE="dense"` saat bootstrap
- Naik ke `SEARCH_MODE="hybrid"` setelah koleksi siap sparse vectors
- Aktifkan reranker hanya jika kualitas retrieval masih kurang

## 5. Cara Menjalankan (Local)

Gunakan virtual environment project.

```bash
# Windows
venv\Scripts\python main.py status

# Ingest URL
venv\Scripts\python main.py ingest-web https://id.wikipedia.org/wiki/SaaS

# Chat
venv\Scripts\python main.py chat "apa itu SaaS?"
```

## 6. Docker Deployment

File:
- `Dockerfile`
- `docker-compose.yml`

Jalankan:

```bash
docker compose up --build
```

Service:
- `qdrant` di `http://localhost:6333`
- `rag-bot` menjalankan `python main.py gateway`

Tips:
- Pastikan `.env` terisi sebelum `docker compose up`
- `QDRANT_URL` untuk bot di dalam Docker harus mengarah ke service `qdrant` (bukan localhost host)

## 7. Verification Checklist

1. Validasi test:
```bash
venv\Scripts\python -m pytest tests/
```

2. Validasi mode dense:
- Set `SEARCH_MODE="dense"`
- Jalankan ingest + chat
- Pastikan jawaban basic semantic berjalan

3. Validasi mode hybrid:
- Set `SEARCH_MODE="hybrid"`
- Re-ingest sumber data
- Uji query keyword exact (mis. kode error)

4. Validasi reranker:
- Set `RERANKER_ENABLED=true`
- Bandingkan relevansi top context terhadap mode tanpa reranker

5. Validasi deployment:
- `docker compose up --build`
- Cek log startup qdrant dan rag-bot

## 8. Troubleshooting

### Error: dimension mismatch

Penyebab:
- `EMBEDDER_DIMENSION` tidak cocok dengan dimensi koleksi yang sudah ada

Solusi:
- Samakan env dengan model embedding yang dipakai
- Atau lakukan clear/re-index (`main.py clear`) jika migrasi model

### Hybrid tidak terasa beda dengan dense

Periksa:
- `SEARCH_MODE` sudah `hybrid`
- Collection sudah dibuat dengan sparse vectors support
- Dokumen sudah di-ingest ulang setelah mode hybrid

### Startup Docker gagal konek Qdrant

Periksa:
- Service `qdrant` running
- `QDRANT_URL` dalam container bot mengarah ke `http://qdrant:6333`

## 9. Referensi File Implementasi

- `src/vector_store.py`
- `src/sparse_encoder.py`
- `src/hybrid_retriever.py`
- `src/reranker.py`
- `src/chat.py`
- `Dockerfile`
- `docker-compose.yml`
