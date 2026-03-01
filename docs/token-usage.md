# Penggunaan Token & Dampak Biaya

## Apa Itu Token?

Token adalah satuan terkecil yang diproses oleh model AI. Secara kasar:
- **1 token ≈ 4 karakter** (bahasa Inggris)
- **1 token ≈ 2-3 karakter** (bahasa Indonesia, karena karakter non-Latin lebih mahal)
- **1000 token ≈ 750 kata** (bahasa Inggris)

## Di Mana Token Digunakan?

Sistem RAG menggunakan token di **2 tahap** yang berbeda:

### 1. Embedding (Ingestion & Search)

| Kapan | Apa yang terjadi | Token terpakai |
|-------|------------------|----------------|
| **Ingestion** | Setiap chunk teks dikonversi ke vektor | ±250 token/chunk |
| **Search** | Pertanyaan user dikonversi ke vektor | ±10-50 token/pertanyaan |

**Contoh perhitungan ingestion:**
```
Artikel Wikipedia ~5000 karakter
→ Dipotong jadi 6 chunk × ~250 token = 1.500 token embedding

Folder 100 file PDF (~200 halaman total)
→ ~2000 chunk × ~250 token = 500.000 token embedding
```

### Contoh Lengkap: Dari Qdrant ke LLM dan Kembali ke User

Berikut simulasi nyata ketika user bertanya **"apa itu SaaS?"** setelah meng-ingest halaman Wikipedia.

#### Langkah 1 — Data yang Tersimpan di Qdrant

Setelah `ingest-web`, Qdrant menyimpan chunk-chunk seperti ini:

```json
// Point #1 di collection "universal_rag_collection"
{
  "id": "a3f8c1d2-...",
  "vector": [0.023, -0.156, 0.891, ..., 0.044],   // 768 float (dari nomic-embed-text)
  "payload": {
    "page_content": "Perangkat lunak sebagai layanan (bahasa Inggris: Software as a service, disingkat SaaS) adalah model lisensi dan pengiriman perangkat lunak di mana perangkat lunak dilisensikan secara berlangganan dan di-hosting secara terpusat. SaaS juga dikenal sebagai perangkat lunak berbasis web.",
    "metadata": {
      "source": "https://id.wikipedia.org/wiki/Perangkat_lunak_sebagai_layanan",
      "source_type": "web"
    }
  }
}

// Point #2
{
  "id": "b7e4d5f1-...",
  "vector": [0.041, -0.203, 0.712, ..., 0.089],
  "payload": {
    "page_content": "Keuntungan SaaS meliputi: aksesibilitas dari perangkat apa pun, biaya awal yang rendah, skalabilitas, pembaruan otomatis, dan integrasi yang mudah. Pengguna cukup mengakses aplikasi melalui browser web tanpa perlu menginstal perangkat lunak secara lokal.",
    "metadata": {
      "source": "https://id.wikipedia.org/wiki/Perangkat_lunak_sebagai_layanan",
      "source_type": "web"
    }
  }
}

// Point #3, #4, ... (chunk lainnya)
```

#### Langkah 2 — User Bertanya, Qdrant Mencari

```
User input: "apa itu SaaS?"
```

Pertanyaan di-embed menjadi vektor `[0.019, -0.148, 0.903, ...]`, lalu Qdrant menghitung **cosine similarity**:

```
POST http://localhost:6333/collections/universal_rag_collection/points/query

Hasil pencarian Qdrant:
┌───────┬──────────────────────────────────────────────────────┬───────┐
│ Rank  │ page_content (diringkas)                            │ Score │
├───────┼──────────────────────────────────────────────────────┼───────┤
│  #1   │ "Perangkat lunak sebagai layanan (SaaS) adalah..."  │ 0.92  │
│  #2   │ "Keuntungan SaaS meliputi: aksesibilitas..."        │ 0.87  │
│  #3   │ "Contoh penyedia SaaS: Google Workspace, Salesf..." │ 0.84  │
│  #4   │ "Perbedaan SaaS dengan PaaS dan IaaS terletak..."  │ 0.79  │
│  ──   │ "Kategori: Komputasi awan | Teknologi"              │ 0.41  │ ← DITOLAK (< 0.7)
└───────┴──────────────────────────────────────────────────────┴───────┘
```

> Chunk dengan skor < `SEARCH_SCORE_THRESHOLD` (0.7) otomatis dibuang.

#### Langkah 3 — Payload yang Dikirim ke LLM API

Sistem menggabungkan **System Prompt + Context dari Qdrant + Chat History + Pertanyaan** menjadi satu request:

```json
// POST http://localhost:11434/api/chat  (Ollama)
// atau POST https://api.openai.com/v1/chat/completions  (OpenAI)
{
  "model": "llama3",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant connected to a knowledge base.\nUse the following pieces of retrieved context to answer the user's question.\nIf the answer is not in the context, just say that you don't know based on the provided documents.\nDo not make up information that isn't supported by the context.\n\nContext:\nPerangkat lunak sebagai layanan (bahasa Inggris: Software as a service, disingkat SaaS) adalah model lisensi dan pengiriman perangkat lunak di mana perangkat lunak dilisensikan secara berlangganan dan di-hosting secara terpusat. SaaS juga dikenal sebagai perangkat lunak berbasis web.\n\nKeuntungan SaaS meliputi: aksesibilitas dari perangkat apa pun, biaya awal yang rendah, skalabilitas, pembaruan otomatis, dan integrasi yang mudah. Pengguna cukup mengakses aplikasi melalui browser web tanpa perlu menginstal perangkat lunak secara lokal.\n\nContoh penyedia SaaS: Google Workspace, Salesforce, Microsoft 365, Dropbox, dan Slack.\n\nPerbedaan SaaS dengan PaaS dan IaaS terletak pada tingkat abstraksi layanan yang diberikan kepada pengguna."
    },
    {
      "role": "user",
      "content": "apa itu SaaS?"
    }
  ]
}
```

**Perhitungan token request ini:**
```
System prompt (instruksi)     :  ~100 token
Context (4 chunk dari Qdrant) :  ~800 token
Chat history                  :    0 token  (percakapan baru)
Pertanyaan user               :   ~8 token
──────────────────────────────────────────
Total INPUT                   :  ~908 token
```

#### Langkah 4 — Respons dari LLM

```json
// Response dari Ollama/OpenAI
{
  "message": {
    "role": "assistant",
    "content": "Berdasarkan konteks yang diberikan, SaaS (Software as a Service) atau Perangkat Lunak sebagai Layanan adalah model lisensi dan pengiriman perangkat lunak di mana perangkat lunak dilisensikan secara berlangganan dan di-hosting secara terpusat, juga dikenal sebagai perangkat lunak berbasis web.\n\nKeuntungan SaaS meliputi:\n- Aksesibilitas dari perangkat apa pun\n- Biaya awal yang rendah\n- Skalabilitas\n- Pembaruan otomatis\n- Integrasi yang mudah\n\nContoh penyedia SaaS terkenal antara lain Google Workspace, Salesforce, Microsoft 365, Dropbox, dan Slack."
  }
}
```

**Perhitungan token respons:**
```
Jawaban AI (output)           :  ~150 token
──────────────────────────────────────────
Total OUTPUT                  :  ~150 token
```

#### Langkah 5 — Ditampilkan ke User

```
AI: Berdasarkan konteks yang diberikan, SaaS (Software as a Service)
    atau Perangkat Lunak sebagai Layanan adalah model lisensi dan
    pengiriman perangkat lunak di mana perangkat lunak dilisensikan
    secara berlangganan dan di-hosting secara terpusat...

[Sources Used]:
  1. https://id.wikipedia.org/wiki/Perangkat_lunak_sebagai_layanan
  2. https://id.wikipedia.org/wiki/Perangkat_lunak_sebagai_layanan
```

#### Ringkasan Token untuk 1 Pertanyaan Ini

| Tahap | Proses | Token | Biaya (Ollama) | Biaya (OpenAI gpt-4o-mini) |
|-------|--------|-------|----------------|---------------------------|
| Search | Embed pertanyaan | ~8 | Gratis | $0.000000016 |
| Input | System + Context + Question | ~908 | Gratis | $0.000136 |
| Output | Jawaban AI | ~150 | Gratis | $0.000090 |
| **Total** | | **~1.066** | **Rp 0** | **~$0.0002 (~Rp 3)** |


### 2. LLM Chat (Generation)

| Komponen | Token terpakai | Keterangan |
|----------|----------------|------------|
| System prompt | ~100 token | Instruksi ke AI (tetap) |
| Context (top-K chunks) | ~1000-4000 token | Dari Qdrant retrieval |
| Chat history | ~200-2000 token | Tergantung MEMORY_WINDOW_SIZE |
| User question | ~10-50 token | Pertanyaan user |
| AI answer | ~100-500 token | Jawaban yang dihasilkan |
| **Total per chat turn** | **~1.400 - 6.600 token** | |

## Perbandingan Biaya Provider

### Provider Embedding

| Provider | Model | Dimensi | Harga per 1M token | 10.000 chunk |
|----------|-------|---------|--------------------|--------------| 
| **Ollama** | nomic-embed-text | 768 | **Gratis** (lokal) | **$0** |
| OpenAI | text-embedding-3-small | 1536 | $0.020 | ~$0.05 |
| OpenAI | text-embedding-3-large | 3072 | $0.130 | ~$0.33 |
| Google | embedding-001 | 768 | Gratis (batas harian) | **$0** |

### Provider LLM Chat

| Provider | Model | Harga Input/1M | Harga Output/1M | 100 chat turns |
|----------|-------|----------------|-----------------|----------------|
| **Ollama** | llama3 (8B) | **Gratis** (lokal) | **Gratis** | **$0** |
| **Ollama** | qwen3 (8B) | **Gratis** (lokal) | **Gratis** | **$0** |
| OpenAI | gpt-3.5-turbo | $0.50 | $1.50 | ~$0.30 |
| OpenAI | gpt-4o-mini | $0.15 | $0.60 | ~$0.10 |
| OpenAI | gpt-4o | $2.50 | $10.00 | ~$3.00 |
| Google | gemini-1.5-flash | $0.075 | $0.30 | ~$0.05 |
| Google | gemini-1.5-pro | $1.25 | $5.00 | ~$1.50 |

> **Catatan**: Harga dapat berubah. Cek website resmi masing-masing provider.

## Strategi Menghemat Token

### 1. Hash-Based Caching + Persistent Cache (Sudah Built-in)
```
File A.pdf → SHA-256 hash → sudah ada di cache?
  Ya  → SKIP (0 token terpakai)
  Tidak → proses embedding (250 token/chunk)

Web URL → SHA-256(konten) → sudah ada di cache?
  Ya  → SKIP (0 token terpakai)
  Tidak → scrape → embed → simpan hash
```
Cache disimpan di `.cache/ingestion_hashes.json` — **persisten**, tidak hilang saat restart.

### 2. Deduplication (Sudah Built-in)
```
Ingest URL yang sama 2x?
  → Sistem hapus chunk lama dari Qdrant (by source metadata)
  → Insert chunk baru
  → Hasil: data terupdate, bukan duplikasi
```
Menghemat ruang database dan menjaga akurasi pencarian.

### 3. Smart Chunking (Sudah Built-in)
```
chunk_size = 1000     # Tidak terlalu besar (akurasi tinggi)
chunk_overlap = 100   # Tidak terlalu banyak (hemat token)
```

### 4. Score Threshold (Sudah Built-in)
```env
SEARCH_SCORE_THRESHOLD=0.7  # Hanya chunk dengan skor > 0.7 yang dikirim ke LLM
MAX_SEARCH_RESULTS=4        # Maksimal 4 chunk (kontrol pemakaian token)
```

Semakin tinggi threshold → semakin sedikit chunk → semakin hemat token, tapi jawaban mungkin kurang lengkap.

### 5. Memory Window (Sudah Built-in)
```env
MEMORY_WINDOW_SIZE=5  # Hanya simpan 5 turn terakhir
```

Mencegah riwayat chat membengkak yang menyebabkan token LLM meningkat eksponensial.

### 6. Gunakan Ollama (Rekomendasi untuk Budget Nol)
```env
# Embedding gratis + LLM gratis = $0 total
EMBEDDER_BASE_URL="http://localhost:11434"
EMBEDDER_MODEL="nomic-embed-text:latest"
LLM_BASE_URL="http://localhost:11434"
LLM_MODEL="llama3"
```

## Dampak Pilihan Konfigurasi

### Dimensi Embedding vs Kualitas

| Dimensi | Model | Kualitas Retrieval | Kecepatan | Ukuran DB |
|---------|-------|-------------------|-----------|-----------|
| 768 | nomic-embed-text | ★★★☆☆ (Bagus) | Cepat | Kecil |
| 1536 | text-embedding-3-small | ★★★★☆ (Sangat Bagus) | Sedang | Sedang |
| 3072 | text-embedding-3-large | ★★★★★ (Terbaik) | Lambat | Besar |

Dimensi lebih tinggi = akurasi pencarian lebih baik, **tapi** ukuran database lebih besar dan pencarian lebih lambat.

### chunk_size vs Kualitas Jawaban

| chunk_size | Jumlah Chunk | Token per Query | Kualitas |
|------------|-------------|-----------------|----------|
| 500 | Banyak | Sedikit context | Jawaban spesifik tapi mungkin tidak lengkap |
| 1000 | Sedang | Sedang | **Keseimbangan terbaik** ← default |
| 2000 | Sedikit | Banyak context | Jawaban lengkap tapi token lebih boros |

### MAX_SEARCH_RESULTS vs Akurasi

| K | Token Context | Efek |
|---|---------------|------|
| 2 | ~500-1000 | Hemat, tapi mungkin kurang konteks |
| 4 | ~1000-2000 | **Default — keseimbangan baik** |
| 8 | ~2000-4000 | Komprehensif, tapi token mahal |

## Ringkasan: Skenario Penggunaan

| Skenario | Embedding | LLM | Estimasi Biaya/bulan |
|----------|-----------|-----|---------------------|
| Mahasiswa / Riset pribadi | Ollama nomic | Ollama llama3 | **Rp 0** (gratis) |
| Startup kecil | OpenAI small | gpt-4o-mini | ~Rp 15.000 |
| Enterprise (volume tinggi) | OpenAI large | gpt-4o | ~Rp 500.000+ |
| Hybrid (hemat tapi berkualitas) | Ollama nomic | Gemini flash | ~Rp 5.000 |
