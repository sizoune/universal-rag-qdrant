# Universal RAG System
> Sebuah platform AI berbasis Retrieval-Augmented Generation (RAG) yang efisien, skalabel, dan modular. Menggunakan Qdrant sebagai basis *vector database* untuk mengekstrak dan mencari konteks dokumen atau website menggunakan berbagai model AI (OpenAI, Gemini, Ollama) pilihan Anda.

Universal RAG (Retrieval-Augmented Generation) System adalah framework AI modular yang memungkinkan Anda melakukan "tanya jawab" (Q&A) cerdas dengan data Anda sendiri—baik berupa file lokal maupun artikel website. 

Dibangun di atas fondasi **Qdrant** sebagai *Vector Database* dan **LangChain**, sistem ini dirancang dengan prinsip fleksibilitas tinggi. Anda tidak terikat pada satu penyedia AI; Anda bisa menggunakan OpenAI, Google Gemini, Ollama (untuk inferensi lokal/gratis), atau model *OpenAI-compatible* lainnya hanya dengan mengubah konfigurasi.

## 📖 Dokumentasi

| Dokumen | Deskripsi |
|---------|-----------|
| [Arsitektur RAG](docs/architecture.md) | Cara kerja ingest-web, chunking, embedding, pencarian vektor di Qdrant, dan bagaimana AI menjawab pertanyaan |
| [Penggunaan Token & Biaya](docs/token-usage.md) | Penjelasan token, perbandingan biaya antar provider, strategi hemat, dan dampak konfigurasi |

## 🎯 Target Pengguna

Platform ini dirancang untuk dapat di-skalakan mulai dari penggunaan personal dengan *budget* nol rupiah hingga skala *enterprise*:

*   **Peneliti & Akademisi:** Analisis ratusan jurnal PDF dan artikel web dengan cepat dan gratis menggunakan Ollama (model embedding dan LLM berjalan secara lokal).
*   **Karyawan Perusahaan:** Pencarian informasi instan dari tumpukan SOP, laporan, dan dokumen Word perusahaan menggunakan tingkat akurasi tinggi dari provider *Cloud* seperti OpenAI atau Gemini.
*   **Developer & Data Scientist:** Kerangka kerja (*framework*) siap pakai yang agnostik terhadap *provider* untuk membangun aplikasi AI/RAG kustom dengan pengelolaan *state* (memori), dimensi yang tervalidasi, dan alat migrasi bawaan.

## ✨ Fitur Utama

*   **Multi-Provider Embedding & LLM:** Mendukung OpenAI, Google Gemini, Ollama, dan *OpenAI-Compatible APIs* (Nvidia, Mistral, dll). Cukup ubah URL dan *API Key* di `.env`.
*   **Konfigurasi Independen:** Embedding, Qdrant, dan LLM Chat memiliki konfigurasi terpisah — bisa mix-and-match provider.
*   **Qdrant Vector Database dengan *Strict Dimension Checking*:** Secara otomatis memblokir proses ingesti jika model embedding yang baru memiliki dimensi yang berbeda dengan koleksi data yang sudah ada.
*   **Smart Web Scraper:** Mengekstrak konten utama halaman web secara cerdas (mendukung Wikipedia, artikel HTML5, dan situs umum).
*   **Smart Document Parsing:** Mendukung format `.pdf`, `.txt`, `.csv`, dan `.docx`.
*   **Hash-Based Incremental Caching:** Hanya file baru atau yang berubah isinya yang akan diproses ulang, sangat menghemat biaya API.
*   **Conversational Memory:** Menyimpan konteks percakapan agar AI memahami pertanyaan lanjutan.
*   **CLI Arguments:** Jalankan perintah langsung tanpa menu interaktif.

## 📂 Struktur Proyek

```text
rag-project/
├── .env                  # Konfigurasi (dibuat dari .env.example)
├── .env.example          # Template konfigurasi
├── requirements.txt      # Dependensi Python
├── main.py               # Aplikasi utama (CLI + argparse)
├── migrate.py            # Utility migrasi data antar database/model
├── docs/
│   ├── architecture.md   # Arsitektur RAG & cara kerja sistem
│   └── token-usage.md    # Penggunaan token & dampak biaya
├── src/
│   ├── config.py             # Membaca .env (Embedding, Qdrant, LLM terpisah)
│   ├── embedding_manager.py  # Factory koneksi OpenAI/Gemini/Ollama embedding
│   ├── vector_store.py       # Koneksi Qdrant + validasi dimensi
│   ├── ingestion.py          # Web scraper & pemrosesan dokumen lokal
│   ├── chat.py               # RAG chain (retrieval + LLM) & memori
│   └── utils.py              # Utilitas: hashing, filter file
└── tests/
    ├── conftest.py           # Fixture pytest (isolasi .env)
    ├── test_config.py        # Test konfigurasi
    ├── test_embedding_manager.py  # Test factory embedding
    ├── test_vector_store.py  # Test dimensi & koleksi Qdrant
    └── test_utils.py         # Test hashing & filter
```

## 🚀 Cara Instalasi & Penggunaan

### 1. Persiapan Environment
Pastikan Anda memiliki Python 3.10+ yang terinstal.

```bash
# Buat Virtual Environment 
python -m venv venv

# Aktifkan (Windows)
.\venv\Scripts\activate.bat
# Atau Mac/Linux:
# source venv/bin/activate

# Install Dependensi
pip install -r requirements.txt
```

### 2. Atur Konfigurasi
Copy file `.env.example` menjadi `.env` dan sesuaikan:

```env
# === EMBEDDING CONFIG ===
EMBEDDER_BASE_URL="http://localhost:11434"    # Ollama lokal
EMBEDDER_MODEL="nomic-embed-text:latest"
EMBEDDER_DIMENSION=768

# === QDRANT CONFIG ===
QDRANT_URL="http://localhost:6333"

# === LLM CHAT CONFIG ===
LLM_BASE_URL="http://localhost:11434"         # Bisa beda dari embedding!
LLM_MODEL="llama3"
```

> **Catatan:** Jalankan Qdrant via Docker: `docker run -p 6333:6333 qdrant/qdrant`

### 3. Jalankan Aplikasi

#### Menu Interaktif
```bash
python main.py
```

#### CLI Arguments (Langsung Tanpa Menu)
```bash
# Ingest URL
python main.py ingest-web https://id.wikipedia.org/wiki/SaaS

# Ingest folder/file
python main.py ingest-file ./documents

# Chat interaktif
python main.py chat

# Tanya langsung (jawab & selesai)
python main.py chat "apa itu SaaS?"

# Cek status database
python main.py status

# Hapus database
python main.py clear

# Bantuan
python main.py --help
```

### 4. Menggunakan Migrasi (Opsional)
```bash
python migrate.py
```
Pilih opsi **(1)** untuk memindah ke server Qdrant baru tanpa memproses AI (gratis), atau opsi **(2)** untuk memproses ulang isi teks lama menggunakan Model AI yang baru.
