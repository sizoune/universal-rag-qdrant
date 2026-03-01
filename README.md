# Universal RAG System
> Sebuah platform AI berbasis Retrieval-Augmented Generation (RAG) yang efisien, skalabel, dan modular. Menggunakan Qdrant sebagai basis *vector database* untuk mengekstrak dan mencari konteks dokumen atau website menggunakan berbagai model AI (OpenAI, Gemini, Ollama) pilihan Anda.

Universal RAG (Retrieval-Augmented Generation) System adalah framework AI modular yang memungkinkan Anda melakukan "tanya jawab" (Q&A) cerdas dengan data Anda sendiri—baik berupa file lokal maupun artikel website. 

Dibangun di atas fondasi **Qdrant** sebagai *Vector Database* dan **LangChain**, sistem ini dirancang dengan prinsip fleksibilitas tinggi. Anda tidak terikat pada satu penyedia AI; Anda bisa menggunakan OpenAI, Google Gemini, Ollama (untuk inferensi lokal/gratis), atau model *OpenAI-compatible* lainnya hanya dengan mengubah konfigurasi.

## 🎯 Target Pengguna

Platform ini dirancang untuk dapat di-skalakan mulai dari penggunaan personal dengan *budget* nol rupiah hingga skala *enterprise*:

*   **Peneliti & Akademisi:** Analisis ratusan jurnal PDF dan artikel web dengan cepat dan gratis menggunakan Ollama (model embedding dan LLM berjalan secara lokal).
*   **Karyawan Perusahaan:** Pencarian informasi instan dari tumpukan SOP, laporan, dan dokumen Word perusahaan menggunakan tingkat akurasi tinggi dari provider *Cloud* seperti OpenAI atau Gemini.
*   **Developer & Data Scientist:** Kerangka kerja (*framework*) siap pakai yang agnostik terhadap *provider* untuk membangun aplikasi AI/RAG kustom dengan pengelolaan *state* (memori), dimensi yang tervalidasi, dan alat migrasi bawaan.

## ✨ Fitur Utama (Fase 1 - MVP)

*   **Multi-Provider Embedding & LLM:** Mendukung OpenAI, Google Gemini, Ollama, dan *OpenAI-Compatible APIs* (Nvidia, Mistral, dll). Cukup ubah URL dan *API Key* di `.env`.
*   **Qdrant Vector Database dengan *Strict Dimension Checking*:** Secara otomatis memblokir proses ingesti jika model embedding yang baru memiliki dimensi (misal: 1536 vs 4096) yang berbeda dengan koleksi data yang sudah ada, mencegah error *database* yang fatal.
*   **Smart Document Parsing:** Mendukung format `.pdf`, `.txt`, `.csv`, dan `.docx`.
*   **Web Scraper bawaan:** Mengekstrak teks bersih dari URL mana pun secara otomatis.
*   **Hash-Based Incremental Caching:** Secara otomatis mengenkripsi *hash* (SHA-256) setiap file; hanya file baru atau yang berubah isinya yang akan diproses ulang, sangat menghemat biaya API.
*   **Conversational Memory (Memori Obrolan):** Menyimpan konteks percakapan secara *real-time* (dengan batas jendela memori yang dapat diatur) agar AI memahami pertanyaan lanjutan (*follow-up questions*).
*   **Database Migration Utilities:** Alat khusus (`migrate.py`) untuk:
    *   **Skenario A:** Memindahkan data dari Qdrant lama ke Qdrant baru tanpa membuang biaya untuk *embed* ulang.
    *   **Skenario B:** Mengekstrak teks dari *database* lama untuk di-*embed* ulang saat Anda menggunakan model AI baru yang memiliki dimensi berbeda.

## 📂 Struktur Proyek

```text
rag-project/
├── .env                  # (Harus dibuat dari .env.example) Konfigurasi rahasia dan parameter sistem.
├── requirements.txt      # Daftar dependensi Python (LangChain, Qdrant Client, dll).
├── main.py               # Aplikasi utama (CLI) yang menampilkan menu interaktif.
├── migrate.py            # Utility script untuk migrasi data antar database atau model.
├── src/
│   ├── config.py             # Membaca file .env dan menginisiasi variabel utama.
│   ├── embedding_manager.py  # Factory untuk meremot koneksi ke OpenAI/Gemini/Ollama berdasarkan config.
│   ├── vector_store.py       # Koneksi ke Qdrant dan validasi Strict Dimension Checking.
│   ├── ingestion.py          # Modul ekstraksi Web URL dan pemrosesan folder/dokumen.
│   ├── chat.py               # Logika rantai LLM (Q&A) dan memori obrolan.
│   └── utils.py              # Fungsi-fungsi utilitas untuk hashing dan filter file (.git, dll).
```

## 🚀 Cara Instalasi & Penggunaan

### 1. Persiapan Environment
Pastikan Anda memiliki Python 3.10+ yang terinstal.
Buka terminal/CMD di dalam folder `rag-project`:

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
Copy file `.env.example` menjadi `.env`.
Isilah setidaknya parameter berikut (ini contoh menggunakan Gemini):

```env
EMBEDDER_BASE_URL=""                            # Kosongkan untuk Gemini/OpenAI standard
EMBEDDER_API_KEY="AIzaSy...your_gemini_key"
EMBEDDER_MODEL="models/embedding-001"
EMBEDDER_DIMENSION=768                          # Pastikan dimensi sesuai dengan model Anda!
```

*Catatan: Jika menggunakan Qdrant lokal, jalankan via Docker: `docker run -p 6333:6333 qdrant/qdrant` terlebih dahulu.*

### 3. Jalankan Aplikasi Utama
```bash
python main.py
```
Anda akan disuguhkan menu CLI interaktif:
1.  **[1] Ingest Web URL:** Masukkan link website apa pun untuk dibaca AI.
2.  **[2] Ingest Local Folder/File:** Ketik *path* folder Anda (misal: `./data_saya`), sistem akan mencari file PDF/DOCX dsb di dalamnya (mengabaikan file biner atau folder node_modules secara cerdas).
3.  **[3] Chat / Q&A:** Mulai ngobrol dengan data Anda!
4.  **[4] Cek Status Index:** Lihat berapa dokumen yang sudah tersimpan di Qdrant beserta pengaturan dimensinya.
5.  **[5] Re-Index:** Hapus *database* untuk mengatur ulang (biasanya dipakai jika ingin mengganti ke model AI lain).

### 4. Menggunakan Migrasi (Opsional)
Jika Anda butuh memigrasi data:
```bash
python migrate.py
```
Pilih opsi **(1)** untuk memindah ke server Qdrant baru tanpa memproses AI (gratis), atau opsi **(2)** untuk memproses ulang isi teks lama menggunakan Model AI yang baru.
