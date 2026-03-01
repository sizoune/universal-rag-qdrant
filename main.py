import sys
import logging
from src.vector_store import initialize_vector_store, get_db_stats, clear_database
from src.ingestion import parse_web_url, process_directory
from src.chat import chat_interface
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_menu():
    print("\n=============================================")
    print("      UNIVERSAL RAG SYSTEM (Qdrant MVP)      ")
    print("=============================================")
    print("1. [1] Ingest Web URL")
    print("2. [2] Ingest Local Folder/File")
    print("3. [3] Chat / Q&A")
    print("4. [4] Cek Status Index")
    print("5. [5] Re-Index / Clear Database (DANGER)")
    print("6. [0] Exit")
    print("=============================================")


def main():
    try:
        # 1. Initialize Vector Store (includes Strict Dimension Checking)
        print("Initializing Vector Store...")
        vector_store = initialize_vector_store()
        print("Initialization Successful.")
    except Exception as e:
        print(f"\nCRITICAL ERROR INITIALIZING SYSTEM:\n{e}")
        sys.exit(1)

    while True:
        print_menu()
        choice = input("Pilih menu (0-5): ").strip()

        if choice == "1":
            url = input("Masukkan URL Web: ").strip()
            if url:
                docs = parse_web_url(url)
                if docs:
                    print(f"Adding {len(docs)} chunks to Qdrant...")
                    vector_store.add_documents(docs)
                    print("Ingestion selesai.")
                else:
                    print("Gagal mengekstrak teks dari URL.")

        elif choice == "2":
            path = input("Masukkan Path Direktori/File Lokal: ").strip()
            if path:
                docs = process_directory(path)
                if docs:
                    print(
                        f"Adding {len(docs)} chunks to Qdrant (Batch Size: {config.EMBEDDING_BATCH_SIZE})..."
                    )
                    # In a real app we would chunk the add_documents call
                    vector_store.add_documents(docs)
                    print("Ingestion selesai.")
                else:
                    print("Tidak ada dokumen baru yang diproses (atau path salah).")

        elif choice == "3":
            chat_interface(vector_store)

        elif choice == "4":
            stats = get_db_stats()
            print("\n--- Qdrant Status ---")
            for k, v in stats.items():
                print(f"{k}: {v}")

        elif choice == "5":
            print(
                "\nPERINGATAN: Opsi ini akan menghapus semua data (Collection) saat ini!"
            )
            print(
                "Gunakan opsi ini jika Anda telah mengganti Model Embedding (Dimensi berbeda)."
            )
            confirm = input("Ketik 'YA' untuk menghapus: ")
            if confirm == "YA":
                if clear_database():
                    print(
                        "Database dihapus. Silakan restart aplikasi untuk membuat Collection baru."
                    )
                    sys.exit(0)
                else:
                    print("Gagal menghapus database.")
            else:
                print("Operasi dibatalkan.")

        elif choice == "0":
            print("Keluar dari program. Goodbye!")
            sys.exit(0)

        else:
            print("Pilihan tidak valid.")


if __name__ == "__main__":
    main()
