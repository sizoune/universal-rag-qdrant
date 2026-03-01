import sys
import os
import argparse
import logging
from src.vector_store import (
    initialize_vector_store,
    get_db_stats,
    clear_database,
    delete_by_source,
)
from src.ingestion import parse_web_url, process_directory
from src.chat import chat_interface
from src.config import config

# Configure logging (use LOG_LEVEL from .env, default INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
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


def init_system():
    """Initialize the vector store or exit on failure."""
    try:
        print("Initializing Vector Store...")
        vector_store = initialize_vector_store()
        print("Initialization Successful.")
        return vector_store
    except Exception as e:
        print(f"\nCRITICAL ERROR INITIALIZING SYSTEM:\n{e}")
        sys.exit(1)


def do_ingest_web(vector_store, url):
    """Ingest a web URL into the vector store (with dedup)."""
    if not url:
        url = input("Masukkan URL Web: ").strip()
    if url:
        docs, changed = parse_web_url(url)
        if not changed:
            print("Content unchanged (cached). Skipping ingestion.")
            return
        if docs:
            # Delete old chunks from this URL first
            deleted = delete_by_source(url)
            if deleted:
                print(f"Removed {deleted} old chunks for this URL.")
            print(f"Adding {len(docs)} new chunks to Qdrant...")
            vector_store.add_documents(docs)
            print("Ingestion selesai.")
        else:
            print("Gagal mengekstrak teks dari URL.")


def do_ingest_file(vector_store, path):
    """Ingest a local directory/file into the vector store (with dedup)."""
    if not path:
        path = input("Masukkan Path Direktori/File Lokal: ").strip()
    if path:
        docs, changed_sources = process_directory(path)
        if docs:
            # Delete old chunks for each changed source
            total_deleted = 0
            for source in changed_sources:
                total_deleted += delete_by_source(source)
            if total_deleted:
                print(f"Removed {total_deleted} old chunks from changed files.")
            print(
                f"Adding {len(docs)} new chunks to Qdrant (Batch Size: {config.EMBEDDING_BATCH_SIZE})..."
            )
            vector_store.add_documents(docs)
            print("Ingestion selesai.")
        else:
            print(
                "Tidak ada dokumen baru yang diproses (semua file unchanged atau path salah)."
            )


def do_status():
    """Print database status."""
    stats = get_db_stats()
    print("\n--- Qdrant Status ---")
    for k, v in stats.items():
        print(f"{k}: {v}")


def do_clear():
    """Clear the entire database (dangerous!)."""
    print("\nPERINGATAN: Opsi ini akan menghapus semua data (Collection) saat ini!")
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


def interactive_menu(vector_store):
    """Run the interactive menu loop."""
    while True:
        print_menu()
        choice = input("Pilih menu (0-5): ").strip()

        if choice == "1":
            do_ingest_web(vector_store, None)
        elif choice == "2":
            do_ingest_file(vector_store, None)
        elif choice == "3":
            chat_interface(vector_store)
        elif choice == "4":
            do_status()
        elif choice == "5":
            do_clear()
        elif choice == "0":
            print("Keluar dari program. Goodbye!")
            sys.exit(0)
        else:
            print("Pilihan tidak valid.")


def build_parser():
    """Build the argparse CLI parser."""
    parser = argparse.ArgumentParser(
        description="Universal RAG System — Ingest, Index, Chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                  # Interactive menu
  python main.py ingest-web https://example.com   # Ingest a URL
  python main.py ingest-file ./documents           # Ingest a folder
  python main.py chat                              # Interactive chat
  python main.py chat "apa itu SaaS"               # Single question, direct answer
  python main.py status                            # Check index status
  python main.py clear                             # Clear database
  python main.py gateway                           # Start Telegram Bot
        """,
    )

    sub = parser.add_subparsers(dest="command")

    # ingest-web
    p_web = sub.add_parser("ingest-web", help="Ingest a web URL")
    p_web.add_argument("url", help="URL to scrape and ingest")

    # ingest-file
    p_file = sub.add_parser("ingest-file", help="Ingest a local folder or file")
    p_file.add_argument("path", help="Path to the folder or file")

    # chat
    p_chat = sub.add_parser("chat", help="Chat / Q&A (interactive or single question)")
    p_chat.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Optional question — if provided, answers directly without interactive loop",
    )

    # status
    sub.add_parser("status", help="Check Qdrant index status")

    # clear
    sub.add_parser("clear", help="Re-Index / Clear database (DANGER)")

    # gateway
    sub.add_parser("gateway", help="Start Telegram Bot gateway")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    vector_store = init_system()

    if args.command is None:
        # No subcommand → interactive menu
        interactive_menu(vector_store)
    elif args.command == "ingest-web":
        do_ingest_web(vector_store, args.url)
    elif args.command == "ingest-file":
        do_ingest_file(vector_store, args.path)
    elif args.command == "chat":
        if args.question:
            # Single-shot mode: answer and exit
            from src.chat import get_chat_chain, print_token_usage

            chain = get_chat_chain(vector_store)
            print(f"\nYou: {args.question}\n")
            print("Thinking...")
            try:
                response = chain.invoke({"input": args.question, "chat_history": []})
                answer = response.get("answer", "No answer generated.")
                print(f"AI: {answer}")
                context_docs = response.get("context", [])
                if context_docs:
                    seen = list(
                        dict.fromkeys(
                            doc.metadata.get("source", "Unknown")
                            for doc in context_docs
                        )
                    )
                    print("\n[Sources Used]:")
                    for i, source in enumerate(seen):
                        print(f"  {i+1}. {source}")
                print_token_usage(context_docs, [], args.question, answer)
            except Exception as e:
                print(f"Error: {e}")
        else:
            chat_interface(vector_store)
    elif args.command == "status":
        do_status()
    elif args.command == "clear":
        do_clear()
    elif args.command == "gateway":
        from src.telegram_bot import start_bot

        start_bot(vector_store)


if __name__ == "__main__":
    main()
