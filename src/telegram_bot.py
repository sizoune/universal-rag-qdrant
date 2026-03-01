"""
Telegram Bot gateway for Universal RAG System.
Start with: python main.py gateway
"""

import os
import logging
import tempfile
from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ParseMode, ChatAction

from src.config import config
from src.ingestion import parse_web_url, load_local_document
from src.vector_store import get_db_stats, clear_database, delete_by_source
from src.chat import get_chat_chain, estimate_tokens, SYSTEM_PROMPT_TEMPLATE
from src.cache_store import load_cache, save_cache, get_content_hash
from src.utils import get_file_hash

from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# Per-user chat history
user_histories: dict[int, list] = {}

# Will be set by start_bot()
_vector_store = None
_chain = None


def _get_allowed_users() -> set[int]:
    """Parse TELEGRAM_ALLOWED_USERS into a set of user IDs."""
    raw = config.TELEGRAM_ALLOWED_USERS.strip()
    if not raw:
        return set()
    try:
        return {int(uid.strip()) for uid in raw.split(",") if uid.strip()}
    except ValueError:
        return set()


def _is_authorized(user_id: int) -> bool:
    """Check if user is allowed. Empty whitelist = allow everyone."""
    allowed = _get_allowed_users()
    return len(allowed) == 0 or user_id in allowed


def _escape_md(text: str) -> str:
    """Minimal escape for Telegram MarkdownV2."""
    chars = r"_*[]()~`>#+-=|{}.!"
    for c in chars:
        text = text.replace(c, f"\\{c}")
    return text


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    if not _is_authorized(update.effective_user.id):
        await update.message.reply_text("⛔ Anda tidak memiliki akses.")
        return

    welcome = (
        "🤖 *Universal RAG System*\n\n"
        "Saya adalah AI assistant yang terhubung ke knowledge base\\.\n\n"
        "*Commands:*\n"
        "• Ketik pertanyaan apa saja → RAG chat\n"
        "• `/web <url>` → Ingest halaman web\n"
        "• `/status` → Cek database Qdrant\n"
        "• `/clear` → Hapus semua data\n"
        "• `/history` → Reset riwayat chat\n"
        "• Kirim file \\(PDF/TXT/DOCX/CSV\\) → Auto\\-ingest\n\n"
        f"Your User ID: `{update.effective_user.id}`"
    )
    await update.message.reply_text(welcome, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_web(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /web <url> command — ingest a web URL."""
    if not _is_authorized(update.effective_user.id):
        await update.message.reply_text("⛔ Akses ditolak.")
        return

    if not context.args:
        await update.message.reply_text(
            "❌ Gunakan: /web <url>\nContoh: /web https://id.wikipedia.org/wiki/SaaS"
        )
        return

    url = context.args[0]
    await update.message.chat.send_action(ChatAction.TYPING)

    msg = await update.message.reply_text(f"🔄 Scraping {url}...")

    docs, changed = parse_web_url(url)

    if not changed:
        await msg.edit_text("✅ Konten tidak berubah (cached). Skipped.")
        return

    if not docs:
        await msg.edit_text("❌ Gagal mengekstrak teks dari URL.")
        return

    # Delete old chunks + insert new
    deleted = delete_by_source(url)
    status = f"🗑️ Dihapus {deleted} chunk lama\n" if deleted else ""

    await msg.edit_text(f"{status}📦 Embedding {len(docs)} chunks...")
    _vector_store.add_documents(docs)

    await msg.edit_text(
        f"✅ Ingestion selesai!\n"
        f"{status}"
        f"📦 {len(docs)} chunks baru ditambahkan\n"
        f"🔗 {url}"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command."""
    if not _is_authorized(update.effective_user.id):
        await update.message.reply_text("⛔ Akses ditolak.")
        return

    stats = get_db_stats()

    if "error" in stats:
        await update.message.reply_text(f"❌ Error: {stats['error']}")
        return

    text = (
        f"📊 *Qdrant Status*\n\n"
        f"Collection: `{stats.get('collection_name', 'N/A')}`\n"
        f"Documents: `{stats.get('document_count', 0)}`\n"
        f"Dimension: `{stats.get('dimension', 'N/A')}`\n"
        f"Status: `{stats.get('status', 'Unknown')}`"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /clear command with confirmation."""
    if not _is_authorized(update.effective_user.id):
        await update.message.reply_text("⛔ Akses ditolak.")
        return

    # Require "confirm" argument
    if not context.args or context.args[0].lower() != "confirm":
        await update.message.reply_text(
            "⚠️ *PERINGATAN:* Ini akan menghapus SEMUA data\\!\n\n"
            "Untuk konfirmasi, ketik:\n`/clear confirm`",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    success = clear_database()
    if success:
        # Clear hash cache too
        save_cache({})
        await update.message.reply_text("🗑️ Database berhasil dihapus.")
    else:
        await update.message.reply_text("❌ Gagal menghapus database.")


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /history — reset chat history."""
    user_id = update.effective_user.id
    user_histories[user_id] = []
    await update.message.reply_text("🔄 Riwayat chat direset.")


async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle file uploads — auto-ingest documents."""
    if not _is_authorized(update.effective_user.id):
        await update.message.reply_text("⛔ Akses ditolak.")
        return

    document = update.message.document
    if not document:
        return

    filename = document.file_name or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    allowed_exts = {".pdf", ".txt", ".csv", ".docx", ".md", ".py", ".js", ".html"}

    if ext not in allowed_exts:
        await update.message.reply_text(
            f"❌ Format `{ext}` tidak didukung.\n"
            f"Format yang didukung: {', '.join(sorted(allowed_exts))}"
        )
        return

    await update.message.chat.send_action(ChatAction.TYPING)
    msg = await update.message.reply_text(f"📥 Downloading {filename}...")

    # Download file to temp
    file = await document.get_file()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = tmp.name
        await file.download_to_drive(tmp_path)

    try:
        await msg.edit_text(f"📄 Processing {filename}...")
        docs = load_local_document(tmp_path)

        if not docs:
            await msg.edit_text(f"❌ Gagal memproses {filename}")
            return

        # Set source to original filename
        for doc in docs:
            doc.metadata["source"] = filename
            doc.metadata["source_type"] = "telegram_upload"

        # Delete old + insert new
        deleted = delete_by_source(filename)
        status = f"🗑️ Dihapus {deleted} chunk lama\n" if deleted else ""

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        await msg.edit_text(f"{status}📦 Embedding {len(chunks)} chunks...")
        _vector_store.add_documents(chunks)

        await msg.edit_text(
            f"✅ File berhasil di-ingest!\n"
            f"{status}"
            f"📄 {filename}\n"
            f"📦 {len(chunks)} chunks"
        )
    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def handle_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages — RAG chat."""
    if not _is_authorized(update.effective_user.id):
        await update.message.reply_text("⛔ Akses ditolak.")
        return

    user_id = update.effective_user.id
    question = update.message.text.strip()

    if not question:
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    # Get or create user history
    history = user_histories.get(user_id, [])

    try:
        response = _chain.invoke({"input": question, "chat_history": history})
        answer = response.get("answer", "Tidak ada jawaban.")

        # Build reply
        reply_parts = [answer]

        # Sources (deduplicated)
        context_docs = response.get("context", [])
        if context_docs:
            seen = list(
                dict.fromkeys(
                    doc.metadata.get("source", "Unknown") for doc in context_docs
                )
            )
            reply_parts.append("\n📚 Sources:")
            for i, source in enumerate(seen):
                reply_parts.append(f"  {i+1}. {source}")

        # Token usage
        context_text = (
            "\n".join(doc.page_content for doc in context_docs) if context_docs else ""
        )
        history_text = " ".join(msg.content for msg in history) if history else ""
        t_context = estimate_tokens(context_text)
        t_history = estimate_tokens(history_text)
        t_question = estimate_tokens(question)
        t_system = estimate_tokens(SYSTEM_PROMPT_TEMPLATE)
        t_answer = estimate_tokens(answer)
        t_input = t_system + t_context + t_history + t_question
        t_total = t_input + t_answer

        reply_parts.append(
            f"\n📊 Tokens: ~{t_input:,} in / ~{t_answer:,} out / ~{t_total:,} total"
        )

        full_reply = "\n".join(reply_parts)

        # Telegram message limit is 4096 chars
        if len(full_reply) > 4000:
            # Send in chunks
            for i in range(0, len(full_reply), 4000):
                await update.message.reply_text(full_reply[i : i + 4000])
        else:
            await update.message.reply_text(full_reply)

        # Update history
        history.extend([HumanMessage(content=question), AIMessage(content=answer)])
        if len(history) > config.MEMORY_WINDOW_SIZE * 2:
            history = history[-config.MEMORY_WINDOW_SIZE * 2 :]
        user_histories[user_id] = history

    except Exception as e:
        logger.error(f"Chat error for user {user_id}: {e}")
        await update.message.reply_text(f"❌ Error: {e}")


async def post_init(app: Application):
    """Set bot commands after initialization."""
    commands = [
        BotCommand("start", "Mulai & help"),
        BotCommand("web", "Ingest URL: /web <url>"),
        BotCommand("status", "Cek status database"),
        BotCommand("clear", "Hapus database"),
        BotCommand("history", "Reset riwayat chat"),
    ]

    # Delete old commands first to force refresh
    await app.bot.delete_my_commands()

    # Set for all scopes
    await app.bot.set_my_commands(commands)

    # Also set for private chats explicitly
    from telegram import BotCommandScopeAllPrivateChats

    await app.bot.set_my_commands(commands, scope=BotCommandScopeAllPrivateChats())

    bot_info = await app.bot.get_me()
    logger.info(f"Bot commands registered for @{bot_info.username}")
    print(f"   Bot: @{bot_info.username}")
    print(f"   Commands: {', '.join('/' + c.command for c in commands)}")


def start_bot(vector_store):
    """Start the Telegram bot (polling mode)."""
    global _vector_store, _chain

    token = config.TELEGRAM_BOT_TOKEN
    if not token:
        print("\n❌ TELEGRAM_BOT_TOKEN tidak diset di .env!")
        print("   1. Buka Telegram → cari @BotFather")
        print("   2. Kirim /newbot → ikuti instruksi")
        print('   3. Copy token ke .env: TELEGRAM_BOT_TOKEN="your-token"')
        return

    _vector_store = vector_store
    _chain = get_chat_chain(vector_store)

    print("\n🤖 Starting Telegram Bot Gateway...")
    print("   Press Ctrl+C to stop\n")

    app = Application.builder().token(token).post_init(post_init).build()

    # Command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("web", cmd_web))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("history", cmd_history))

    # File handler
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))

    # Chat handler (any text that's not a command)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_chat))

    logger.info("Telegram Bot started in polling mode.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
