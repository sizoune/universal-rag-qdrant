from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from src.config import config
import logging

logger = logging.getLogger(__name__)

# Basic In-Memory Chat History
chat_history = []

# System prompt (defined once for token counting)
SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful AI assistant connected to a knowledge base.\n"
    "Use the following pieces of retrieved context to answer the user's question.\n"
    "If the answer is not in the context, just say that you don't know based on the provided documents. "
    "Do not make up information that isn't supported by the context.\n\n"
    "Context:\n{context}"
)


class DenseThresholdFallbackRetriever(BaseRetriever):
    """Dense retriever with score-threshold first, then similarity fallback."""

    threshold_retriever: object
    similarity_retriever: object

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs):
        docs = self.threshold_retriever.invoke(query)
        if docs:
            return docs
        logger.info(
            "No docs passed score_threshold=%.3f. Falling back to top-k similarity.",
            config.SEARCH_SCORE_THRESHOLD,
        )
        return self.similarity_retriever.invoke(query)


def estimate_tokens(text: str) -> int:
    """Estimate token count. ~1 token per 3 chars for Indonesian, ~4 for English."""
    if not text:
        return 0
    return max(1, len(text) // 3)


def print_token_usage(context_docs, history, question, answer):
    """Print estimated token usage breakdown."""
    context_text = (
        "\n".join(doc.page_content for doc in context_docs) if context_docs else ""
    )
    history_text = " ".join(msg.content for msg in history) if history else ""

    t_system = estimate_tokens(SYSTEM_PROMPT_TEMPLATE)
    t_context = estimate_tokens(context_text)
    t_history = estimate_tokens(history_text)
    t_question = estimate_tokens(question)
    t_answer = estimate_tokens(answer)
    t_input = t_system + t_context + t_history + t_question
    t_total = t_input + t_answer

    print(f"\n[Token Usage (estimated)]:")
    print(
        f"  Context   : ~{t_context:,} tokens ({len(context_docs) if context_docs else 0} chunks)"
    )
    print(f"  History   : ~{t_history:,} tokens")
    print(f"  Question  : ~{t_question:,} tokens")
    print(f"  System    : ~{t_system:,} tokens")
    print(f"  ─────────────────────────")
    print(f"  Input     : ~{t_input:,} tokens")
    print(f"  Output    : ~{t_answer:,} tokens")
    print(f"  TOTAL     : ~{t_total:,} tokens")


def get_llm():
    """Factory function for Chat LLMs using separate LLM_* config."""
    base_url = config.LLM_BASE_URL
    model_name = config.LLM_MODEL
    api_key = config.LLM_API_KEY

    # Heuristic to determine LLM provider
    if base_url and "api.openai.com" in base_url:
        logger.info(f"Using OpenAI Chat with model {model_name}")
        return ChatOpenAI(model=model_name, api_key=api_key)
    elif api_key and api_key.startswith("AIza"):
        logger.info(f"Using Google Gemini Chat with model {model_name}")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    elif base_url and ("ollama" in base_url.lower() or ":11434" in base_url):
        logger.info(f"Using Ollama Chat with model {model_name} at {base_url}")
        return ChatOllama(model=model_name, base_url=base_url)
    elif base_url:
        # Generic OpenAI compatible
        logger.info(
            f"Using OpenAI Compatible Chat at {base_url} with model {model_name}"
        )
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key or "sk-dummy",
            openai_api_base=base_url,
        )
    else:
        # Default Fallback
        logger.info(f"Defaulting to OpenAI Chat with model {model_name}")
        return ChatOpenAI(model=model_name, api_key=api_key)


def get_chat_chain(vector_store):
    """Sets up the retrieval + LLM conversational chain.
    Supports dense (default) and hybrid (dense+sparse) search modes,
    with optional cross-encoder re-ranking.
    """
    llm = get_llm()
    search_mode = config.SEARCH_MODE.lower()

    if search_mode == "hybrid":
        logger.info("Using HYBRID search mode (dense + sparse BM25)")
        from src.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever(
            vector_store=vector_store,
            score_threshold=config.SEARCH_SCORE_THRESHOLD,
            k=config.MAX_SEARCH_RESULTS,
        )
    else:
        logger.info("Using DENSE search mode")
        threshold_retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": config.SEARCH_SCORE_THRESHOLD,
                "k": config.MAX_SEARCH_RESULTS,
            },
        )
        similarity_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.MAX_SEARCH_RESULTS},
        )
        retriever = DenseThresholdFallbackRetriever(
            threshold_retriever=threshold_retriever,
            similarity_retriever=similarity_retriever,
        )

    system_prompt = SYSTEM_PROMPT_TEMPLATE

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


async def stream_chat_response(question: str, session_id: str, vector_store, history: list):
    """Async generator for SSE streaming chat. Two-phase: sync retrieval + async LLM streaming."""
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    search_mode = config.SEARCH_MODE.lower()

    if search_mode == "hybrid":
        from src.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever(
            vector_store=vector_store,
            score_threshold=config.SEARCH_SCORE_THRESHOLD,
            k=config.MAX_SEARCH_RESULTS,
        )
    else:
        threshold_retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": config.SEARCH_SCORE_THRESHOLD,
                "k": config.MAX_SEARCH_RESULTS,
            },
        )
        similarity_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.MAX_SEARCH_RESULTS},
        )
        retriever = DenseThresholdFallbackRetriever(
            threshold_retriever=threshold_retriever,
            similarity_retriever=similarity_retriever,
        )

    # Phase A: Sync retrieval
    context_docs = retriever.invoke(question)

    context_text = "\n".join(doc.page_content for doc in context_docs) if context_docs else ""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_text)

    from langchain_core.messages import SystemMessage

    llm = get_llm()
    formatted = [SystemMessage(content=system_prompt)] + list(history) + [HumanMessage(content=question)]

    # Phase B: Async LLM streaming
    full_answer = []
    async for chunk in llm.astream(formatted):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            full_answer.append(token)
            yield token, "token"

    answer = "".join(full_answer)

    # Yield sources
    sources = list(dict.fromkeys(doc.metadata.get("source", "Unknown") for doc in context_docs))
    yield sources, "sources"

    # Yield token usage
    history_text = " ".join(msg.content for msg in history) if history else ""
    t_input = (
        estimate_tokens(SYSTEM_PROMPT_TEMPLATE)
        + estimate_tokens(context_text)
        + estimate_tokens(history_text)
        + estimate_tokens(question)
    )
    t_output = estimate_tokens(answer)
    yield {"input_estimate": t_input, "output_estimate": t_output, "total_estimate": t_input + t_output}, "token_usage"

    # Update history
    history.extend([HumanMessage(content=question), AIMessage(content=answer)])
    if len(history) > config.MEMORY_WINDOW_SIZE * 2:
        history[:] = history[-config.MEMORY_WINDOW_SIZE * 2 :]


def chat_interface(vector_store):
    """Interactive CLI loop for chatting."""
    print("\n--- Interactive Chat (Type '/exit' to quit) ---")
    chain = get_chat_chain(vector_store)

    global chat_history

    while True:
        user_input = input("\nYou: ")
        if user_input.strip() == "/exit":
            break

        print("\nThinking...")

        try:
            response = chain.invoke({"input": user_input, "chat_history": chat_history})

            answer = response.get("answer", "No answer generated.")
            print(f"AI: {answer}")

            # Optionally print sources (deduplicated)
            context_docs = response.get("context", [])
            if context_docs:
                seen = list(
                    dict.fromkeys(
                        doc.metadata.get("source", "Unknown") for doc in context_docs
                    )
                )
                print("\n[Sources Used]:")
                for i, source in enumerate(seen):
                    print(f"  {i+1}. {source}")

            # Print token usage
            print_token_usage(context_docs, chat_history, user_input, answer)

            # Update Memory Window
            chat_history.extend(
                [HumanMessage(content=user_input), AIMessage(content=answer)]
            )

            # Truncate memory as per config
            if len(chat_history) > config.MEMORY_WINDOW_SIZE * 2:
                chat_history = chat_history[-config.MEMORY_WINDOW_SIZE * 2 :]

        except Exception as e:
            logger.error(f"Chat Error: {e}")
            print(f"Error generating response: {e}")
