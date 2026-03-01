from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from src.config import config
import logging

logger = logging.getLogger(__name__)

# Basic In-Memory Chat History
chat_history = []


def get_llm():
    """Factory function for Chat LLMs (parallel to get_embedder)"""
    base_url = config.EMBEDDER_BASE_URL
    model_name = (
        config.EMBEDDER_MODEL
    )  # Defaulting to same, might need separate config in future
    api_key = config.EMBEDDER_API_KEY

    # We will use simple heuristic to match the LLM provider
    if base_url and "api.openai.com" in base_url:
        return ChatOpenAI(
            model="gpt-3.5-turbo", api_key=api_key
        )  # Fixed standard model for chat
    elif api_key and api_key.startswith("AIza"):
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    elif base_url and "ollama" in base_url.lower():
        return ChatOllama(model="llama3", base_url=base_url)
    elif base_url:
        # Generic OpenAI compatible
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key or "sk-dummy",
            openai_api_base=base_url,
        )
    else:
        # Default Fallback
        return ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)


def get_chat_chain(vector_store):
    """Sets up the retrieval + LLM conversational chain."""
    llm = get_llm()

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": config.SEARCH_SCORE_THRESHOLD,
            "k": config.MAX_SEARCH_RESULTS,
        },
    )

    system_prompt = (
        "You are a helpful AI assistant connected to a knowledge base.\n"
        "Use the following pieces of retrieved context to answer the user's question.\n"
        "If the answer is not in the context, just say that you don't know based on the provided documents. "
        "Do not make up information that isn't supported by the context.\n\n"
        "Context:\n{context}"
    )

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

            # Optionally print sources
            sources = response.get("context", [])
            if sources:
                print("\n[Sources Used]:")
                for i, doc in enumerate(sources):
                    source = doc.metadata.get("source", "Unknown")
                    print(f"  {i+1}. {source}")

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
