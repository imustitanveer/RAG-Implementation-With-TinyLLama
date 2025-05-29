from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from get_embedding import get_embedding_function

import time

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def load_chain(model) -> Runnable:
    """Builds the prompt -> model -> parser chain."""
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    parser = StrOutputParser()
    return prompt | model | parser


def query_rag(query: str, db: Chroma, chain: Runnable):
    """Performs RAG search + generation."""
    results = db.similarity_search_with_score(query, k=5)
    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    inputs = {"context": context, "question": query}

    print("\n🧠 Response:\n", end="", flush=True)
    for chunk in chain.stream(inputs):
        print(chunk, end="", flush=True)
        time.sleep(0.01)

    sources = [doc.metadata.get("id") for doc, _ in results]
    print(f"\n📚 Sources: {sources}")


def main():
    print("🔍 Ask your question. Type 'exit' to quit.")
    embedding_fn = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_fn)
    model = Ollama(model="mistral")
    chain = load_chain(model)

    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in {"exit", "quit", "end"}:
            print("👋 Goodbye!")
            break
        query_rag(query, db, chain)


if __name__ == "__main__":
    main()