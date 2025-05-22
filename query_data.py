from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import time
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from get_embedding import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    print("🔍 Enter your question below. Type 'end' to quit.")
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = Ollama(model="tinyllama")

    while True:
        query_text = input("\nYour question: ").strip()
        if query_text.lower() in {"end", "exit", "quit"}:
            print("👋 Goodbye!")
            break
        query_rag(query_text, db, model)


def query_rag(query_text: str, db, model):
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Construct a runnable chain
    chain: Runnable = prompt_template | model | StrOutputParser()

    print("\n🧠 Response:\n", end="", flush=True)

    for chunk in chain.stream({"context": context_text, "question": query_text}):
        for char in chunk:
            print(char, end="", flush=True)
            time.sleep(0.01)  # You can adjust speed here

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    print(f"\n📚 Sources: {sources}")


if __name__ == "__main__":
    main()