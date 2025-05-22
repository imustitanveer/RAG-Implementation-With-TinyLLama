from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

def get_embedding_function():
    load_dotenv()  # load .env file

    # Optionally validate the key is loaded
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in .env")

    embeddings = OpenAIEmbeddings()
    return embeddings