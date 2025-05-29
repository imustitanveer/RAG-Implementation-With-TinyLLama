import requests
from langchain.embeddings.base import Embeddings

class OllamaMistralEmbeddings(Embeddings):
    def __init__(self, model="mistral", endpoint="http://localhost:11434/api/embeddings"):
        self.model = model
        self.endpoint = endpoint

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        response = requests.post(
            self.endpoint,
            json={"model": self.model, "prompt": text}
        )
        if response.status_code != 200:
            raise Exception(f"Failed to get embedding: {response.text}")
        return response.json()["embedding"]

def get_embedding_function():
    return OllamaMistralEmbeddings()