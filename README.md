# 🧠 RAG Test – Retrieval-Augmented Generation with LangChain

This is a simple Python-based Retrieval-Augmented Generation (RAG) application that uses:

- **LangChain**
- **Chroma** for vector storage
- **OpenAI** for embeddings
- **Ollama** (e.g., Mistral) for LLM inference
- `.env` for secure API key handling

---

## 🚀 Features

- Interactive CLI for question-answering
- Vector store with similarity search
- Uses OpenAI Embeddings
- Powered by Ollama's local LLMs (e.g., `mistral`)
- Modular and easy to extend

---

## 🧱 Project Structure

```
RAG Test/
├── chroma/                  # Vector store directory (auto-created)
├── data/                    # Place raw text/data files here
├── rag/                     # Python virtual environment
├── .env                     # Environment variables (not tracked by Git)
├── .gitignore
├── get_embedding.py         # Sets up embedding function
├── populate_db.py           # Loads documents into the Chroma DB
├── query_data.py            # One-off querying logic (optional)
├── test_rag.py              # Main RAG loop script (interactive)
```
---

## 📦 Setup

### 1. Clone the repo and set up your virtual environment

```bash
python -m venv rag
source rag/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install langchain langchain-community openai chromadb python-dotenv
```

3. Create your .env file
```bash
OPENAI_API_KEY=your-openai-key-here
```

## 📚 Populate the Vector Store

Place your text files inside the data/ directory and run:

```bash
python populate_db.py
```

This loads your documents into Chroma with OpenAI embeddings.

## 💬 Run the Interactive RAG Tool
```bash
python test_rag.py
```
You’ll enter a loop where you can type questions.
Type end or exit to quit.

## 🛡️ .gitignore Note

This repo ignores:
	•	rag/ virtual environment
	•	.env file
	•	chroma/ vector DB folder
	•	Python cache files

## 📌 Dependencies
	•	LangChain
	•	ChromaDB
	•	OpenAI
	•	Ollama



## 🧪 Example

Your question: What does the '7' card in Uno No Mercy do?

🧠 Response:
LangChain is a framework for building applications powered by language models...

📚 Sources: ['Monopoly Rules', 'Uno No Mercy Rules']




## 📮 License

MIT or your preferred license.


## 🙋‍♂️ Author

Built by [Mustassum Tanvir]. Questions or suggestions? Open an issue or reach out!