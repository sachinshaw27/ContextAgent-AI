# 🧠 ContextAgent

An Agentic AI powered PDF RAG chatbot built using LangChain, LangGraph, ChromaDB, HuggingFace Embeddings, and OpenRouter LLMs.

ContextAgent allows users to upload PDF documents and ask contextual questions directly from the uploaded files using Retrieval-Augmented Generation (RAG).

---

# 🚀 Features

- 📄 Upload multiple PDF documents
- 🔍 Semantic search using vector embeddings
- 🧠 Agentic AI workflow with LangChain Agents
- 💾 Persistent vector storage using ChromaDB
- 🤖 OpenRouter free LLM integration
- 🧵 Memory-enabled conversations with LangGraph
- ⚡ Interactive Streamlit chat interface

---

# 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- LangGraph
- ChromaDB
- HuggingFace Embeddings
- OpenRouter API
- GPT-OSS-20B Free Model

---

# 🏗️ Architecture

```text
                ┌─────────────────┐
                │   User Uploads  │
                │      PDFs       │
                └────────┬────────┘
                         │
                         ▼
              ┌────────────────────┐
              │ PDF Loader         │
              │ PyPDFDirectoryLoader
              └────────┬───────────┘
                       │
                       ▼
             ┌─────────────────────┐
             │ Text Splitter       │
             │ RecursiveCharacter  │
             │ TextSplitter        │
             └────────┬────────────┘
                      │
                      ▼
             ┌─────────────────────┐
             │ HuggingFace         │
             │ Embeddings Model    │
             └────────┬────────────┘
                      │
                      ▼
             ┌─────────────────────┐
             │ Chroma Vector DB    │
             └────────┬────────────┘
                      │
                      ▼
             ┌─────────────────────┐
             │ Retrieval Tool      │
             │ Similarity Search   │
             └────────┬────────────┘
                      │
                      ▼
             ┌─────────────────────┐
             │ LangChain Agent     │
             │ + LangGraph Memory  │
             └────────┬────────────┘
                      │
                      ▼
             ┌─────────────────────┐
             │ OpenRouter LLM      │
             │ GPT-OSS-20B         │
             └────────┬────────────┘
                      │
                      ▼
                ┌────────────┐
                │ AI Answer  │
                └────────────┘
```

---

# 📂 Project Structure

```text
ContextAgent/
│
├── agents.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
├── chroma_db/
├── doc_files/
└── .venv/
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone 
cd ContextAgent
```

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

## 3. Activate Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Get API Key from:

https://openrouter.ai/

---

# ▶️ Run Application

```bash
streamlit run agents.py
```

---

# 📸 Demo Workflow

1. Upload PDF documents
2. Documents are chunked and embedded
3. Embeddings are stored in ChromaDB
4. User asks questions
5. Agent retrieves relevant context
6. OpenRouter LLM generates contextual answers

---

# 🎯 Future Improvements

- Multi-agent workflows
- Hybrid search
- PDF summarization
- Citation support
- Voice interaction
- Authentication system
- Cloud deployment

---

# 👨‍💻 Author

Sachin Shaw

Built for learning Agentic AI, RAG pipelines, and LLM application development.