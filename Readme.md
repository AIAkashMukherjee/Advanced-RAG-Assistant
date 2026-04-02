# Research Assistant — Advanced RAG System

A powerful, modular **Research Assistant** built with **Advanced Retrieval-Augmented Generation (RAG)**. It combines Hybrid Search, Parent-Child Chunking, Reranking, and Long-Context Reordering to deliver highly accurate and contextually rich answers from your research documents.

Designed for researchers, students, and knowledge workers who need reliable answers from PDFs, research papers, theses, and reports.

---

## ✨ Key Features

- **Hybrid Retrieval**: Vector Search + BM25 (Keyword) using Ensemble
- **Parent-Child Chunking**: Retrieves small precise chunks but returns rich context
- **Reranking**: Reorders results using FlashRank (fast & local)
- **Long Context Reorder**: Optimizes context for large language models
- **Multi-Document Support**: PDF, TXT, and Web pages
- **Streamlit UI**: Beautiful chat interface with source citations
- **Fully Modular & Configurable**: Easy to extend or swap components
- **Local-first**: Supports HuggingFace embeddings and open-source LLMs

---

## 🏗️ Architecture

The system follows a clean, production-ready modular design:

Advanced RAG Pipeline
├── Document Loader
├── Text Chunker (Recursive + Parent-Child)
├── Embedding Model (HuggingFace / OpenAI)
├── Vector Store (Chroma)
├── Hybrid Retriever (Vector + BM25)
├── Parent Retriever
├── Merger Retriever
├── Reranker (FlashRank)
├── Long Context Reorder
└── LLM (OpenAI / Groq / Ollama / etc.)


## 🚀 Installation



### 1. Clone the repository

```
git clone https://github.com/AIAkashMukherjee/Advanced-RAG-Assistant.git
```

```
cd Advanced-RAG-Assistant
```


### 2. Create virtual environment

> python -m venv my_env
> source my_env/bin/activate  
> my_env\Scripts\activate

### 3. Install dependencies

Bash

```
pip install -r requirements.txt
```

## 💡 Usage

### 1. Command Line (CLI)

Bash

```
python main.py
```

### 2. Streamlit Web UI (Recommended)

Bash

```
streamlit run app/streamlit_app.py
```

 **Features in UI** :

* Upload multiple PDFs/TXT files
* Chat with your documents
* View sources with highlighted context
* Clean chat history

## 🔄 How It Works (Pipeline Flow)

1. **Query** → Hybrid Retriever (Vector + BM25)
2. **Retrieval** → Parent Retriever (rich context)
3. **Reranking** → FlashRank reorders by relevance
4. **Reordering** → Long Context Reorder (best chunks in middle)
5. **LLM Generation** → Final accurate answer with sources

---

## 🧪 Example Use Cases

* Research paper summarization
* Literature review assistance
* Thesis/Dissertation Q&A
* Company knowledge base
* Legal document analysis
* Technical documentation assistant

## 🙌 Acknowledgments

Built with:

* [LangChain](https://www.langchain.com/)
* [Chroma](https://www.trychroma.com/)
* [FlashRank](https://github.com/prithivirajdamodaran/flashrank)
* [Streamlit](https://streamlit.io/)

---

**Made with ❤️ for researchers who want accurate answers.**
