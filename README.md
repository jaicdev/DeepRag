# 📘 DeepRag: Intelligent Document Assistant

DeepRag is a Streamlit app that lets you chat with your PDF documents using advanced RAG techniques. Upload any PDF and ask questions to get concise, accurate answers extracted directly from the document content.

## ✨ Key Features

* **Advanced RAG Pipeline** with hybrid search, multi-vector retrieval, and LLM re-ranking
* **Private & Local** processing using Ollama for language models
* **Source Citations** showing exactly which document passages were used
* **Dark-themed UI** with intuitive chat interface

## 🚀 Quick Start

1. Install Ollama from [ollama.com](https://ollama.com/)
2. Pull the model: `ollama pull deepseek-r1:1.5b`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `streamlit run app.py`

## 🧠 How It Works

DeepRag creates multiple representations of your document (original text, summaries, entities), indexes them using both keyword and semantic search, and employs an LLM to re-rank and synthesize the most relevant information when answering your questions.

Built with Python, Streamlit, LangChain, and Ollama.
