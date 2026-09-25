# AI-Powered Multi-Document Q&A Bot (RAG)

An AI-powered Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF, DOCX, and TXT documents and ask questions in natural language. The application retrieves the most relevant content using FAISS and generates accurate answers with Groq's Llama 3.3 70B model.

## Features

- Upload PDF, DOCX, and TXT files
- Supports multiple document uploads
- Automatic text chunking
- Semantic search with FAISS
- HuggingFace embeddings
- AI-powered question answering using Groq
- Interactive Streamlit interface

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq LLM
- Llama 3.3 70B

## Model Used

- Provider: Groq
- Model: llama-3.3-70b-versatile
- Context Window: 131K tokens

## Workflow

1. Upload Documents
2. Load Files
3. Split Text into Chunks
4. Generate Embeddings
5. Store in FAISS
6. Retrieve Relevant Chunks
7. Generate Answer with Groq LLM

## Project Structure

RAG_QA_BOT/
│── app.py
│── requirements.txt
│── .env
│── README.md
│
├── data/
│   ├── sample.pdf
│   ├── notes.docx
│   └── history.txt
│
└── vectorstore/
    └── faiss_index

## Installation

```bash
git clone https://github.com/your-username/RAG_QA_BOT.git

cd RAG_QA_BOT

pip install -r requirements.txt

streamlit run app.py