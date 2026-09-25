from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv

import streamlit as st
import tempfile
import os

load_dotenv()

st.set_page_config(
    page_title="Basic Document Q&A Bot",
    page_icon="📄",
    layout="wide"
)
st.title("📄 Basic Document Q&A Bot")
uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, TXT Files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)
all_documents = []
if uploaded_files:
    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            suffix = "." + uploaded_file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_file_path = tmp_file.name
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_file"] = uploaded_file.name
                    doc.metadata["page_number"] = (
                        doc.metadata.get("page", 0) + 1
                    )
                all_documents.extend(docs)

            elif uploaded_file.name.endswith(".txt"):
                loader = TextLoader(
                    temp_file_path,
                    encoding="utf-8"
                )
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_file"] = uploaded_file.name
                    doc.metadata["page_number"] = "TXT File"

                all_documents.extend(docs)

            elif uploaded_file.name.endswith(".docx"):
                loader = Docx2txtLoader(temp_file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_file"] = uploaded_file.name
                    doc.metadata["page_number"] = "DOCX File"
                all_documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(
            all_documents
        )
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = FAISS.from_documents(
            chunks,
            embedding
        )
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}
        )
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            api_key=os.getenv("GROQ_API_KEY")
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )
    st.success("Documents processed successfully")
    query = st.text_input("Ask Question")
    submit_btn = st.button("Enter")
    if submit_btn and query:
        with st.spinner("Generating answer..."):
            result = qa_chain.invoke({
                "query": query
            })
        st.subheader("Answer")
        st.write(result["result"])
        st.subheader("Sources")
        shown = set()
        for doc in result["source_documents"]:
            source = doc.metadata.get(
                "source_file",
                "Unknown File"
            )
            page = doc.metadata.get(
                "page_number",
                "Unknown Page"
            )
            key = f"{source}-{page}"
            if key not in shown:
                st.write(
                    f"File: {source} | Page: {page}"
                )
                shown.add(key)
