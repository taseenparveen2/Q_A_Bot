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
import os

load_dotenv()

documents = []
folder = "data"

# Load files
for file in os.listdir(folder):
    path = os.path.join(folder, file)

    if file.endswith(".pdf"):
        loader = PyPDFLoader(path)
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = file
            page.metadata["page"] = page.metadata.get("page", 0) + 1

        documents.extend(pages)

    elif file.endswith(".txt"):
        loader = TextLoader(path, encoding="utf-8")
        text = loader.load()

        for page in text:
            page.metadata["source"] = file
            page.metadata["page"] = "TXT"

        documents.extend(text)

    elif file.endswith(".docx"):
        loader = Docx2txtLoader(path)
        text = loader.load()

        for page in text:
            page.metadata["source"] = file
            page.metadata["page"] = "DOCX"

        documents.extend(text)

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# Ask questions
while True:
    question = input("Ask: ")

    if question.lower() == "exit":
        break

    response = qa.invoke({"query": question})

    print("\nAnswer:\n")
    print(response["result"])

    print("\nSources:\n")

    shown = set()

    for doc in response["source_documents"]:
        file_name = doc.metadata["source"]
        page_no = doc.metadata["page"]

        key = f"{file_name}-{page_no}"

        if key not in shown:
            print(f"File: {file_name} | Page: {page_no}")
            shown.add(key)

    print("\n" + "=" * 50 + "\n")