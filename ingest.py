"""
ingest.py
Step 1 of the pipeline: read PDFs from DATA_DIR, split them into chunks,
embed each chunk, and store the embeddings in a local Chroma vector DB.

Run this once whenever you add or change source documents:
    python ingest.py
"""

import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import config


def load_documents():
    """Load every PDF found in the data directory."""
    if not config.DATA_DIR.is_dir() or not list(config.DATA_DIR.glob("*.pdf")):
        raise FileNotFoundError(
            f"No files found in '{config.DATA_DIR}/'. Add some PDFs there first."
        )
    loader = PyPDFDirectoryLoader(str(config.DATA_DIR))
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from '{config.DATA_DIR}/'")
    return documents


def split_documents(documents):
    """Break long documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    """Embed chunks and persist them to a local Chroma database."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(config.VECTOR_DB_DIR),
    )
    print(f"Vector store saved to '{config.VECTOR_DB_DIR}/'")
    return vectorstore


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    print("Ingestion complete. You can now run app.py or rag_chain.py")
