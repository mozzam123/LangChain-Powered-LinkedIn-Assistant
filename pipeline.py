# src/pipeline.py

from loaders.connections_loader import ConnectionsCSVLoader
from loaders.messages_loader import MessagesCSVLoader
from splitter.text_splitter import split_documents
from embedding.embedder import get_embedder
from vectorstore.faiss_store import build_or_load_faiss_index
import os
import sys

def run_pipeline():
    # Step 1: Load raw documents from CSV files
    messages_loader = MessagesCSVLoader("data/messages.csv")
    connections_loader = ConnectionsCSVLoader("data/connections.csv")

    message_docs = messages_loader.load()
    connection_docs = connections_loader.load()

    # Step 2: Combine all documents
    all_docs = message_docs + connection_docs

    # Step 3: Split documents into chunks
    split_docs = split_documents(all_docs, chunk_size=500, chunk_overlap=50)

    # Step 4: Initialize embedding model
    embedder = get_embedder()

    # Step 5: Build or load FAISS vector index
    vectorstore = build_or_load_faiss_index(split_docs, embedder)

    print(f"Pipeline completed successfully. Indexed {len(split_docs)} chunks.")

def run_cli():
    embedder = get_embedder()
    index_path = "vectorstore/faiss_index"

    if not os.path.exists(index_path):
        print(f"❌ FAISS index not found at {index_path}. Please run pipeline.py first to build the index.")
        sys.exit(1)

    vectorstore = build_or_load_faiss_index([], embedder)

if __name__ == "__main__":
    run_pipeline()
