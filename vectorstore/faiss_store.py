# src/vectorstore/faiss_store.py
import os
from typing import List
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings

def build_or_load_faiss_index(
    docs: List[Document],
    embedder: Embeddings,
    index_path: str = "vectorstore/faiss_index"
):
    """
    Builds a FAISS vector store from documents or loads it if already exists.
    """
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embedder)
    
    vectorstore = FAISS.from_documents(docs, embedder)
    vectorstore.save_local(index_path)
    return vectorstore
