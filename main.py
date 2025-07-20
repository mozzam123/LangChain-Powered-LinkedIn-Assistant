# src/cli/main.py

import os
import sys
from dotenv import load_dotenv
from embedding.embedder import get_embedder
from vectorstore.faiss_store import build_or_load_faiss_index

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_groq import ChatGroq

def run_cli():
    print("🤖 LangChain-Powered LinkedIn Assistant (Groq Edition)")
    print("Ask your LinkedIn-related question below. Type 'exit' to quit.")

    # Load environment variables
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        print("❌ GROQ_API_KEY not found in .env")
        sys.exit(1)

    # Step 1: Load embedder
    embedder = get_embedder()

    # Step 2: Load or build FAISS vector store
    vectorstore = build_or_load_faiss_index([], embedder)

    # Step 3: Initialize Groq LLM (Mixtral model or another)
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama3-8b-8192",
        temperature=0
    )

    # Step 4: Build Retrieval-QA chain
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    # Step 5: CLI Loop
    while True:
        query = input("\n🧠 Ask: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("👋 Exiting the assistant.")
            break

        try:
            answer = qa_chain.run(query)
            print(f"\n💬 Answer:\n{answer}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_cli()




