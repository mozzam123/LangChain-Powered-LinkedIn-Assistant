from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    print("=========> embedding the data")
    """
    Returns a LangChain-compatible embedding function using HuggingFaceEmbeddings.
    """
    return HuggingFaceEmbeddings(model_name=model_name)