from sentence_transformers import SentenceTransformer



def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """
    Returns an embedding function using SentenceTransformer model.
    """
    return SentenceTransformer(model_name=model_name)