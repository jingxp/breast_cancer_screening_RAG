import paper_chunk
import chromadb
from google import genai

google_client = genai.Client()
EMBEDDING_MODEL = "gemini-embedding-exp-03-07"
LLM_MODEL = "gemini-2.5-flash-preview-05-20"

def embed(text: str) -> list[float]:
    """
    Embeds the input text using the specified embedding model.
    
    Args:
        text (str): The text to be embedded.
        
    Returns:
        list[float]: The embedding vector for the input text.
    """
    response = google_client.embed(
        model=EMBEDDING_MODEL,
        content=text
    )
    return response.embedding