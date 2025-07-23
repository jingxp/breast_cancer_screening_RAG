import paper_chunk
import chromadb
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
google_client = genai.Client(api_key=GOOGLE_API_KEY)
EMBEDDING_MODEL = "gemini-embedding-exp-03-07"
LLM_MODEL = "gemini-2.5-flash-preview-05-20"

chromadb_client = chromadb.PersistentClient('./chroma_db')
chromadb_collection = chromadb_client.get_or_create_collection(
    name="breast_cancer_MRI",
)

def embed(text: str, store: bool) -> list[float]:
    """
    Embeds the input text using the specified embedding model.
    
    Args:
        text (str): The text to be embedded.
        
    Returns:
        list[float]: The embedding vector for the input text.
    """
    response = google_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={
            "task_type": "RETRIEVAL_DOCUMENT" if store else "RETRIEVAL_QUERY",
        }
    )
    assert response.embeddings
    assert response.embeddings[0].values
    return response.embeddings[0].values

def store_embedding() -> None:
    for i, chunk in enumerate(paper_chunk.get_chunks(paper_chunk.read_data())):
        print(f"Processing chunk {chunk}...")
        embedding = embed(chunk, store=True)
        chromadb_collection.upsert(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
        )

if __name__ == "__main__":
    store_embedding()