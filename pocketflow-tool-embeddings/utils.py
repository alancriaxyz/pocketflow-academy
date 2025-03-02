import os
from functools import lru_cache
from typing import List, Dict, Any
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@lru_cache(maxsize=1000)
def get_embedding(text: str) -> List[float]:
    """
    Get embedding for a single text using caching to avoid duplicate API calls.
    
    Args:
        text: Text to embed
        
    Returns:
        List of embedding values
    """
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def get_embeddings_batch(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """
    Get embeddings for multiple texts using batching.
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts to process in each batch
        
    Returns:
        List of embeddings
    """
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Check cache first
        batch_embeddings = [get_embedding(text) for text in batch]
        all_embeddings.extend(batch_embeddings)
        
    return all_embeddings

def create_index(embeddings: List[List[float]]) -> faiss.IndexFlatL2:
    """
    Create a Faiss index for vector similarity search.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        Faiss index
    """
    # Convert to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Create index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    return index

def search_similar(index: faiss.IndexFlatL2, 
                  query_embedding: List[float], 
                  k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Search for similar vectors in the index.
    
    Args:
        index: Faiss index
        query_embedding: Query vector
        k: Number of results to return
        
    Returns:
        Tuple of (distances, indices)
    """
    query_array = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_array, k)
    return distances[0], indices[0]

def calculate_similarity(embedding1: List[float], 
                       embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Similarity score between 0 and 1
    """
    # Convert to numpy arrays
    a = np.array(embedding1)
    b = np.array(embedding2)
    
    # Calculate cosine similarity
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def preprocess_text(text: str) -> str:
    """
    Preprocess text before embedding.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text 