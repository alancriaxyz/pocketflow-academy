import os
from typing import List, Dict, Any
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from pocketflow import Node

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class TextPreprocessNode(Node):
    """Node for preprocessing text before embedding"""
    
    def prep(self, shared):
        return shared.get("text", "")
        
    def exec(self, text):
        # Convert to lowercase and remove extra whitespace
        text = text.lower()
        text = " ".join(text.split())
        return text
        
    def post(self, shared, prep_res, exec_res):
        shared["preprocessed_text"] = exec_res
        return "default"

class GetEmbeddingNode(Node):
    """Node for getting embeddings from OpenAI API"""
    
    def prep(self, shared):
        return shared["preprocessed_text"]
        
    def exec(self, text):
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
        
    def post(self, shared, prep_res, exec_res):
        shared["embedding"] = exec_res
        return "default"

class CreateSearchIndexNode(Node):
    """Node for creating a FAISS search index"""
    
    def prep(self, shared):
        return shared.get("embeddings", [])
        
    def exec(self, embeddings):
        if not embeddings:
            return None
            
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Create index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        return index
        
    def post(self, shared, prep_res, exec_res):
        if exec_res is not None:
            shared["search_index"] = exec_res
        return "default"

class SearchSimilarNode(Node):
    """Node for searching similar vectors"""
    
    def prep(self, shared):
        return {
            "index": shared.get("search_index"),
            "query_embedding": shared.get("query_embedding"),
            "k": shared.get("top_k", 5)
        }
        
    def exec(self, inputs):
        if not inputs["index"] or not inputs["query_embedding"]:
            return None
            
        query_array = np.array([inputs["query_embedding"]]).astype('float32')
        distances, indices = inputs["index"].search(query_array, inputs["k"])
        
        return {
            "distances": distances[0],
            "indices": indices[0]
        }
        
    def post(self, shared, prep_res, exec_res):
        if exec_res is not None:
            shared["search_results"] = exec_res
        return "default"

class BatchEmbeddingNode(Node):
    """Node for batch processing of embeddings"""
    
    def prep(self, shared):
        return {
            "texts": shared.get("texts", []),
            "batch_size": shared.get("batch_size", 100)
        }
        
    def exec(self, inputs):
        texts = inputs["texts"]
        batch_size = inputs["batch_size"]
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=batch
            )
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings
        
    def post(self, shared, prep_res, exec_res):
        shared["embeddings"] = exec_res
        return "default" 