from pocketflow import Flow
from nodes import (
    TextPreprocessNode,
    GetEmbeddingNode,
    CreateSearchIndexNode,
    SearchSimilarNode,
    BatchEmbeddingNode,
    EmbeddingNode
)

def create_single_embedding_flow():
    """Create a flow for single text embedding"""
    # Create nodes
    preprocess = TextPreprocessNode()
    get_embedding = GetEmbeddingNode()
    
    # Connect nodes
    preprocess >> get_embedding
    
    # Create and return flow
    return Flow(start=preprocess)

def create_search_flow():
    """Create a flow for semantic search"""
    # Create nodes
    batch_embed = BatchEmbeddingNode()
    create_index = CreateSearchIndexNode()
    preprocess = TextPreprocessNode()
    get_embedding = GetEmbeddingNode()
    search = SearchSimilarNode()
    
    # Connect nodes for indexing
    batch_embed >> create_index
    
    # Connect nodes for query
    preprocess >> get_embedding
    
    # Create flow
    flow = Flow(start=batch_embed)
    
    return flow, preprocess, get_embedding, search

def create_batch_embedding_flow():
    """Create a flow for batch embedding"""
    # Create node
    batch_embed = BatchEmbeddingNode()
    
    # Create and return flow
    return Flow(start=batch_embed)

def create_embedding_flow():
    """Create a flow for text embedding"""
    # Create embedding node
    embedding = EmbeddingNode()
    
    # Create and return flow
    return Flow(start=embedding) 