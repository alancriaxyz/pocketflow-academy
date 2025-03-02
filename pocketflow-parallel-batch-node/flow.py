"""AsyncFlow implementation for parallel article processing."""

from pocketflow import AsyncFlow, Node
from nodes import LoadArticles, ParallelSummarizer

class NoOp(Node):
    """Node that does nothing, used to properly end the flow."""
    pass

def create_flow():
    """Create and connect nodes into a flow."""
    
    # Create nodes
    loader = LoadArticles()
    summarizer = ParallelSummarizer()
    end = NoOp()
    
    # Connect nodes
    loader - "process" >> summarizer
    summarizer - "default" >> end  # Properly end the flow
    
    # Create flow starting with loader
    flow = AsyncFlow(start=loader)
    return flow 