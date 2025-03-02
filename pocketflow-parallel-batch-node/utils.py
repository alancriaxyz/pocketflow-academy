"""Utility functions for parallel processing."""

import os
import asyncio
import aiohttp
from openai import AsyncOpenAI
from tqdm import tqdm

# Semaphore to limit concurrent API calls
MAX_CONCURRENT_CALLS = 3
semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

async def call_llm_async(prompt):
    """Make async LLM call with rate limiting."""
    async with semaphore:  # Limit concurrent calls
        print(f"\nProcessing: {prompt[:50]}...")
        
        # Simulate API call with delay
        await asyncio.sleep(1)
        
        # Mock LLM response (in real app, would call OpenAI)
        summary = f"Summary of: {prompt[:30]}..."
        return summary

async def load_articles():
    """Load articles from data directory."""
    # For demo, generate mock articles
    articles = [
        "Article 1: AI advances in 2024...",
        "Article 2: New quantum computing breakthrough...",
        "Article 3: Latest developments in robotics..."
    ]
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save mock articles to files
    for i, content in enumerate(articles, 1):
        with open(os.path.join(data_dir, f"article{i}.txt"), "w") as f:
            f.write(content)
    
    return articles

def save_summaries(summaries):
    """Save summaries to output file."""
    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    with open(os.path.join(data_dir, "summaries.txt"), "w") as f:
        for i, summary in enumerate(summaries, 1):
            f.write(f"{i}. {summary}\n") 