"""AsyncParallelBatchNode implementation for article summarization."""

from pocketflow import AsyncParallelBatchNode, AsyncNode
from utils import call_llm_async, load_articles, save_summaries

class LoadArticles(AsyncNode):
    """Node that loads articles to process."""
    
    async def prep_async(self, shared):
        """Load articles from data directory."""
        print("\nLoading articles...")
        articles = await load_articles()
        return articles
    
    async def exec_async(self, articles):
        """No processing needed."""
        return articles
    
    async def post_async(self, shared, prep_res, exec_res):
        """Store articles in shared store."""
        shared["articles"] = exec_res
        print(f"Found {len(exec_res)} articles to process")
        return "process"

class ParallelSummarizer(AsyncParallelBatchNode):
    """Node that summarizes articles in parallel."""
    
    async def prep_async(self, shared):
        """Get articles from shared store."""
        print("\nProcessing in parallel...")
        return shared["articles"]
    
    async def exec_async(self, article):
        """Summarize a single article (called in parallel)."""
        summary = await call_llm_async(article)
        return summary
    
    async def post_async(self, shared, prep_res, summaries):
        """Store summaries and save to file."""
        shared["summaries"] = summaries
        
        print("\nSummaries generated:")
        for i, summary in enumerate(summaries, 1):
            print(f"{i}. {summary}")
        
        save_summaries(summaries)
        print("\nFinal report saved to: summaries.txt")
        return "default" 