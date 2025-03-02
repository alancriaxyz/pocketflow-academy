# PocketFlow Parallel Batch Node Example

This example demonstrates parallel processing using AsyncParallelBatchNode to summarize multiple news articles concurrently. It shows how to:
1. Process multiple items in parallel
2. Handle I/O-bound tasks efficiently
3. Manage rate limits with throttling

## What this Example Does

When you run the example:
1. It loads multiple news articles from a data directory
2. Processes them in parallel using AsyncParallelBatchNode
3. For each article:
   - Extracts key information
   - Generates a summary using an LLM
   - Saves the results
4. Combines all summaries into a final report

## How it Works

The example uses AsyncParallelBatchNode to process articles in parallel:

```python
class ParallelSummarizer(AsyncParallelBatchNode):
    async def prep_async(self, shared):
        # Return list of articles to process
        return shared["articles"]

    async def exec_async(self, article):
        # Process single article (called in parallel)
        summary = await call_llm_async(f"Summarize: {article}")
        return summary

    async def post_async(self, shared, prep_res, summaries):
        # Combine all summaries
        shared["summaries"] = summaries
        return "default"
```

Key features demonstrated:
- Parallel execution of `exec_async`
- Rate limiting with semaphores
- Error handling for failed requests
- Progress tracking for parallel tasks

## Project Structure
```
pocketflow-parallel-batch-node/
├── README.md
├── requirements.txt
├── data/
│   ├── article1.txt
│   ├── article2.txt
│   └── article3.txt
├── main.py
├── flow.py
├── nodes.py
└── utils.py
```

## Running the Example

```bash
# Install dependencies
pip install -r requirements.txt

# Run the example
python main.py
```

## Sample Output
```
Loading articles...
Found 3 articles to process

Processing in parallel...
[1/3] Processing article1.txt...
[2/3] Processing article2.txt...
[3/3] Processing article3.txt...

Summaries generated:
1. First article summary...
2. Second article summary...
3. Third article summary...

Final report saved to: summaries.txt
```

## Key Concepts

1. **Parallel Processing**
   - Using AsyncParallelBatchNode for concurrent execution
   - Managing parallel tasks efficiently

2. **Rate Limiting**
   - Using semaphores to control concurrent requests
   - Avoiding API rate limits

3. **Error Handling**
   - Graceful handling of failed requests
   - Retrying failed tasks

4. **Progress Tracking**
   - Monitoring parallel task progress
   - Providing user feedback 