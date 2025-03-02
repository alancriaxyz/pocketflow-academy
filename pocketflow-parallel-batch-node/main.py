import asyncio
from flow import create_flow

async def main():
    """Run the parallel processing flow."""
    # Create flow
    flow = create_flow()
    
    # Create shared store
    shared = {}
    
    # Run flow
    print("\nParallel Article Summarizer")
    print("-------------------------")
    await flow.run_async(shared)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 