from flow import (
    create_single_embedding_flow,
    create_search_flow,
    create_batch_embedding_flow
)

def demonstrate_single_embedding():
    """Demonstrate single text embedding"""
    print("\n1. Single Text Embedding")
    print("-" * 50)
    
    # Create flow
    flow = create_single_embedding_flow()
    
    # Prepare shared data
    shared = {"text": "The quick brown fox jumps over the lazy dog"}
    print(f"Input text: {shared['text']}")
    
    # Run flow
    flow.run(shared)
    
    # Show results
    print(f"Embedding dimension: {len(shared['embedding'])}")
    print(f"First 5 values: {shared['embedding'][:5]}")

def demonstrate_semantic_search():
    """Demonstrate semantic search with embeddings"""
    print("\n2. Semantic Search")
    print("-" * 50)
    
    # Sample documents
    documents = [
        "The cat sleeps on the windowsill in the sun",
        "A dog chases a ball in the park",
        "Birds sing in the trees at sunrise",
        "Fish swim in the clear blue ocean",
        "The cat plays with a ball of yarn",
        "Dogs bark at passing cars on the street",
    ]
    print("Documents to index:")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc}")
    
    # Create flow and nodes
    flow, preprocess, get_embedding, search = create_search_flow()
    
    # Index documents
    shared = {"texts": documents}
    flow.run(shared)
    
    # Prepare for search
    query = "pets playing with toys"
    print(f"\nSearching for: '{query}'")
    
    # Get query embedding
    shared["text"] = query
    preprocess.run(shared)
    get_embedding.run(shared)
    shared["query_embedding"] = shared["embedding"]
    shared["top_k"] = 2
    
    # Search
    search.run(shared)
    
    # Show results
    print("\nMost similar documents:")
    for i, (dist, idx) in enumerate(zip(
        shared["search_results"]["distances"],
        shared["search_results"]["indices"]
    ), 1):
        print(f"{i}. '{documents[idx]}' (distance: {dist:.3f})")

def demonstrate_batch_processing():
    """Demonstrate batch processing of texts"""
    print("\n3. Batch Processing")
    print("-" * 50)
    
    # Create flow
    flow = create_batch_embedding_flow()
    
    # Generate sample texts
    texts = [f"Sample text number {i}" for i in range(1, 11)]
    
    # Prepare shared data
    shared = {
        "texts": texts,
        "batch_size": 5
    }
    
    print(f"Processing {len(texts)} texts in batches of {shared['batch_size']}...")
    
    # Run flow
    flow.run(shared)
    
    # Show results
    print(f"Successfully generated {len(shared['embeddings'])} embeddings")
    print(f"Each embedding has dimension: {len(shared['embeddings'][0])}")

def main():
    print("Embeddings Tool Example with PocketFlow")
    print("=" * 50)
    
    # Run demonstrations
    demonstrate_single_embedding()
    demonstrate_semantic_search()
    demonstrate_batch_processing()

if __name__ == "__main__":
    main() 