from utils import (
    get_embedding,
    get_embeddings_batch,
    create_index,
    search_similar,
    calculate_similarity,
    preprocess_text
)

def demonstrate_basic_embedding():
    """Demonstrate basic embedding generation"""
    print("\n1. Basic Embedding Generation")
    print("-" * 50)
    
    text = "The quick brown fox jumps over the lazy dog"
    embedding = get_embedding(preprocess_text(text))
    
    print(f"Text: {text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

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
    
    # Get embeddings for all documents
    print("Getting embeddings for documents...")
    embeddings = get_embeddings_batch([preprocess_text(doc) for doc in documents])
    
    # Create search index
    print("Creating search index...")
    index = create_index(embeddings)
    
    # Search query
    query = "pets playing with toys"
    print(f"\nSearching for: '{query}'")
    
    # Get query embedding and search
    query_embedding = get_embedding(preprocess_text(query))
    distances, indices = search_similar(index, query_embedding, k=2)
    
    print("\nMost similar documents:")
    for i, (dist, idx) in enumerate(zip(distances, indices), 1):
        print(f"{i}. '{documents[idx]}' (distance: {dist:.3f})")

def demonstrate_document_similarity():
    """Demonstrate document similarity comparison"""
    print("\n3. Document Similarity")
    print("-" * 50)
    
    # Sample texts to compare
    text1 = "The weather is sunny and warm today"
    text2 = "It's a beautiful sunny day with clear skies"
    text3 = "The stock market showed significant gains"
    
    # Get embeddings
    embedding1 = get_embedding(preprocess_text(text1))
    embedding2 = get_embedding(preprocess_text(text2))
    embedding3 = get_embedding(preprocess_text(text3))
    
    # Calculate similarities
    sim1_2 = calculate_similarity(embedding1, embedding2)
    sim1_3 = calculate_similarity(embedding1, embedding3)
    
    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Text 3: '{text3}'")
    print(f"\nSimilarity between Text 1 and 2: {sim1_2:.3f}")
    print(f"Similarity between Text 1 and 3: {sim1_3:.3f}")

def demonstrate_batch_processing():
    """Demonstrate batch processing of texts"""
    print("\n4. Batch Processing")
    print("-" * 50)
    
    # Generate a larger set of texts
    texts = [f"Sample text number {i}" for i in range(1, 11)]
    
    print(f"Processing {len(texts)} texts in batch...")
    embeddings = get_embeddings_batch(texts, batch_size=5)
    
    print(f"Successfully generated {len(embeddings)} embeddings")
    print(f"Each embedding has dimension: {len(embeddings[0])}")

def main():
    print("Embeddings Tool Example")
    print("=" * 50)
    
    # Run demonstrations
    demonstrate_basic_embedding()
    demonstrate_semantic_search()
    demonstrate_document_similarity()
    demonstrate_batch_processing()

if __name__ == "__main__":
    main() 