from utils import (
    call_llm_cached,
    call_llm_with_history,
    call_llm_structured,
    manage_context_length,
    count_tokens
)

def demonstrate_basic_completion():
    """Demonstrate basic LLM completion with caching"""
    print("\n1. Basic LLM Completion")
    print("-" * 50)
    
    prompt = "What is the capital of France?"
    print(f"Prompt: {prompt}")
    
    # First call will hit the API
    response = call_llm_cached(prompt)
    print(f"Response: {response}")
    
    # Second call will use cache
    print("\nCalling again (should use cache)...")
    response = call_llm_cached(prompt)
    print(f"Response: {response}")

def demonstrate_chat_history():
    """Demonstrate chat with history"""
    print("\n2. Chat with History")
    print("-" * 50)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi, I'd like to learn about Python."},
        {"role": "assistant", "content": "I'd be happy to help you learn Python! What specific aspect would you like to know about?"},
        {"role": "user", "content": "What are the basic data types?"}
    ]
    
    print("Chat history:")
    for msg in messages:
        print(f"{msg['role'].capitalize()}: {msg['content']}")
    
    response = call_llm_with_history(messages)
    print(f"\nAssistant: {response}")

def demonstrate_structured_output():
    """Demonstrate structured output generation"""
    print("\n3. Structured Output")
    print("-" * 50)
    
    prompt = "Analyze the sentiment of this text: I really love this product, it's amazing!"
    format_instructions = """
Output in YAML format:
```yaml
sentiment: positive/negative/neutral
confidence: float between 0 and 1
explanation: reason for the sentiment
```"""
    
    print(f"Prompt: {prompt}")
    print(f"Format instructions: {format_instructions}")
    
    response = call_llm_structured(prompt, format_instructions)
    print(f"\nStructured response:\n{response}")

def demonstrate_context_management():
    """Demonstrate context length management"""
    print("\n4. Context Management")
    print("-" * 50)
    
    # Create a long conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Let's have a long conversation about the history of computing."},
        {"role": "assistant", "content": "I'd be happy to discuss the history of computing! Where would you like to start?"},
        {"role": "user", "content": "Tell me about the first computers."},
        {"role": "assistant", "content": "The history of computers begins with early mechanical calculators..." + "." * 1000},  # Long response
        {"role": "user", "content": "That's interesting! What came next?"}
    ]
    
    print(f"Original conversation length: {len(messages)} messages")
    total_tokens = sum(count_tokens(msg["content"]) for msg in messages)
    print(f"Total tokens: {total_tokens}")
    
    # Manage context length
    max_tokens = 2000
    truncated_messages = manage_context_length(messages.copy(), max_tokens)
    
    print(f"\nAfter truncation (max {max_tokens} tokens):")
    print(f"Messages remaining: {len(truncated_messages)}")
    total_tokens = sum(count_tokens(msg["content"]) for msg in truncated_messages)
    print(f"Total tokens: {total_tokens}")

def main():
    print("LLM Wrapper Example")
    print("=" * 50)
    
    # Run demonstrations
    demonstrate_basic_completion()
    demonstrate_chat_history()
    demonstrate_structured_output()
    demonstrate_context_management()

if __name__ == "__main__":
    main() 