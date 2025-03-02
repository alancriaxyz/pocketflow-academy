import os
from functools import lru_cache
from typing import List, Dict, Any, Optional
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: Input text
        model: Model name to use for tokenization
        
    Returns:
        Number of tokens
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

@lru_cache(maxsize=1000)
def call_llm_cached(prompt: str, 
                    model: str = "gpt-4",
                    temperature: float = 0.7,
                    max_tokens: Optional[int] = None) -> str:
    """
    Call LLM with caching to avoid duplicate calls.
    
    Args:
        prompt: Input prompt
        model: Model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        
    Returns:
        LLM response text
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def call_llm_with_history(messages: List[Dict[str, str]],
                         model: str = "gpt-4",
                         temperature: float = 0.7,
                         max_tokens: Optional[int] = None) -> str:
    """
    Call LLM with chat history.
    
    Args:
        messages: List of message dictionaries with role and content
        model: Model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        
    Returns:
        LLM response text
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def call_llm_structured(prompt: str,
                       format_instructions: str,
                       model: str = "gpt-4",
                       temperature: float = 0.7) -> str:
    """
    Call LLM with structured output format.
    
    Args:
        prompt: Input prompt
        format_instructions: Instructions for output format
        model: Model to use
        temperature: Sampling temperature
        
    Returns:
        Structured LLM response
    """
    full_prompt = f"{prompt}\n\n{format_instructions}"
    return call_llm_cached(full_prompt, model, temperature)

def manage_context_length(messages: List[Dict[str, str]], 
                         max_tokens: int = 4000,
                         model: str = "gpt-4") -> List[Dict[str, str]]:
    """
    Manage context length by removing old messages if needed.
    
    Args:
        messages: List of message dictionaries
        max_tokens: Maximum allowed tokens
        model: Model name for token counting
        
    Returns:
        Truncated message list
    """
    while messages:
        total_tokens = sum(count_tokens(msg["content"], model) for msg in messages)
        if total_tokens <= max_tokens:
            break
        # Remove oldest message (after system message if present)
        start_idx = 1 if messages[0]["role"] == "system" else 0
        messages.pop(start_idx)
    return messages 