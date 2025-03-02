import os
from pathlib import Path
from dotenv import load_dotenv
from flow import create_vision_flow

def main():
    # Load environment variables
    load_dotenv()
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Create flow
    flow = create_vision_flow()
    
    # Example usage
    shared = {
        "pdf_path": "example.pdf",  # Path to your PDF file
        "extraction_prompt": "Extract all text from this document, preserving formatting and layout.",
    }
    
    # Run flow
    flow.run(shared)
    
    # Print results
    print("\nExtracted Text:")
    print("-" * 50)
    print(shared.get("final_text", "No text extracted"))

if __name__ == "__main__":
    main()
