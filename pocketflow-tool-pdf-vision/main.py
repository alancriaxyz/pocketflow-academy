import os
from pathlib import Path
from flow import create_vision_flow

def list_pdfs(pdf_dir: str) -> list:
    """List all PDF files in the given directory"""
    pdf_files = []
    for file in os.listdir(pdf_dir):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(pdf_dir, file))
    return pdf_files

def main():
    # Get project root directory
    root_dir = Path(__file__).parent
    pdf_dir = root_dir / "pdfs"
    
    # Create flow
    flow = create_vision_flow()
    
    # Get list of PDFs
    pdf_files = list_pdfs(pdf_dir)
    if not pdf_files:
        print("No PDF files found in 'pdfs' directory!")
        return
        
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process each PDF
    for pdf_path in pdf_files:
        print(f"\nProcessing: {os.path.basename(pdf_path)}")
        print("-" * 50)
        
        # Run flow
        shared = {
            "pdf_path": pdf_path,
            "extraction_prompt": "Extract all text from this document, preserving formatting and layout.",
        }
        flow.run(shared)
        
        # Print results
        print("\nExtracted Text:")
        print("-" * 50)
        print(shared.get("final_text", "No text extracted"))

if __name__ == "__main__":
    main()
