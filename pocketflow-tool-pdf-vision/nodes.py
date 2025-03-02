from pocketflow import Node
from tools.pdf import pdf_to_images
from tools.vision import extract_text_from_image
from typing import List, Dict, Any

class LoadPDFNode(Node):
    """Node for loading and converting PDF to images"""
    
    def prep(self, shared):
        return shared.get("pdf_path", "")
        
    def exec(self, pdf_path):
        return pdf_to_images(pdf_path)
        
    def post(self, shared, prep_res, exec_res):
        shared["page_images"] = exec_res
        return "default"

class ExtractTextNode(Node):
    """Node for extracting text from images using Vision API"""
    
    def prep(self, shared):
        return (
            shared.get("page_images", []),
            shared.get("extraction_prompt", None)
        )
        
    def exec(self, inputs):
        images, prompt = inputs
        results = []
        
        for img, page_num in images:
            text = extract_text_from_image(img, prompt)
            results.append({
                "page": page_num,
                "text": text
            })
            
        return results
        
    def post(self, shared, prep_res, exec_res):
        shared["extracted_text"] = exec_res
        return "default"

class CombineResultsNode(Node):
    """Node for combining and formatting extracted text"""
    
    def prep(self, shared):
        return shared.get("extracted_text", [])
        
    def exec(self, results):
        # Sort by page number
        sorted_results = sorted(results, key=lambda x: x["page"])
        
        # Combine text with page numbers
        combined = []
        for result in sorted_results:
            combined.append(f"=== Page {result['page']} ===\n{result['text']}\n")
            
        return "\n".join(combined)
        
    def post(self, shared, prep_res, exec_res):
        shared["final_text"] = exec_res
        return "default"
