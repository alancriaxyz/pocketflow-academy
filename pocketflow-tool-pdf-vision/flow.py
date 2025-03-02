from pocketflow import Flow
from nodes import LoadPDFNode, ExtractTextNode, CombineResultsNode

def create_vision_flow():
    """Create a flow for PDF processing with Vision API"""
    
    # Create nodes
    load_pdf = LoadPDFNode()
    extract_text = ExtractTextNode()
    combine_results = CombineResultsNode()
    
    # Connect nodes
    load_pdf >> extract_text >> combine_results
    
    # Create and return flow
    return Flow(start=load_pdf)
