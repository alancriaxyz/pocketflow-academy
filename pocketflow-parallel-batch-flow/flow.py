"""Flow definitions for parallel image processing."""

from pocketflow import AsyncFlow, AsyncParallelBatchFlow
from nodes import LoadImage, ApplyFilter, SaveImage, NoOp

def create_base_flow():
    """Create flow for processing a single image with one filter."""
    # Create nodes
    load = LoadImage()
    apply_filter = ApplyFilter()
    save = SaveImage()
    noop = NoOp()
    
    # Connect nodes
    load - "apply_filter" >> apply_filter
    apply_filter - "save" >> save
    save - "default" >> noop
    
    # Create flow
    return AsyncFlow(start=load)

class ImageBatchFlow(AsyncParallelBatchFlow):
    """Flow that processes multiple images with multiple filters in parallel."""
    
    async def prep_async(self, shared):
        """Generate parameters for each image-filter combination."""
        # Get list of images and filters
        images = shared.get("images", [])
        filters = ["grayscale", "blur", "sepia"]
        
        # Create parameter combinations
        params = []
        for image_path in images:
            for filter_type in filters:
                params.append({
                    "image_path": image_path,
                    "filter": filter_type
                })
        
        print(f"\nProcessing {len(images)} images with {len(filters)} filters...")
        print(f"Total combinations: {len(params)}")
        return params

def create_flow():
    """Create the complete parallel processing flow."""
    # Create base flow for single image processing
    base_flow = create_base_flow()
    
    # Wrap in parallel batch flow
    return ImageBatchFlow(start=base_flow) 