import os
import asyncio
import numpy as np
from PIL import Image
from flow import create_flow

def get_image_paths():
    """Get paths of existing images in the images directory."""
    images_dir = "images"
    if not os.path.exists(images_dir):
        raise ValueError(f"Directory '{images_dir}' not found!")
    
    # List all jpg files in the images directory
    image_paths = []
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(images_dir, filename))
    
    if not image_paths:
        raise ValueError(f"No images found in '{images_dir}' directory!")
    
    print(f"\nFound {len(image_paths)} images:")
    for path in image_paths:
        print(f"- {path}")
    
    return image_paths

async def main():
    """Run the parallel image processing example."""
    print("\nParallel Image Processor")
    print("-" * 30)
    
    # Get existing image paths
    image_paths = get_image_paths()
    
    # Create shared store with image paths
    shared = {"images": image_paths}
    
    # Create and run flow
    flow = create_flow()
    
    await flow.run_async(shared)
    
    print("\nProcessing complete! Check the output/ directory for results.")

if __name__ == "__main__":
    asyncio.run(main()) 