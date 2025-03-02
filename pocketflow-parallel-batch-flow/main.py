"""Entry point for parallel image processing example."""

import os
import asyncio
import numpy as np
from PIL import Image
from flow import create_flow

def create_sample_images():
    """Create sample images if they don't exist."""
    # Create images directory if needed
    os.makedirs("images", exist_ok=True)
    
    # Sample image paths
    image_paths = [
        "images/cat.jpg",
        "images/dog.jpg",
        "images/bird.jpg"
    ]
    
    # Only create if they don't exist
    if all(os.path.exists(p) for p in image_paths):
        return image_paths
    
    print("\nCreating sample images...")
    
    # Image size
    size = (300, 200)
    
    # Create gradient image
    gradient = np.linspace(0, 255, size[0], dtype=np.uint8)
    gradient = np.tile(gradient, (size[1], 1))
    gradient_img = Image.fromarray(gradient)
    gradient_img.save(image_paths[0])
    print(f"Created: {image_paths[0]}")
    
    # Create checkerboard image
    checkerboard = np.zeros(size, dtype=np.uint8)
    checkerboard[::2, ::2] = 255
    checkerboard[1::2, 1::2] = 255
    checkerboard_img = Image.fromarray(checkerboard)
    checkerboard_img.save(image_paths[1])
    print(f"Created: {image_paths[1]}")
    
    # Create circles image
    circles = np.zeros(size, dtype=np.uint8)
    center_x, center_y = size[0] // 2, size[1] // 2
    for r in range(0, min(size) // 2, 20):
        for x in range(size[0]):
            for y in range(size[1]):
                if abs((x - center_x)**2 + (y - center_y)**2 - r**2) < 100:
                    circles[y, x] = 255
    circles_img = Image.fromarray(circles)
    circles_img.save(image_paths[2])
    print(f"Created: {image_paths[2]}")
    
    return image_paths

async def main():
    """Run the parallel image processing example."""
    print("\nParallel Image Processor")
    print("-" * 30)
    
    # Create sample images
    image_paths = create_sample_images()
    
    # Create shared store with image paths
    shared = {"images": image_paths}
    
    # Create and run flow
    flow = create_flow()
    await flow.run_async(shared)
    
    print("\nProcessing complete! Check the output/ directory for results.")

if __name__ == "__main__":
    asyncio.run(main()) 