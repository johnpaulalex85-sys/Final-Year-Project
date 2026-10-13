from transformers import pipeline
import cv2
import numpy as np

def test_depth():
    print("Loading model...")
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small")
    print("Model loaded.")
    
    # Create a dummy image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Convert to PIL
    from PIL import Image
    pil_img = Image.fromarray(img)
    
    print("Running inference...")
    result = pipe(pil_img)
    print("Inference successful.")
    depth = result["depth"]
    print("Depth size:", depth.size)
    print("Type:", type(depth))

if __name__ == "__main__":
    test_depth()
