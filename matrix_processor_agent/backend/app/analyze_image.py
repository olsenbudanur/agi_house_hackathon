from PIL import Image
import os
from pathlib import Path
import openai
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_matrix_image(image_path: str):
    """Analyze the matrix image and print its properties and OCR results."""
    try:
        import pytesseract
        import cv2
        import numpy as np
        from PIL import Image
        
        # Load image directly
        image = Image.open(image_path)
        if not image:
            print("Could not load image")
            return
        print(f"Image dimensions: {image.size}")
        print(f"Image mode: {image.mode}")
        print(f"Image format: {image.format}")
        
        # Save processed image
        output_dir = Path(os.path.dirname(image_path))
        processed_path = output_dir / "processed_matrix.png"
        image.save(processed_path)
        print(f"Saved processed image to: {processed_path}")
        
        # Convert PIL Image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Image preprocessing steps
        # 1. Convert to grayscale
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # 2. Thresholding
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # 3. Detect table structure
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
        
        horizontal_lines = cv2.erode(thresh, horizontal_kernel, iterations=1)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)
        
        vertical_lines = cv2.erode(thresh, vertical_kernel, iterations=1)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)
        
        # Combine lines
        table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Remove table structure
        clean_image = cv2.subtract(thresh, table_structure)
        
        # OCR on different versions
        print("\nOriginal Image OCR:")
        original_text = pytesseract.image_to_string(image)
        print(original_text[:500] + "...")
        
        print("\nThresholded Image OCR:")
        thresh_text = pytesseract.image_to_string(Image.fromarray(thresh))
        print(thresh_text[:500] + "...")
        
        print("\nCleaned Image OCR:")
        clean_text = pytesseract.image_to_string(Image.fromarray(clean_image))
        print(clean_text[:500] + "...")
        
    except Exception as e:
        print(f"Error analyzing image: {e}")

if __name__ == "__main__":
    image_path = "/home/ubuntu/attachments/4d1b1ab6-1714-4314-bbd1-d094645f9d75/Full+Doc+Matrix+Thumbnail.png"
    analyze_matrix_image(image_path)
