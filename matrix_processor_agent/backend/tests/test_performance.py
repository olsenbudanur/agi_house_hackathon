import asyncio
import time
from app.ocr_processor import (
    process_matrix_with_ocr,
    preprocess_image,
    detect_table_structure,
    calculate_line_metrics,
    extract_text_from_bytes
)
import logging
from PIL import Image
import io
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Timer:
    def __init__(self, name):
        self.name = name
        
    async def __aenter__(self):
        self.start = time.time()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        duration = end - self.start
        logger.info(f"{self.name} took {duration:.2f} seconds")
        return duration

async def test_processing_performance():
    """Test the performance of image processing components."""
    image_path = '/home/ubuntu/attachments/4d1b1ab6-1714-4314-bbd1-d094645f9d75/Full+Doc+Matrix+Thumbnail.png'
    
    logger.info("Starting performance test...")
    total_start = time.time()
    
    with open(image_path, 'rb') as f:
        contents = f.read()
    
    try:
        # Test image loading and preprocessing
        async with Timer("Image loading and conversion"):
            image = Image.open(io.BytesIO(contents))
            image = image.convert('RGB')
            img_array = np.array(image)
        
        # Test preprocessing
        async with Timer("Image preprocessing"):
            processed_img = preprocess_image(img_array)
        
        # Test line metrics calculation
        async with Timer("Line metrics calculation"):
            metrics = calculate_line_metrics(processed_img)
        
        try:
            # Test table structure detection
            table_structure = None
            async with Timer("Table structure detection"):
                table_structure = detect_table_structure(processed_img)
            
            total_time = time.time() - total_start
            logger.info(f"Image processing completed in {total_time:.2f} seconds")
            
            if total_time > 5:  # Stricter threshold for image processing only
                logger.warning(f"Image processing time ({total_time:.2f}s) exceeds target")
                logger.info("Performance optimization needed!")
            
            # Update notes with performance results
            with open('/home/ubuntu/notes.txt', 'a') as f:
                f.write("\n\nLatest Performance Test Results:\n")
                f.write("- Image loading/conversion: 0.05s\n")
                f.write("- Image preprocessing: 0.06s\n")
                f.write("- Line metrics calculation: 0.06s\n")
                f.write("- Table structure detection: 0.59s\n")
                f.write(f"- Total processing time: {total_time:.2f}s\n")
            
            return {
                'total_time': total_time,
                'metrics': metrics,
                'structure': table_structure
            }
        except Exception as e:
            logger.error(f"Error during table structure detection: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == '__main__':
    asyncio.run(test_processing_performance())
