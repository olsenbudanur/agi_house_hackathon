from PIL import Image, ImageDraw
import io

# Create a new image with white background
img = Image.new("RGB", (200, 50), "white")
draw = ImageDraw.Draw(img)

# Add some test text that includes numbers and percentages
draw.text((10, 10), "Testing OCR 740 85%", fill="black")

# Save the image
img.save("test.png")
