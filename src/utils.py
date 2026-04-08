import base64
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_path

def pdf_to_images(pdf_path):
    # Converts PDF pages to PIL Images
    return convert_from_path(pdf_path, dpi=100)

def pil_to_base64(image):
    # Encodes PIL image to base64 for LLM transmission
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# In src/utils.py
def load_image_or_pdf_page(source_path: str, page_num: int = None):
    """Universal loader for both images and PDFs"""
    if source_path.lower().endswith('.pdf'):
        images = pdf_to_images(source_path)
        return images[page_num] if page_num is not None else images[0]
    else:
        return Image.open(source_path)