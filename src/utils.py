import base64
from io import BytesIO
from functools import lru_cache
import fitz
from PIL import Image
import gc
from pdf2image import convert_from_path

RAG_DPI = 300
# Global page-level cache: (pdf_path, page_num) -> PIL Image
_page_cache = {}

def get_pdf_page(pdf_path: str, page_num: int, dpi: int = RAG_DPI) -> Image.Image:
    """Load a single page from a PDF, with caching."""
    key = (pdf_path, page_num)
    if key in _page_cache:
        return _page_cache[key]
    
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    
    _page_cache[key] = img
    return img

def clear_page_cache():
    """Call this after each query to free RAM."""
    _page_cache.clear()
    gc.collect()


def pdf_to_images(pdf_path):
    # Converts PDF pages to PIL Images
    return convert_from_path(pdf_path, dpi=RAG_DPI)

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