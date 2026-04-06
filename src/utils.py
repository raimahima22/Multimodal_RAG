# src/utils.py

import os
import base64
from PIL import Image
import pytesseract
import numpy as np

# -----------------------------
# 1. Image Utilities
# -----------------------------

def load_image(image_path: str) -> Image.Image:
    """
    Load an image from disk and convert to RGB.
    """
    return Image.open(image_path).convert("RGB")


def load_images_from_folder(folder_path: str):
    """
    Load all images from a directory.
    """
    images = []
    filenames = []

    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder_path, fname)
            images.append(load_image(path))
            filenames.append(fname)

    return images, filenames


# -----------------------------
# 2. OCR Utilities (Optional but useful)
# -----------------------------

def extract_text_from_image(image: Image.Image) -> str:
    """
    Extract text using Tesseract OCR.
    """
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"OCR failed: {e}")
        return ""


def extract_text_from_folder(folder_path: str):
    """
    Extract OCR text from all images in a folder.
    """
    texts = {}
    images, filenames = load_images_from_folder(folder_path)

    for img, fname in zip(images, filenames):
        texts[fname] = extract_text_from_image(img)

    return texts


# -----------------------------
# 3. Base64 Encoding (for multimodal LLMs)
# -----------------------------

def image_to_base64(image_path: str) -> str:
    """
    Convert image to base64 string (useful for multimodal APIs).
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# -----------------------------
# 4. ColPali Embedding Helpers
# -----------------------------

def normalize_embeddings(embeddings):
    """
    Normalize embeddings for cosine similarity.
    Works for both single-vector and multi-vector outputs.
    """
    embeddings = np.array(embeddings)

    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / (norms + 1e-10)


def flatten_multivector(embeddings):
    """
    Convert ColPali multi-vector embeddings into a flat list
    (useful for vector DB storage if needed).
    """
    if isinstance(embeddings, list):
        return [vec for sub in embeddings for vec in sub]
    return embeddings


def to_numpy(embeddings):
    """
    Ensure embeddings are numpy arrays.
    """
    if hasattr(embeddings, "cpu"):
        return embeddings.cpu().numpy()
    return np.array(embeddings)


# -----------------------------
# 5. Metadata Helpers
# -----------------------------

def build_metadata(filename: str, ocr_text: str = None):
    """
    Build metadata payload for vector DB.
    """
    return {
        "filename": filename,
        "text": ocr_text if ocr_text else ""
    }


# -----------------------------
# 6. Context Builder for RAG
# -----------------------------

def build_context_from_hits(hits, ocr_dict=None):
    """
    Convert retrieval results into LLM-ready context.
    """
    context_parts = []

    for fname, score in hits:
        text = ""
        if ocr_dict and fname in ocr_dict:
            text = ocr_dict[fname]

        context_parts.append(
            f"[Document: {fname} | Score: {score:.4f}]\n{text}"
        )

    return "\n\n".join(context_parts)


# -----------------------------
# 7. Debug / Logging Helpers
# -----------------------------

def print_retrieval_results(hits):
    """
    Pretty print retrieval results.
    """
    print("\nTop Results:")
    for fname, score in hits:
        print(f"{fname} → {score:.4f}")