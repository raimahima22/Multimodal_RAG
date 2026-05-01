from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.utils import pil_to_base64, pdf_to_images, get_pdf_page, clear_page_cache
import os
import torch
import time
import gc
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from groq import RateLimitError


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# How many retrieved pages to send to the LLM.
# 3 is the sweet spot for multimodal RAG:
#   - Enough context to handle multi-page answers
#   - Stays well under token limits
#   - Additional pages rarely change the answer meaningfully
MAX_IMAGES = 5

# LLM render resolution.  150 DPI is sufficient for text-heavy PDFs and cuts
# image token cost by ~75 % vs 300 DPI.  Bump to 200 only if you see LLM
# misreads on small-font tables.
RENDER_DPI = 150

# Longest edge (px) after resizing.  Groq/Llama-4 Scout handles 1120 px well;
# going higher adds tokens but no accuracy gain on document text.
MAX_EDGE_PX = 1120

# Minimum OCR text length (chars) to trust OCR and skip image for that page.
# Pages with rich OCR text are sent as text-only, saving image tokens.
OCR_TEXT_THRESHOLD = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

load_dotenv('/content/drive/MyDrive/.env')


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def create_llm(api_key: str) -> ChatGroq:
    return ChatGroq(
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        groq_api_key=api_key,
        temperature=0.2,
        max_tokens=1024,
    )


def resize_for_llm(img: Image.Image, max_edge: int = MAX_EDGE_PX) -> Image.Image:
    """
    Downscale image so its longest edge equals max_edge.
    Uses LANCZOS for quality.  No-ops if already small enough.
    """
    w, h = img.size
    longest = max(w, h)
    if longest <= max_edge:
        return img
    scale = max_edge / longest
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)


# ---------------------------------------------------------------------------
# API-key rotation wrapper
# ---------------------------------------------------------------------------

GROQ_KEYS = [
    os.environ.get("GROQ_API_KEY"),
    os.environ.get("GROQ_API_KEY2"),
    os.environ.get("GROQ_API_KEY3"),
]


class GroqLLMWrapper:
    def __init__(self, keys):
        self.keys = [k for k in keys if k]
        if not self.keys:
            raise ValueError("No GROQ API keys provided")
        self.current_key_index = 0
        self.llm = create_llm(self.keys[self.current_key_index])

    def switch_key(self) -> bool:
        self.current_key_index += 1
        if self.current_key_index >= len(self.keys):
            return False
        print(f"[GroqLLMWrapper] Switching to key {self.current_key_index + 1}")
        self.llm = create_llm(self.keys[self.current_key_index])
        return True

    def invoke(self, messages):
        last_error = None
        while self.current_key_index < len(self.keys):
            try:
                return self.llm.invoke(messages)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                print(f"[GroqLLMWrapper] Error with key {self.current_key_index + 1}: {e}")
                if any(x in err_str for x in [
                    "rate_limit", "429", "quota",
                    "timeout", "connection", "temporarily unavailable"
                ]):
                    print("[GroqLLMWrapper] Trying next API key...")
                    if not self.switch_key():
                        break
                else:
                    raise e
        raise RuntimeError(f"ALL_KEYS_FAILED: {last_error}")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class MultimodalGenerator:
    """
    Multimodal RAG answer generator.

    Strategy
    --------
    For each retrieved page we have two signals:
      1. OCR text  (stored in Qdrant payload)
      2. Page image (rendered on-the-fly from PDF)

    We send BOTH to the LLM, but smartly:
      - Text context is always included (cheap tokens, good for exact phrases).
      - Images are included for every page so the LLM can read tables/charts
        that OCR misses, but they are resized to MAX_EDGE_PX before encoding.
      - Only MAX_IMAGES pages are processed to cap total token spend.

    This hybrid approach outperforms image-only AND text-only strategies on
    insurance/financial documents where tables are critical but OCR is noisy.
    """

    def __init__(self):
        self.llm = GroqLLMWrapper(GROQ_KEYS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_answer(self, query: str, retrieved_points: list) -> str:
        start = time.time()

        pages = retrieved_points[:MAX_IMAGES]

        ocr_texts: list[str] = []
        image_messages: list[dict] = []

        for point in pages:
            source     = point.payload["source"]
            page_num   = point.payload.get("page_number", 0)
            ocr_text   = point.payload.get("ocr_text", "").strip()

            # ---- OCR text (always include) ----
            label = f"[Page {page_num} | {os.path.basename(source)}]"
            ocr_texts.append(f"{label}\n{ocr_text}" if ocr_text else label)

            # ---- Image (always include, but resized) ----
            if str(source).lower().endswith(".pdf"):
                page_img = get_pdf_page(source, page_num, dpi=RENDER_DPI)
            else:
                page_img = Image.open(source).convert("RGB")

            page_img = resize_for_llm(page_img)                 # ← key saving
            b64      = pil_to_base64(page_img)

            image_messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

            del page_img, b64

        combined_text = "\n\n---\n\n".join(ocr_texts)

        # ---- Build prompt ----
        prompt = self._build_prompt(query, combined_text)

        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            *image_messages,
        ])

        response = self.llm.invoke([message])

        # ---- Logging ----
        elapsed = time.time() - start
        self._log_usage(response, elapsed)

        aggressive_cleanup()
        return response.content

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, ocr_context: str) -> str:
        return f"""You are a professional document analyst.

CONTEXT (OCR text extracted from the relevant pages):
{ocr_context}

The page images are also attached — use them to verify tables, numbers, or anything the OCR may have garbled.

INSTRUCTIONS:
- Answer using ONLY information explicitly present in the context or images.
- Be concise and direct. No reasoning walk-throughs unless asked.
- Use bullet points only when they genuinely improve readability.
- Do NOT compare plans or infer shared rules unless clearly stated.

QUESTION:
{query}

If the answer cannot be found, respond with:
"Answer not found in provided documents."
"""

    @staticmethod
    def _log_usage(response, elapsed: float):
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            u = response.usage_metadata
            print(
                f"[Token usage] Input: {u.get('input_tokens', 'N/A')} | "
                f"Output: {u.get('output_tokens', 'N/A')} | "
                f"Total: {u.get('total_tokens', 'N/A')}"
            )
        else:
            print("[Token usage] Metadata not available.")
        print(f"[Generation time] {elapsed:.2f}s")