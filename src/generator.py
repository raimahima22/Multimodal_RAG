from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.utils import pil_to_base64, pdf_to_images
import os
import torch
import time
import easyocr
import gc
import pytesseract
import numpy as np
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
def aggressive_cleanup():
    
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


class MultimodalGenerator:
    def __init__(self):
        self.llm = ChatGroq(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.2,      # Lower for more factual answers
            max_tokens=1024,
        )
        # self.llm = ChatOpenAI(
        #     model_name="qwen/qwen2.5-vl-72b-instruct",   # Official OpenRouter name
        #     openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        #     openai_api_base="https://openrouter.ai/api/v1",
        #     temperature=0.2,
        #     max_tokens=1024,
            
        # )

        self.reader = easyocr.Reader(['en'], gpu=True, model_storage_directory="easyocr_models")
    
    def _extract_text(self, image: Image.Image) -> str:
        image=image.convert("RGB")
        img = np.array(image)
        results = self.reader.readtext(img)

        texts = [r[1] for r in results]
        del img, results
        gc.collect()
    
        return "\n".join(texts)
    # def _extract_text(self, image: Image.Image) -> str:
    #     try:
    #         text = pytesseract.image_to_string(
    #             image.convert("RGB"),
    #             config='--psm 6'   # assume uniform block of text
    #         )
    #         return text.strip()
    #     finally:
    #         aggressive_cleanup()
    


    def generate_answer(self, query, retrieved_point):
        
        start_gen = time.time()
        source = retrieved_point.payload['source']
        page_num = retrieved_point.payload.get('page_number')
        retrieved_text = retrieved_point.payload.get('text', '')

        if str(source).lower().endswith('.pdf'):
            if page_num is None:
                raise ValueError(f"Page number missing for PDF: {source}")
            images = pdf_to_images(source)
            target_image = images[page_num]
        else:
            target_image = Image.open(source)

        #OCR text extraction
        extracted_text = self._extract_text(target_image)

        if not extracted_text.strip():
            extracted_text = "No readable text found in image."


        b64_image = pil_to_base64(target_image)

#         message = HumanMessage(
#             content=[
#                 {
#                     "type": "text",
#                     "text": f"""You are an expert document analyst. 
# You are given both an image of a document and an OCR extracted text from the same image.
# Use BOTH to answer accurately.

# Question: {query}"""
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
#                 }
#             ]
#         )
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""
You are an expert document analyst.

You are given:
1. An image of a document
2. OCR extracted text from the same image

Use BOTH to answer accurately.

-------------------
OCR TEXT:
{extracted_text}
-------------------

Question:
{query}

If the answer is not present, clearly say so.
"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_image}"
                    }
                }
            ]
        )

        response = self.llm.invoke([message])
        gen_time = time.time() - start_gen
        print(f"Answer generation time: {gen_time:.2f} seconds")
        aggressive_cleanup()
        return response.content

      
        