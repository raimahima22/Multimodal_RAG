from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.utils import pil_to_base64, pdf_to_images, get_pdf_page, clear_page_cache
import os
import torch
import time
import easyocr
import gc
# import tiktoken
import numpy as np
from dotenv import load_dotenv
from PIL import Image

load_dotenv('/content/drive/MyDrive/.env')
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

        # self.reader = easyocr.Reader(['en'], gpu=True, model_storage_directory="easyocr_models")
        self.reader = easyocr.Reader(
            ['en'],
            gpu=torch.cuda.is_available(),
            model_storage_directory="easyocr_models"
        )

        self.pdf_cache = {}
    
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
    


    def generate_answer(self, query, retrieved_points):
        
        start_gen = time.time()
        # 
        images = []
        texts = []

        for point in retrieved_points:
            source = point.payload['source']
            page_num = point.payload.get('page_number', 0)

             # single page load instead of full PDF
            if str(source).lower().endswith('.pdf'):
                page_img = get_pdf_page(source, page_num, dpi=300)
            else:
                page_img = Image.open(source).convert("RGB")

            images.append(page_img)

            extracted_text = self._extract_text(page_img)
            texts.append(extracted_text)
        combined_text = "\n\n---\n\n".join(texts)

        image_messages = [
            {
                "type": "image_url",
                "image_url":{
                    "url": f"data:image/jpeg;base64,{pil_to_base64(img)}"
                }
            }
            for img in images[:3]
        ]
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""
        You are a professional document analyst. Please answer the user's question based on the provided context.
        Answer ONLY using explicitly stated information. Do NOT infer shared rules unless clearly stated.

        Guidelines:
        - Answer clearly, concisely and directly. Do NOT explain your reasoning process or compare different plans unless specifically asked to do so.
        - Be natural and professional
        - Use bullet points only when they improve readability
        - Do NOT explain step-by-step unless asked
       
        OCR TEXT:
        {combined_text}
        
        QUESTION:
        {query}

        If the answer is not present, say:
        "Answer not found in provided documents."
        """
               },
               *image_messages
            ]
        )

        response = self.llm.invoke([message])
        gen_time = time.time() - start_gen

        #token usage
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            print(f"Token Usage → Input: {usage.get('input_tokens', 'N/A')} | "
                  f"Output: {usage.get('output_tokens', 'N/A')} | "
                  f"Total: {usage.get('total_tokens', 'N/A')}")
        else:
            print("Token usage metadata not available.")

        print(f"Answer generation time: {gen_time:.2f} seconds")
        aggressive_cleanup()
        return response.content

      
        