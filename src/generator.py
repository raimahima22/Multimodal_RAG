from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.utils import pil_to_base64, pdf_to_images
import os
import torch
import time
import easyocr
import gc
# import pytesseract
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
            page_num = point.payload.get('page_number')

            if str(source).lower().endswith('.pdf'):
                pages = pdf_to_images(source)
                page_img = imgs[page_num]
            else:
                page_img = Image.open(source)

            images.append(page_img)

            extracted_text = self._extract_text(img)
            texts.append(extracted_text)
        combined_text = "\n\n---\n\n".join(texts[:5])


        # b64_image = pil_to_base64(target_image)

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
        image_messages = [
            {
                "type": "image_url",
                "image_url":{
                    "url": f"data:image/jpeg;base64,{pil_to_base64(img)}"
                }
            }
            for img in images[:5]
        ]
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""
        Answer the question using the provided context.

        Rules:
        - Be concise and direct
        - Give only the final answer
        - Use bullet points ONLY if needed
        - Do NOT explain step-by-step unless asked
        - Do NOT repeat the question

        -------------------
        CONTEXT:
        {combined_text}
        -------------------

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
        print(f"Answer generation time: {gen_time:.2f} seconds")
        aggressive_cleanup()
        return response.content

      
        