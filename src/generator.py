from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from src.utils import pil_to_base64, pdf_to_images
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


class MultimodalGenerator:
    def __init__(self):
        self.llm = ChatGroq(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",  # ✅ Current best vision model on Groq (2026)
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.3,
            max_tokens=1024,
        )

    def generate_answer(self, query, retrieved_point):
        source = retrieved_point.payload['source']
        page_num = retrieved_point.payload.get('page_number')

        # === Handle both PDF pages and direct images ===
        if str(source).lower().endswith('.pdf'):
            if page_num is None:
                raise ValueError(f"Page number missing for PDF: {source}")
            
            images = pdf_to_images(source)
            if page_num >= len(images):
                raise IndexError(f"Page {page_num} not found. Only {len(images)} pages available.")
            target_image = images[page_num]
        else:
            # Direct image file
            target_image = Image.open(source)

        b64_image = pil_to_base64(target_image)

        # Improved prompt
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": f"""You are an expert document analyst. 
Carefully examine the image and answer the following question:

Question: {query}"""
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                }
            ]
        )

        response = self.llm.invoke([message])
        return response.content