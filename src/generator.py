from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from src.utils import pil_to_base64, pdf_to_images
import os
import time
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


class MultimodalGenerator:
    def __init__(self):
        self.llm = ChatGroq(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.2,      # Lowered slightly for more factual answers
            max_tokens=1024,
        )

    def generate_answer(self, query, retrieved_point):
        start_gen = time.time()
        source = retrieved_point.payload['source']
        page_num = retrieved_point.payload.get('page_number')

        if str(source).lower().endswith('.pdf'):
            if page_num is None:
                raise ValueError(f"Page number missing for PDF: {source}")
            images = pdf_to_images(source)
            target_image = images[page_num]
        else:
            target_image = Image.open(source)

        b64_image = pil_to_base64(target_image)

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""You are an expert document analyst. 
Carefully examine the provided image and answer the question accurately.
If the answer is not in the image, say so clearly.

Question: {query}"""
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                }
            ]
        )

        response = self.llm.invoke([message])
        gen_time = time.time() - start_gen
        print(f"Answer generation time: {gen_time:.2f} seconds")
        return response.content