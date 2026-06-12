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
from groq import RateLimitError
import time

#load environment variables from .env file
load_dotenv('/content/drive/MyDrive/.env')
# load_dotenv()

def create_llm():
    """
    Create and return the language model instance.

    Current Model:
    meta-llama/llama-4-scout-17b-16e-instruct
    served through OpenRouter.

    Configuration:
    - temperature=0.2
        Lower temperature for more factual and stable outputs.

    - max_tokens=1024
        Maximum output token limit.

    Returns:
        ChatOpenAI
    """
    return ChatOpenAI(
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=1024,
    )

def aggressive_cleanup():
    """
    Clear python and CUDA memory
    """
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


class MultimodalGenerator:
    """
    Multimodal answer generation system using:
    
    - Retrieved document pages/images
    - Vision-language model (Llama 4 Scout)
    - Prompt-based grounded answering

    Pipeline:
    User Query
        ↓
    Retrieved Pages
        ↓
    Convert pages to images
        ↓
    Send images + prompt to VLM
        ↓
    Generate grounded answer

    Features:
    - Multi-page image reasoning
    - PDF page extraction
    - Token usage tracking
    - OpenRouter integration
    - OCR-ready architecture

    """
    def __init__(self):
        """
        Initialize the multimodal generator.
        """
        # self.llm = ChatGroq(
        #     model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        #     groq_api_key=os.environ.get("GROQ_API_KEY"),
        #     temperature=0.2,      # Lower for more factual answers
        #     max_tokens=1024,
        # )

        # self.llm = GroqLLMWrapper(GROQ_KEYS)
        self.llm = create_llm()
        # token usage tracking
        self.last_usage = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None
        }


    def generate_answer(self, query, retrieved_points):
        """
        Generate answer using retrieved multimodal documents.

        Steps:
        ------
        1. Load retrieved pages/images
        2. Convert pages into base64 image inputs
        3. Build multimodal prompt
        4. Invoke vision-language model
        5. Track token usage
        6. Return generated answer

        Args:
            query (str):
                User question.

            retrieved_points (List[ScoredPoint]):
                Retrieved Qdrant search results.

        Returns:
            str:
                Generated answer.
        """
        
        start_gen = time.time()
        # containers for retrieved content
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
        
        #convert images to OpenAI-compatible image messages
        image_messages = [
            {
                "type": "image_url",
                "image_url":{
                    "url": f"data:image/jpeg;base64,{pil_to_base64(img)}"
                }
            }
            for img in images[:3]
        ]

        #prompt construction
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
        -Documents are ranked by relevance (Rank 1 = most relevant)
        - Prefer higher ranked documents when answering
        - Use lower ranked documents if needed

        - Use bullet points only when they improve readability
        - Do NOT explain step-by-step unless asked
       
        
        
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

            self.last_usage = {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }

            print(f"Token Usage → Input: {self.last_usage['input_tokens']} | "
                  f"Output: {self.last_usage['output_tokens']} | "
                  f"Total: {self.last_usage['total_tokens']}")
        else:
            self.last_usage = {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None
            }

        print(f"Answer generation time: {gen_time:.2f} seconds")
        aggressive_cleanup()
        return response.content

      
        