# src/llm_agent.py

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from retrieval import retrieve
import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model_name="gpt-4")

PROMPT = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are given document passages extracted via ColPali visual retrieval.
Use the context to answer the question.

Context:
{context}

Question:
{query}
"""
)

def answer_query(query: str):
    hits = retrieve(query)
    context = "\n".join([f"Page: {f}" for f, _ in hits])
    completion = llm(PROMPT.format(context=context, query=query))
    return completion