Multimodal RAG System
Overview

This project implements a robust Multimodal Retrieval-Augmented Generation (RAG) pipeline capable of understanding, indexing, retrieving, and answering questions from multiple document modalities.

The system is designed to process:

PDFs
scanned documents
images
OCR-heavy files
mixed visual-text documents


The final system enables accurate, document-grounded responses from visually rich and text-heavy sources.

Features
Document Ingestion

Supports ingestion of:

PDF documents
scanned PDFs
images
OCR-based documents

Patch-Based Visual Chunking
Documents are segmented into smaller visual regions for improved retrieval granularity.

This improves:

localized retrieval
page-region understanding
multimodal relevance scoring
Multimodal Embeddings

Vector Database Integration

Stores embeddings inside Qdrant for efficient similarity search and scalable retrieval.

Hybrid Retrieval Pipeline

Combines multiple retrieval strategies, including:

semantic vector search
OCR-aware retrieval
multimodal similarity matching

Retrieved chunks are reranked using OCR-sensitive scoring logic to improve answer relevance.

Vision-Language Answer Generation

Uses a vision-language model to generate grounded answers from retrieved multimodal context.

Page-Level Aggregation

Aggregates retrieved chunks at the page level to improve contextual coherence during answer generation.

High-Level Architecture
                ┌─────────────────┐
                │   Documents     │
                │ PDFs / Images   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    Chunking  
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Visual Embedding│
                │  ColQwen2.5     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Qdrant VectorDB │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Hybrid Retrieval│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Vision-Language │
                │      LLM        │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Final Answer   │
                └─────────────────┘
Tech Stack

| Component               | Technology              |
| ----------------------- | ----------------------- |
| Embedding Model         | ColQwen2.5              |
| Vector Database         | Qdrant                  |
| OCR Engine              | Tesseract OCR           |
| LLM                     | Llama 4 Scout           |
| Framework               | LangChain               |
| PDF Processing          | PyMuPDF                 |
| Image Processing        | PIL / Pillow            |
| Deep Learning Framework | PyTorch                 |
| Retrieval Pipeline      | Hybrid Retrieval        |


Installation
Clone Repository
git clone <repository_url>
cd multimodal-rag-system
Create Virtual Environment
Linux / macOS
python -m venv venv
source venv/bin/activate
Windows
python -m venv venv
venv\Scripts\activate
Install Dependencies
pip install -r requirements.txt
Environment Variables

Create a .env file in the root directory:

OPENROUTER_API_KEY=your_api_key

Usage
Index Documents

Use the indexer pipeline to process and store multimodal document embeddings.

from src.indexer import MultimodalIndexer

indexer = MultimodalIndexer()

indexer.index_all_data("data")
Retrieve Relevant Chunks

Perform multimodal retrieval using semantic and OCR-aware search.

from src.retriever import MultimodalRetriever

retriever = MultimodalRetriever(indexer)

results = retriever.search(
    "What is the deductible?"
)
Generate Answers

Generate grounded answers using retrieved multimodal context.

from src.generator import MultimodalGenerator

generator = MultimodalGenerator()

answer = generator.generate_answer(
    query,
    results
)

print(answer)
