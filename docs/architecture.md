# Architecture

## Overview

The system is a Multimodal Retrieval-Augmented Generation (RAG) pipeline that processes PDFs, scanned documents, images, and OCR-heavy files to answer questions using retrieved document context.

It is designed to handle visually rich and text-heavy documents and generate accurate, grounded responses.

---

## Pipeline Flow

```text
Documents
   ↓
PDF to Image (pdf2image)
   ↓
Chunking (Visual Patches)
   ↓
Embeddings (ColQwen2.5)
   ↓
Vector Store (Qdrant)
   ↓
Hybrid Retrieval
   ↓
Vision-Language LLM
   ↓
Final Answer
```

---

## Components

## Document Processing

- Converts PDFs into images using `pdf2image`
- Extracts text from images using OCR (Tesseract)

---

## Chunking

- Splits document pages into smaller visual regions (patches)
- Improves retrieval precision at region level

---

## Embeddings

- Uses **ColQwen2.5**
- Generates multimodal embeddings from:
  - text
  - images
  - OCR output

---

## Vector Database

- Uses **Qdrant**
- Stores embeddings for fast similarity search

---

## Hybrid Retrieval

- Combines:
  - semantic vector search
  - OCR-aware retrieval
- Returns most relevant document chunks

---

## Generation

- Uses a Vision-Language LLM
- Generates final answer using retrieved context

---

## Goal

To provide accurate, document-grounded answers from multimodal inputs including scanned and image-based documents.
```