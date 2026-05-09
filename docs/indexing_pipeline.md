
## 1. Purpose

# Indexing Pipeline

The indexing pipeline converts documents into searchable multimodal embeddings and stores them inside Qdrant.

---

## 2. Supported Inputs

## Supported Formats

- PDF
- PNG
- JPG
- JPEG

---

## 3. Chunking Strategy

## Sliding Window Chunking

Each page is divided into overlapping image patches.

### Configuration:
- Chunk size: 512px
- Overlap: 160px
- Stride: 352px

This improves:
- layout preservation
- local semantic capture
- OCR accuracy

---

## 4. OCR Extraction

## OCR Strategy

Two OCR levels are stored:

### Page-Level OCR
Full-page textual context.

### Patch-Level OCR
Localized OCR extracted from each image patch.

Patch OCR is discarded if:
- text length < 20 characters

This reduces OCR noise.

---

## 5. Embedding Generation

## Embedding Model

The system uses **ColQwen2.5** multi-vector embeddings.

Each image patch generates:
- token-level embeddings
- multi-vector representations

These embeddings are stored using cosine similarity.

---

## 6. Qdrant Storage

## Vector Storage

Embeddings are stored in **Qdrant** using:

- multi-vector indexing
- cosine distance
- MAX_SIM comparator