# Schema

# Qdrant Vector Schema

## Collection Name

mrag_collection

---

## Distance Metric

Cosine Similarity

---

## Vector Type

Multi-vector embeddings

---

## Payload Structure

```json
{
  "page_number": 0,
  "source": "document.pdf",
  "x": 0,
  "y": 512,
  "chunk_size": 512,
  "num_tokens": 143,
  "page_ocr": "...",
  "patch_ocr": "..."
}