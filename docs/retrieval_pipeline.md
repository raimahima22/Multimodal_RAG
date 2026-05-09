
## 1. Purpose

The retrieval system combines dense visual retrieval with OCR-aware reranking to improve answer grounding.

---

## 2. Retrieval Stages

### Stage 1 — Dense Retrieval

Query embeddings are generated using ColQwen2.5 and searched against Qdrant.

---

### Stage 2 — OCR-Aware Reranking

Retrieved candidates are reranked using:

- BM25
- keyword overlap
- phrase overlap
- numeric overlap

---

## 3. Hybrid Scoring Formula

```text
final_score =
0.50 * embedding_similarity +
0.20 * BM25 +
0.15 * keyword_overlap +
0.10 * phrase_overlap +
0.05 * numeric_overlap
````

---

## 4. Page Aggregation

Multiple patches may originate from the same page.

The retriever keeps:

* the highest-scoring patch
* one representative result per page

This reduces duplicate retrieval.

```
```
