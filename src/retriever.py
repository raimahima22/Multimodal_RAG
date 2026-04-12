import torch
import numpy as np
import gc
import time
from collections import defaultdict
import pytesseract
from qdrant_client.models import Filter, FieldCondition, MatchText

def aggressive_cleanup():
    
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


class MultimodalRetriever:
    def __init__(self, indexer):
        self.indexer = indexer

    def _extract_text_embedding(self, query_text: str):
        """Extract MULTI-VECTOR text embeddings for ColPali / ColSmol late interaction"""
        start = time.time()
        
        inputs = self.indexer.processor(
            text=[query_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.indexer.device)

        with torch.no_grad():
            outputs = self.indexer.model(**inputs)

            # === ColPali / ColSmol family handling for MULTI-VECTOR ===
            if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
                # Most ColPali-style models return (batch, num_tokens, embed_dim)
                # We need all token embeddings, not just the first one
                embedding = outputs.text_embeds[0]   # shape: (num_tokens, embed_dim)

            elif hasattr(outputs, "last_hidden_state"):
                last_hidden = outputs.last_hidden_state[0]  # (seq_len, hidden_size)
                embedding = last_hidden  # Use all tokens (better for late interaction)

            elif isinstance(outputs, torch.Tensor):
                if outputs.dim() == 3:   # (batch, seq_len, dim) → all tokens
                    embedding = outputs[0]
                elif outputs.dim() == 2:
                    # Fallback: treat as single vector but wrap in list (not ideal)
                    embedding = outputs[0].unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected output tensor shape: {outputs.shape}")
            else:
                raise AttributeError(
                    f"Model output has neither 'text_embeds' nor 'last_hidden_state'. "
                    f"Got type: {type(outputs)}"
                )

            # Move to CPU and convert to list of numpy vectors
            embedding = embedding.cpu().numpy().astype(np.float32)

            # L2 normalize EACH vector individually (critical for MaxSim)
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embedding = embedding / norms

        embed_time = time.time() - start

        del inputs, outputs
        aggressive_cleanup()
        return embedding, embed_time   # Now returns (num_tokens, dim) array

    def search(self, query_text: str, top_k: int = 15, source_filter: str = None):
        start_search = time.time()
        
        # 1. Get multi-vector query embedding (list of token vectors)
        query_emb_array, embed_time = self._extract_text_embedding(query_text)
        num_query_tokens = query_emb_array.shape[0]
        
        # Convert to list of lists for Qdrant (multi-vector format)
        query_multi_vec = query_emb_array.tolist()   # list[list[float]]

        # 2. Build filter (unchanged)
        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchText(text=source_filter.lower())
                    )
                ]
            )

        retr_start = time.time()
        results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_multi_vec,           # ← Multi-vector: list of vectors (no outer list wrapper)
            using="image",                   # Must match the named multi-vector field in your collection
            query_filter=query_filter,
            limit=top_k,
            # Optional: enable rescoring / oversampling for better quality vs speed
            score_threshold=0.2,
        ).points
        retr_time = time.time() - retr_start

        for point in results:
            if hasattr(point, 'score') and point.score is not None:
                point.score = point.score / num_query_tokens

        # 3. Group best result per page (still excellent for page-level deduplication)
        page_best = defaultdict(lambda: {"score": -1.0, "point": None})
        for point in results:
            key = (point.payload.get("source"), point.payload.get("page_number"))
            if point.score > page_best[key]["score"]:
                page_best[key] = {"score": point.score, "point": point}

        sorted_pages = sorted(page_best.values(), key=lambda x: x["score"], reverse=True)
        top_results = [item["point"] for item in sorted_pages[:5]]

        total_time = time.time() - start_search
        print(f"Query embed (multi-vector): {embed_time:.3f}s | Retrieval: {retr_time:.3f}s | Total: {total_time:.3f}s")
        
        print("Retrieved pages (late-interaction MaxSim):")
        for i, p in enumerate(top_results, 1):
            src = p.payload.get("source", "unknown")
            pg = p.payload.get("page_number", "?")
            score = p.score
            print(f"  {i}. {src} (Page {pg}) — score={score:.4f}")
        
        aggressive_cleanup()

        return top_results