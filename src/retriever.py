import torch
import numpy as np
import gc
import time
from collections import defaultdict

from qdrant_client.models import Filter, FieldCondition, MatchText


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class MultimodalRetriever:
    def __init__(self, indexer):
        self.indexer = indexer

    def _extract_text_embedding(self, query_text: str):
        """Extract multi-vector text embedding for ColQwen2.5"""
        start = time.time()

        # === IMPORTANT: Use process_queries for ColQwen2.5 ===
        inputs = self.indexer.processor.process_queries(
            [query_text],          # Note: list of strings
            # return_tensors="pt",
            # padding=True,
            # truncation=True,
            # max_length=512,
        ).to(self.indexer.device)

        with torch.no_grad():
            outputs = self.indexer.model(**inputs)

            # ColQwen2.5 / colpali-engine usually returns .embeddings
            if hasattr(outputs, "embeddings") and outputs.embeddings is not None:
                embedding = outputs.embeddings[0]          # (num_tokens, embed_dim)
            elif hasattr(outputs, "text_embeds"):
                embedding = outputs.text_embeds[0]
            elif hasattr(outputs, "last_hidden_state"):
                embedding = outputs.last_hidden_state[0]
            else:
                raise ValueError(f"Unknown output format: {type(outputs)}")

            # Convert to numpy + L2 normalize each token vector
            embedding = embedding.cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embedding = embedding / norms

        embed_time = time.time() - start

        del inputs, outputs
        aggressive_cleanup()
        return embedding, embed_time   # shape: (num_tokens, dim)

    def search(self, query_text: str, top_k: int = 15, source_filter: str = None):
        start_search = time.time()
        
        # 1. Get multi-vector query embedding
        query_emb_array, embed_time = self._extract_text_embedding(query_text)
        num_query_tokens = query_emb_array.shape[0]
        
        query_multi_vec = query_emb_array.tolist()

        # 2. Optional source filter
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

        # 3. Query Qdrant (Multi-vector late interaction)
        retr_start = time.time()
        results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_multi_vec,
            using="image",                   # vector name used during indexing
            query_filter=query_filter,
            limit=top_k,
            score_threshold=0.3,             # lowered a bit for ColQwen2.5
        ).points
        retr_time = time.time() - retr_start

        # Average score by number of query tokens (common practice)
        for point in results:
            if hasattr(point, 'score') and point.score is not None:
                point.score = point.score / num_query_tokens

        # Group by page and keep best score per page
        page_best = defaultdict(lambda: {"score": -1.0, "point": None})
        for point in results:
            key = (point.payload.get("source"), point.payload.get("page_number"))
            if point.score > page_best[key]["score"]:
                page_best[key] = {"score": point.score, "point": point}

        sorted_pages = sorted(page_best.values(), key=lambda x: x["score"], reverse=True)
        top_results = [item["point"] for item in sorted_pages[:5]]

        total_time = time.time() - start_search
        print(f"Query embed: {embed_time:.3f}s | Retrieval: {retr_time:.3f}s | Total: {total_time:.3f}s")
        
        print("Retrieved pages:")
        for i, p in enumerate(top_results, 1):
            src = p.payload.get("source", "unknown")
            pg = p.payload.get("page_number", "?")
            score = p.score
            print(f"  {i}. {src} (Page {pg}) — score={score:.4f}")

        aggressive_cleanup()
        return top_results