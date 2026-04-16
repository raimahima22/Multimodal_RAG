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
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    return np.asarray(x, dtype=np.float32)

class MultimodalRetriever:
    def __init__(self, indexer):
        self.indexer = indexer

    def _extract_text_embedding(self, query_text: str):
        """Extract multi-vector text embedding for ColQwen2.5"""
        start = time.time()
    
        inputs = self.indexer.processor.process_queries([query_text]).to(self.indexer.device)
    
        with torch.no_grad():
            outputs = self.indexer.model(**inputs)
        
            # ColQwen2.5 specific
            if hasattr(outputs, "query_embeds") and outputs.query_embeds is not None:
                embedding = to_numpy(outputs.query_embeds[0])
            elif hasattr(outputs, "query_embeddings") and outputs.query_embeddings is not None:
                embedding = to_numpy(outputs.query_embeddings[0])
            elif isinstance(outputs, torch.Tensor):
                embedding = to_numpy(outputs[0])
            else:
                embedding = to_numpy(outputs)  # last resort
        
        # L2 normalize each token vector (ColQwen2.5 expects this)
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embedding = embedding / norms
    
        embed_time = time.time() - start
        print(f"Query embedded in {embed_time:.3f}s | Tokens: {embedding.shape[0]}")
    
        del inputs, outputs
        aggressive_cleanup()
        return embedding, embed_time

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
            score_threshold=None,             # lowered a bit for ColQwen2.5
        ).points
        retr_time = time.time() - retr_start

        # Average score by number of query tokens (common practice)
        # for point in results:
        #     if hasattr(point, 'score') and point.score is not None:
        #         point.score = point.score / num_query_tokens

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