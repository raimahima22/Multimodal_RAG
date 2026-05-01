import torch
import numpy as np
import gc
import re
import math
from collections import defaultdict
from qdrant_client.models import Filter, FieldCondition, MatchText

# ================= IMPROVED UTILS =================

def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    return np.asarray(x, dtype=np.float32)

def soft_tokenize(text: str):
    """Keep more context than basic tokenize. Don't strip too much."""
    return re.findall(r"[a-z0-9]+", text.lower())

def fuzzy_match_score(query_tokens, text):
    """Handles minor OCR glitches by checking partial overlaps."""
    text = text.lower()
    score = 0
    for token in query_tokens:
        if len(token) < 3: continue 
        if token in text:
            score += 1
        elif any(token[1:-1] in word for word in text.split()): # Middle-chunk match
            score += 0.5
    return score

# ================= THE OVERHAULED RETRIEVER =================

class MultimodalRetriever:
    def __init__(self, indexer):
        self.indexer = indexer

    def _extract_text_embedding(self, query_text):
        # CRITICAL: Do NOT normalize the query text here. Let the model see the full sentence.
        inputs = self.indexer.processor.process_queries([query_text]).to(self.indexer.device)
        
        with torch.no_grad():
            outputs = self.indexer.model(**inputs)
            # Handle different model output naming conventions
            emb = getattr(outputs, "query_embeds", 
                  getattr(outputs, "query_embeddings", outputs))
            if isinstance(emb, torch.Tensor):
                emb = emb[0]

        emb = to_numpy(emb)
        # Vector must be flattened if it's a patch-based embedding
        if len(emb.shape) > 1:
            emb = emb.mean(axis=0) 
            
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-8)

    def search(self, query_text, top_k=5, source_filter=None):
        # 1. Get Neural Hits (Intent)
        query_vec = self._extract_text_embedding(query_text).tolist()
        
        query_filter = None
        if source_filter:
            query_filter = Filter(must=[FieldCondition(key="source", match=MatchText(text=source_filter.lower()))])

        # Expand limit to ensure we don't miss the 'needle' in the initial vector sweep
        raw_results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_vec,
            using="image",
            query_filter=query_filter,
            limit=100, # Higher initial recall
        ).points

        # 2. Deduplicate by Page (Keep highest neural score per page)
        page_best = {}
        for p in raw_results:
            key = (p.payload.get("source"), p.payload.get("page_number"))
            if key not in page_best or p.score > page_best[key].score:
                page_best[key] = p
        
        candidates = sorted(page_best.values(), key=lambda x: x.score, reverse=True)[:20]
        
        return self._rerank_hybrid(query_text, candidates, top_k)

    def _rerank_hybrid(self, query, hits, top_k):
        """
        Uses a 'Cross-Validation' approach between Neural and Heuristic scores.
        """
        q_tokens = soft_tokenize(query)
        q_numbers = set(re.findall(r"\d+", query))
        
        scored_hits = []
        
        # Pre-calculate BM25 for the small candidate set
        corpus = [p.payload.get("ocr_text", "") for p in hits]
        bm25 = BM25()
        bm25.fit(corpus)

        for i, hit in enumerate(hits):
            text = hit.payload.get("ocr_text", "").lower()
            
            # --- COMPONENTS ---
            # 1. Neural Score (Normalized)
            s_neural = hit.score 
            
            # 2. Exact/Fuzzy Keyword match (Crucial for OCR)
            s_fuzzy = fuzzy_match_score(q_tokens, text) / (len(q_tokens) + 1)
            
            # 3. Number safety (If query has 2024, page MUST have 2024)
            found_nums = set(re.findall(r"\d+", text))
            s_num = 1.5 if q_numbers & found_nums else 0.0
            
            # 4. BM25 (Semantic keyword density)
            s_bm25 = bm25.score(q_tokens, i)
            
            # --- WEIGHTED FUSION ---
            # We give Neural the lead, but Keyword/Num scores act as 'Boosters'
            final_score = (s_neural * 2.0) + (s_fuzzy * 1.5) + (s_bm25 * 0.5) + s_num
            
            scored_hits.append((final_score, hit))

        # Sort by final fused score
        scored_hits.sort(key=lambda x: x[0], reverse=True)
        
        # Dynamic thresholding: If the top result is way better than others, trust it.
        # Otherwise, provide the top_k cluster.
        final_selection = []
        if scored_hits:
            top_score = scored_hits[0][0]
            for s, h in scored_hits[:top_k]:
                # If the score is within 40% of the top hit, it's likely relevant context
                if s > (top_score * 0.6):
                    final_selection.append(h)

        aggressive_cleanup()
        return final_selection