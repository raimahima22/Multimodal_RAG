import torch
import numpy as np
import gc
import time
from collections import defaultdict
from src.utils import pdf_to_images
from PIL import Image
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

    def search(self, query_text: str, top_k: int = 15, source_filter: str = None, generator=None):
        start_search = time.time()
        
        # 1. Get query embedding
        query_emb_array, embed_time = self._extract_text_embedding(query_text)
        num_query_tokens = query_emb_array.shape[0]
        query_multi_vec = query_emb_array.tolist()

        # 2. Optional source filter
        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[FieldCondition(key="source", match=MatchText(text=source_filter.lower()))]
            )

        # 3. Query Qdrant
        results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_multi_vec,
            using="image",
            query_filter=query_filter,
            limit=top_k,
            score_threshold=None,
        ).points

        # Normalize score once
        for point in results:
            if hasattr(point, 'score') and point.score is not None:
                point.score = point.score / num_query_tokens

        # === IMPROVED GROUPING ===
        page_best = defaultdict(lambda: {
            "score": -1.0, 
            "point": None, 
            "all_scores": []
        })

        for point in results:
            key = (point.payload.get("source"), point.payload.get("page_number"))
            score = point.score if point.score is not None else 0.0
            
            page_best[key]["all_scores"].append(score)
            if score > page_best[key]["score"]:
                page_best[key]["score"] = score
                page_best[key]["point"] = point

        # Sort by best score per page
        sorted_pages = sorted(page_best.values(), key=lambda x: x["score"], reverse=True)

        # Take more candidates for reranking
        initial_hits = [item["point"] for item in sorted_pages[:10]]

        # Rerank
        if generator:
            final_hits = self.rerank_hits(query_text, initial_hits, generator, top_k=6)
        else:
            final_hits = initial_hits[:6]

        # Expand context (very important for split information)
        final_hits = self._expand_adjacent_context(final_hits, max_pages=7)

        print("\nFinal Retrieved Pages (with adjacent context):")
        for i, p in enumerate(final_hits, 1):
            src = p.payload.get("source", "unknown")
            pg = p.payload.get("page_number", "?")
            score = getattr(p, 'score', 0.0)
            print(f"{i}. {src} (Page {pg}) — score={score:.4f}")

        aggressive_cleanup()
        return final_hits
    def rerank_hits(self, query: str, hits, generator, top_k: int = 6):
        if not hits:
            return []

        query_words = set(word.lower() for word in query.split())
        scored = []

        for point in hits:
            emb_score = point.score or 0.0
            page_text = point.payload.get("text", "")
            text_lower = point.payload.get("text_lower", page_text.lower())

            if len(page_text) < 10:   # Very weak page
                scored.append((emb_score * 0.5, point))
                continue

            # 1. Keyword Match Score
            exact_matches = sum(1 for w in query_words if w in text_lower)
            partial_matches = sum(1 for w in query_words if any(w in token for token in text_lower.split()))

            keyword_score = exact_matches * 2.0 + partial_matches * 0.5

            # 2. Density Score (how concentrated the keywords are)
            words_in_page = len(text_lower.split())
            density_score = (exact_matches / max(1, words_in_page ** 0.4)) if words_in_page > 0 else 0

            # 3. Final Hybrid Score (Best combination)
            final_score = (
                emb_score * 0.60 +           # Visual similarity (ColQwen) - main signal
                keyword_score * 2.2 +        # Strong boost for keyword matches
                density_score * 12.0         # Reward pages where keywords are dense
            )

            scored.append((final_score, point))

        # Sort and return top results
        scored.sort(key=lambda x: x[0], reverse=True)
        final_hits = [p for _, p in scored[:top_k]]

        print(f" Hybrid Reranking: {len(hits)} candidates → {len(final_hits)} best pages")
        return final_hits
    
    def _expand_adjacent_context(self, hits, max_pages: int = 7):
        """Smart expansion: Always include neighboring pages when relevant info is found"""
        if not hits:
            return hits

        from collections import defaultdict
        source_pages = defaultdict(list)

        # Group by source document
        for point in hits:
            source = point.payload.get("source")
            page = point.payload.get("page_number")
            if source and page is not None:
                source_pages[source].append((page, point))

        expanded = []

        for source, page_list in source_pages.items():
            page_list.sort(key=lambda x: x[0])  # Sort by page number
            pages = [p[0] for p in page_list]
            points_map = {p[0]: p[1] for p in page_list}

            selected = set()

            for p_num in pages:
                # Add current + previous + next page
                for offset in [-1, 0, 1]:
                    adj = p_num + offset
                    if adj >= 0:
                        selected.add(adj)

            # Add selected pages in order
            for p_num in sorted(selected)[:max_pages]:
                if p_num in points_map:
                    expanded.append(points_map[p_num])
                else:
                    # Optional: You can create a dummy point or skip
                    pass

        # Remove duplicates while keeping order
        seen = set()
        unique = []
        for p in expanded:
            key = (p.payload.get("source"), p.payload.get("page_number"))
            if key not in seen:
                seen.add(key)
                unique.append(p)

        print(f"Context Expanded: {len(hits)} → {len(unique)} pages (adjacent included)")
        return unique[:max_pages]