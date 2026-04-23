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
    
    def _ocr_page(self, point, generator) -> str:
        """OCR a page from a point's payload using generator's EasyOCR reader."""
        source = point.payload.get("source", "")
        page_num = point.payload.get("page_number", 0)
        try:
            if source.lower().endswith(".pdf"):
                pages = pdf_to_images(source)
                img = pages[page_num]
            else:
                img = Image.open(source)
            text = generator._extract_text(img)
            print(f"  OCR: Page {page_num} → {len(text)} chars extracted")
            return text
        except Exception as e:
            print(f"  OCR failed for page {page_num}: {e}")
            return ""


    def search(self, query_text: str, top_k: int = 15, source_filter: str = None, generator=None):
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
        for point in results:
            if hasattr(point, 'score') and point.score is not None:
                point.score = point.score / num_query_tokens


        # Group by page and keep best score per page
        page_best = defaultdict(lambda: {"score": -1.0, "point": None})
        for point in results:
            key = (point.payload.get("source"), point.payload.get("page_number"))
            if point.score > page_best[key]["score"]:
                page_best[key] = {"score": point.score, "point": point}


        # sorted_pages = sorted(page_best.values(), key=lambda x: x["score"], reverse=True)
        # top_results = [item["point"] for item in sorted_pages[:5]]
        sorted_pages = sorted(page_best.values(), key=lambda x: x["score"], reverse=True)


        # get more candidates before reranking
        initial_hits = [item["point"] for item in sorted_pages[:10]]

        print("\n Initial Candidates (before reranking):")
        for i, p in enumerate(initial_hits, 1):
            src = p.payload.get("source", "unknown")
            pg = p.payload.get("page_number", "?")
            score = p.score
            print(f"  {i}. {src} (Page {pg}) — score={score:.4f}")      
        # 5. Rerank (MANDATORY)
        if generator:
            final_hits = self.rerank_hits(query_text, initial_hits,generator, top_k=3)
        else:
            final_hits = initial_hits[:3]




        print("\n Top Retrieved Pages:")
        for i, p in enumerate(final_hits, 1):
            src = p.payload.get("source", "unknown")
            pg = p.payload.get("page_number", "?")
            score = p.score
            print(f"{i}. {src} (Page {pg}) — score={score:.4f}")


        aggressive_cleanup()
        return final_hits
    def rerank_hits(self, query: str, hits, generator, top_k: int = 3):
        if not hits:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())
        query_tokens = query_lower.split()
        scored = []

        for point in hits:
            emb_score = point.score or 0.0
            page_num = point.payload.get("page_number", "?")

            # Always OCR — payload text is empty
            page_text = self._ocr_page(point, generator)
            text_lower = page_text.lower()

            if len(page_text) < 10:
                print(f"  Page {page_num} | too short after OCR → penalized")
                scored.append((emb_score * 0.5, point))
                continue

            # 1. Exact keyword match score
            exact_matches = sum(1 for w in query_words if w in text_lower)

            # 2. Partial keyword match score
            partial_matches = sum(
                1 for w in query_words
                if any(w in token for token in text_lower.split())
            )

            keyword_score = exact_matches * 2.0 + partial_matches * 0.5

            # 3. Density score
            words_in_page = len(text_lower.split())
            density_score = (
                exact_matches / max(1, words_in_page ** 0.4)
                if words_in_page > 0 else 0
            )

            # 4. Phrase match bonus (2-word and 3-word phrases)
            phrase_bonus = 0.0
            for n in (2, 3):
                for i in range(len(query_tokens) - n + 1):
                    phrase = " ".join(query_tokens[i:i + n])
                    if phrase in text_lower:
                        phrase_bonus += n * 1.5

            # 5. Final hybrid score
            final_score = (
                emb_score   * 0.55 +
                keyword_score * 2.2 +
                density_score * 12.0 +
                phrase_bonus  * 1.5
            )

            print(
                f"  Page {page_num} | emb={emb_score:.4f} | "
                f"kw={keyword_score:.2f} | den={density_score:.3f} | "
                f"phrase={phrase_bonus:.2f} | final={final_score:.4f}"
            )

            scored.append((final_score, point))

        scored.sort(key=lambda x: x[0], reverse=True)
        final_hits = [p for _, p in scored[:top_k]]

        print(f"\n Reranking done: {len(hits)} candidates → {len(final_hits)} best pages")
        return final_hits