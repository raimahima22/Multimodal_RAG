import torch
import numpy as np
import gc
import re
import math
from collections import defaultdict
from qdrant_client.models import Filter, FieldCondition, MatchText

VERBOSE = True
STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "and", "or"}

# ================= UTILS =================

def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    return np.asarray(x, dtype=np.float32)

def soft_tokenize(text: str):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]

def fuzzy_match_score(query_tokens, text):
    text = text.lower()
    score = 0
    for token in query_tokens:
        if len(token) < 3: continue 
        if token in text:
            score += 1
        elif len(token) > 4 and token[1:-1] in text: 
            score += 0.5
    return score

# ================= BM25 (For Hybrid Reranking) =================

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def fit(self, corpus):
        self.tokenized = [soft_tokenize(doc) for doc in corpus]
        self.doc_lens = [len(d) for d in self.tokenized]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.tokenized))
        df = defaultdict(int)
        self.freqs = []
        for doc in self.tokenized:
            freq = defaultdict(int)
            for t in doc: freq[t] += 1
            self.freqs.append(freq)
            for t in set(doc): df[t] += 1
        self.idf = {t: math.log((len(self.tokenized) - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}

    def score(self, query_tokens, i):
        score = 0.0
        freq = self.freqs[i]
        dl = self.doc_lens[i]
        for t in query_tokens:
            if t not in freq: continue
            f = freq[t]
            idf = self.idf.get(t, 0.0)
            score += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl)))
        return score

# ================= MULTIMODAL RETRIEVER (ColQwen2.5 Optimized) =================

class MultimodalRetriever:
    def __init__(self, indexer):
        self.indexer = indexer

    def _extract_query_embeddings(self, query_text: str):
        """
        MODIFIED: Returns the full Multi-Vector matrix required for ColPali/ColQwen.
        """
        inputs = self.indexer.processor.process_queries([query_text]).to(self.indexer.device)

        with torch.no_grad():
            outputs = self.indexer.model(**inputs)
            
            # ColQwen2.5 typically returns [Batch, Tokens, Dim]
            if hasattr(outputs, "query_embeds"):
                emb = outputs.query_embeds[0]
            else:
                emb = outputs[0]

        # Convert to list of lists for Qdrant multi-vector compatibility
        return to_numpy(emb).tolist()

    def search(self, query_text, top_k=5, source_filter=None):
        if VERBOSE: print(f"\n🔍 ColQwen Search: {query_text}")

        # 1. Multi-Vector Extraction
        query_multi_vec = self._extract_query_embeddings(query_text)
        
        query_filter = None
        if source_filter:
            query_filter = Filter(must=[
                FieldCondition(key="source", match=MatchText(text=source_filter.lower()))
            ])

        # 2. Recall phase (Using MaxSim logic defined in Indexer)
        raw_results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_multi_vec,
            using="image",
            query_filter=query_filter,
            limit=60, # Increased recall as MaxSim is computationally heavier but more accurate
        ).points

        # 3. Deduplicate by page (Highest patch score wins for the page)
        page_best = {}
        for p in raw_results:
            key = (p.payload.get("source"), p.payload.get("page_number"))
            if key not in page_best or p.score > page_best[key].score:
                page_best[key] = p
        
        candidates = sorted(page_best.values(), key=lambda x: x.score, reverse=True)[:20]
        
        return self._rerank_hybrid(query_text, candidates, top_k)

    def _rerank_hybrid(self, query, hits, top_k):
        q_tokens = soft_tokenize(query)
        q_numbers = set(re.findall(r"\d+", query))
        
        scored_hits = []
        corpus = [p.payload.get("ocr_text", "") for p in hits]
        
        bm25 = BM25()
        if corpus: bm25.fit(corpus)

        for i, hit in enumerate(hits):
            text = hit.payload.get("ocr_text", "").lower()
            
            # 1. Neural Score (This is already the MaxSim score from ColQwen)
            # ColQwen scores are usually higher than standard cosine, so we weight it heavily.
            s_neural = hit.score 
            
            # 2. Keyword Match
            s_fuzzy = fuzzy_match_score(q_tokens, text) / (len(q_tokens) + 1)
            
            # 3. Number match (Crucial for financial/tech data)
            found_nums = set(re.findall(r"\d+", text))
            s_num = 3.0 if q_numbers and (q_numbers & found_nums) else 0.0
            
            # 4. BM25
            s_bm25 = bm25.score(q_tokens, i) if corpus else 0.0
            
            # Fused Score calculation
            # We trust ColQwen's MaxSim score much more than the previous average embeddings.
            final_score = (s_neural * 1.2) + (s_fuzzy * 1.5) + (s_bm25 * 0.5) + s_num
            
            scored_hits.append((final_score, hit))

        scored_hits.sort(key=lambda x: x[0], reverse=True)
        
        # Dynamic thresholding
        final_selection = []
        if scored_hits:
            top_score = scored_hits[0][0]
            for s, h in scored_hits[:top_k]:
                if s > (top_score * 0.5): # Keep results within 50% of the top hit
                    final_selection.append(h)

        if VERBOSE:
            print(f" Reranked {len(hits)} down to {len(final_selection)} relevant pages.")

        aggressive_cleanup()
        return final_selection