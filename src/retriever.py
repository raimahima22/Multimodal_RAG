import torch
import numpy as np
import gc
import re
import math
from collections import defaultdict
from qdrant_client.models import Filter, FieldCondition, MatchText

VERBOSE = True

# We keep a lighter stopword list to ensure "How" or "Under" 
# doesn't get nuked before the embedding model sees it.
STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "and", "or"}

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
    """
    FIX: Added missing function.
    Preserves alphanumeric strings and basic punctuation 
    to help neural models maintain context.
    """
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]

def fuzzy_match_score(query_tokens, text):
    """Handles minor OCR glitches by checking partial overlaps."""
    text = text.lower()
    score = 0
    for token in query_tokens:
        if len(token) < 3: continue 
        if token in text:
            score += 1
        # Middle-chunk match for OCR errors (e.g., 'prof1t' for 'profit')
        elif len(token) > 4 and token[1:-1] in text: 
            score += 0.5
    return score

# ================= BM25 =================

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
            for t in doc:
                freq[t] += 1
            self.freqs.append(freq)
            for t in set(doc):
                df[t] += 1

        self.idf = {
            t: math.log((len(self.tokenized) - n + 0.5) / (n + 0.5) + 1)
            for t, n in df.items()
        }

    def score(self, query_tokens, i):
        score = 0.0
        freq = self.freqs[i]
        dl = self.doc_lens[i]
        for t in query_tokens:
            if t not in freq: continue
            f = freq[t]
            idf = self.idf.get(t, 0.0)
            num = f * (self.k1 + 1)
            den = f + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl))
            score += idf * num / den
        return score

# ================= RETRIEVER =================

class MultimodalRetriever:
    def __init__(self, indexer):
        self.indexer = indexer

    def _extract_text_embedding(self, query_text):
        # We process the query AS IS (no pre-tokenization) 
        # so the model sees the full sentence structure.
        inputs = self.indexer.processor.process_queries([query_text]).to(self.indexer.device)

        with torch.no_grad():
            outputs = self.indexer.model(**inputs)
            # Support for different ColPali/SigLIP output formats
            if hasattr(outputs, "query_embeds"):
                emb = outputs.query_embeds[0]
            elif hasattr(outputs, "query_embeddings"):
                emb = outputs.query_embeddings[0]
            else:
                emb = outputs[0] if isinstance(outputs, torch.Tensor) else outputs

        emb = to_numpy(emb)
        # Ensure it's a 1D vector for Qdrant
        if len(emb.shape) > 1:
            emb = np.mean(emb, axis=0)
            
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def search(self, query_text, top_k=5, source_filter=None):
        if VERBOSE: print(f"\n🔍 Query: {query_text}")

        query_vec = self._extract_text_embedding(query_text).tolist()
        
        query_filter = None
        if source_filter:
            query_filter = Filter(must=[FieldCondition(key="source", match=MatchText(text=source_filter.lower()))])

        # Recall phase
        raw_results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_vec,
            using="image",
            query_filter=query_filter,
            limit=100, 
        ).points

        # Deduplicate
        page_best = {}
        for p in raw_results:
            key = (p.payload.get("source"), p.payload.get("page_number"))
            if key not in page_best or p.score > page_best[key].score:
                page_best[key] = p
        
        candidates = sorted(page_best.values(), key=lambda x: x.score, reverse=True)[:25]
        
        return self._rerank_hybrid(query_text, candidates, top_k)

    def _rerank_hybrid(self, query, hits, top_k):
        q_tokens = soft_tokenize(query)
        q_numbers = set(re.findall(r"\d+", query))
        
        scored_hits = []
        corpus = [p.payload.get("ocr_text", "") for p in hits]
        
        bm25 = BM25()
        bm25.fit(corpus)

        for i, hit in enumerate(hits):
            text = hit.payload.get("ocr_text", "").lower()
            
            # 1. Neural Score
            s_neural = hit.score 
            
            # 2. Keyword/Fuzzy match
            s_fuzzy = fuzzy_match_score(q_tokens, text) / (len(q_tokens) + 1)
            
            # 3. Number match (Weight increased: missing a year is a dealbreaker)
            found_nums = set(re.findall(r"\d+", text))
            s_num = 2.0 if q_numbers and (q_numbers & found_nums) else 0.0
            
            # 4. BM25
            s_bm25 = bm25.score(q_tokens, i)
            
            # Fused Score calculation
            # Weights: Neural (40%), Fuzzy (30%), BM25 (10%), Numbers (20% boost)
            final_score = (s_neural * 2.5) + (s_fuzzy * 1.5) + (s_bm25 * 0.5) + s_num
            
            scored_hits.append((final_score, hit))

        scored_hits.sort(key=lambda x: x[0], reverse=True)
        
        # Dynamic Top-K Selection
        final_selection = []
        if scored_hits:
            top_score = scored_hits[0][0]
            for s, h in scored_hits[:top_k]:
                # Only keep if score is competitive (avoids adding 'noise' pages)
                if s > (top_score * 0.55):
                    final_selection.append(h)

        if VERBOSE:
            print(f"✅ Reranked {len(hits)} down to {len(final_selection)} relevant pages.")

        aggressive_cleanup()
        return final_selection