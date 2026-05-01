import torch
import numpy as np
import gc
import re
import math
from collections import defaultdict

from qdrant_client.models import Filter, FieldCondition, MatchText


VERBOSE = True   # turn OFF during bulk evaluation


STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "in", "on", "at", "to",
    "for", "with", "and", "or", "it", "its", "this", "that", "how", "why",
    "when", "where", "who", "which", "do", "does", "did", "was", "were",
    "be", "been", "being", "has", "have", "had", "by", "from", "about",
    "under"
}


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def tokenize(text: str):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def normalize_query(text: str):
    return " ".join(tokenize(text))


def extract_numbers(text: str):
    return set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text))


# ================= BM25 =================

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def fit(self, corpus):
        self.tokenized = [tokenize(doc) for doc in corpus]
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
            if t not in freq:
                continue

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

    # -------- Query embedding --------
    def _extract_text_embedding(self, query_text):
        inputs = self.indexer.processor.process_queries([query_text]).to(self.indexer.device)

        with torch.no_grad():
            outputs = self.indexer.model(**inputs)

            if hasattr(outputs, "query_embeds"):
                emb = outputs.query_embeds[0]
            elif hasattr(outputs, "query_embeddings"):
                emb = outputs.query_embeddings[0]
            else:
                emb = outputs[0] if isinstance(outputs, torch.Tensor) else outputs

        emb = to_numpy(emb)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

        return emb

    # -------- Main search --------
    def search(self, query_text, top_k=5, source_filter=None):
        if VERBOSE: print(f"\n🔍 Query: {query_text}")

        # 1. Extract Multi-Vector for ColQwen
        # DO NOT NORMALIZE MANUALLY - ColQwen handles its own embedding space
        inputs = self.indexer.processor.process_queries([query_text]).to(self.indexer.device)
        with torch.no_grad():
            outputs = self.indexer.model(**inputs)
            # ColQwen2.5 returns [batch, tokens, dim]
            query_vec = to_numpy(outputs.query_embeds[0]).tolist()

        query_filter = None
        if source_filter:
            query_filter = Filter(must=[FieldCondition(key="source", match=MatchText(text=source_filter.lower()))])

        # 2. Get 60 candidate tiles
        raw_results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_vec,
            using="image",
            query_filter=query_filter,
            limit=60,
        ).points

        # 3. CRITICAL: Deduplicate by Page FIRST
        # This ensures we don't have 10 identical OCR blocks for the same page
        unique_pages = {}
        for p in raw_results:
            page_id = f"{p.payload.get('source')}_{p.payload.get('page_number')}"
            if page_id not in unique_pages or p.score > unique_pages[page_id].score:
                unique_pages[page_id] = p
        
        # Take the top unique candidates for reranking
        candidates = sorted(unique_pages.values(), key=lambda x: x.score, reverse=True)[:15]

        # 4. Rerank unique pages
        return self._rerank_hybrid(query_text, candidates, top_k)

    def _rerank_hybrid(self, query, hits, top_k):
        if not hits: return []
        
        q_tokens = tokenize(query)
        q_nums = extract_numbers(query)
        ocr_corpus = [p.payload.get("ocr_text", "") for p in hits]
        
        bm25 = BM25()
        bm25.fit(ocr_corpus)
        
        scored_hits = []
        for i, hit in enumerate(hits):
            text = ocr_corpus[i].lower()
            
            # Scores
            s_neural = hit.score  # MaxSim score from ColQwen
            s_bm25 = bm25.score(q_tokens, i)
            
            # Simple keyword match
            kw_match = sum(1 for t in q_tokens if t in text) / (len(q_tokens) + 1)
            
            # Number Match (Strong signal for financial/tech data)
            num_match = 2.0 if q_nums and (q_nums & extract_numbers(text)) else 0.0

            # Combined Score (Weights: Neural 50%, BM25 20%, Keywords 10%, Numbers 20%)
            final_score = (s_neural * 2.0) + (s_bm25 * 0.5) + (kw_match * 1.0) + num_match
            scored_hits.append((final_score, hit))

        scored_hits.sort(key=lambda x: x[0], reverse=True)
        return [h for s, h in scored_hits[:top_k]]