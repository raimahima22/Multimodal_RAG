import torch
import numpy as np
import gc
import re
import math
from collections import defaultdict

from qdrant_client.models import Filter, FieldCondition, MatchText
from sentence_transformers import CrossEncoder

VERBOSE = True

# ================= CONFIG =================

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_TOP_K = 5   # rerank only top N pages

# ================= LOAD CROSS ENCODER =================

cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

# ================= STOPWORDS =================

STOPWORDS = {
    "what","is","are","the","a","an","of","in","on","at","to",
    "for","with","and","or","it","its","this","that","how","why",
    "when","where","who","which","do","does","did","was","were",
    "be","been","being","has","have","had","by","from","about","under"
}

# ================= UTILS =================

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

    # -------- MAIN SEARCH --------
    def search(self, query_text, top_k=1, source_filter=None):

        if VERBOSE:
            print("\n" + query_text + "\n")

        clean_query = normalize_query(query_text)
        q_tokens = tokenize(clean_query)
        query_nums = extract_numbers(query_text)

        emb = self._extract_text_embedding(clean_query)
        query_vec = emb.tolist()

        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[FieldCondition(
                    key="source",
                    match=MatchText(text=source_filter.lower())
                )]
            )

        results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_vec,
            using="image",
            query_filter=query_filter,
            limit=80,
        ).points

        for p in results:
            if p.score is not None:
                p.score /= emb.shape[0]

        # -------- PAGE AGGREGATION --------
        page_chunks = defaultdict(list)

        for p in results:
            key = (p.payload.get("source"), p.payload.get("page_number"))
            page_chunks[key].append(p)

        pages = []

        for key, chunks in page_chunks.items():
            scores = [c.score for c in chunks if c.score is not None]
            if not scores:
                continue

            agg_score = max(scores) + 0.3 * np.mean(scores)
            best_chunk = max(chunks, key=lambda x: x.score or 0)
            best_chunk.score = agg_score
            pages.append(best_chunk)

        pages = sorted(pages, key=lambda x: x.score, reverse=True)

        candidates = pages[:25]

        return self._rerank(query_text, clean_query, q_tokens, query_nums, candidates, top_k)

    # -------- RERANK --------
    def _rerank(self, raw_query, query, q_tokens, query_nums, hits, top_k):

        texts = [p.payload.get("ocr_text", "") for p in hits]
        emb_scores = [p.score for p in hits]

        # -------- HARD FILTER --------
        filtered = []
        for i, text in enumerate(texts):
            if len(text) < 40:
                continue
            if not any(w in text.lower() for w in q_tokens):
                continue
            filtered.append(i)

        if filtered:
            hits = [hits[i] for i in filtered]
            texts = [texts[i] for i in filtered]
            emb_scores = [emb_scores[i] for i in filtered]

        # -------- BM25 --------
        bm25 = BM25()
        bm25.fit(texts)
        bm25_scores = [bm25.score(q_tokens, i) for i in range(len(texts))]

        # -------- KEYWORD / NUMERIC --------
        keyword_scores = []

        for text in texts:
            t = text.lower()
            score = 0

            score += sum(2.0 for w in q_tokens if w in t)

            for n in (2, 3):
                for i in range(len(q_tokens) - n + 1):
                    phrase = " ".join(q_tokens[i:i+n])
                    if phrase in t:
                        score += n * 3.5

            nums = extract_numbers(t)
            overlap = len(query_nums & nums)
            score += overlap * 5.0

            keyword_scores.append(score)

        def norm(x):
            m = max(x) if x else 1.0
            return [i / (m + 1e-8) for i in x]

        bm25_norm = norm(bm25_scores)
        kw_norm = norm(keyword_scores)

        # -------- PRELIM SCORE --------
        prelim_scores = []

        for i in range(len(hits)):
            score = (
                0.65 * emb_scores[i] +
                0.20 * kw_norm[i] +
                0.15 * bm25_norm[i]
            )
            prelim_scores.append(score)

        ranked = sorted(
            list(enumerate(prelim_scores)),
            key=lambda x: x[1],
            reverse=True
        )

        # -------- CROSS-ENCODER FINAL STEP --------
        top_indices = [idx for idx, _ in ranked[:CROSS_TOP_K]]
        cross_inputs = [(raw_query, texts[i]) for i in top_indices]

        cross_scores = cross_encoder.predict(cross_inputs)

        final_candidates = []
        for i, idx in enumerate(top_indices):
            final_score = (
                0.7 * cross_scores[i] +
                0.3 * prelim_scores[idx]
            )
            final_candidates.append((idx, final_score))

        final_ranked = sorted(final_candidates, key=lambda x: x[1], reverse=True)

        if VERBOSE:
            print("\nFinal Ranking (with Cross-Encoder):\n")
            for i, (idx, score) in enumerate(final_ranked[:top_k], 1):
                page = hits[idx].payload.get("page_number")
                print(f"{i}. Page {page} | Final={score:.4f}")

        aggressive_cleanup()

        return [hits[idx] for idx, _ in final_ranked[:top_k]]