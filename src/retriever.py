import torch
import numpy as np
import gc
import re
import math
from collections import defaultdict

from qdrant_client.models import Filter, FieldCondition, MatchText


VERBOSE = True

STOPWORDS = {
    "what","is","are","the","a","an","of","in","on","at","to",
    "for","with","and","or","it","its","this","that","how","why",
    "when","where","who","which","do","does","did","was","were",
    "be","been","being","has","have","had","by","from","about","under"
}


# ================= UTIL =================

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


# ================= NEW SCORING =================

def answer_span_score(query_tokens, text):
    """Detect if answer-like span exists"""
    sentences = re.split(r'[.?!\n]', text.lower())
    best = 0

    for s in sentences:
        overlap = sum(1 for t in query_tokens if t in s)
        if overlap >= 2:
            best = max(best, overlap * 2)

    return best


def proximity_score(query_tokens, text):
    """Reward tokens appearing close together"""
    words = text.lower().split()
    positions = {t: [] for t in query_tokens}

    for i, w in enumerate(words):
        if w in positions:
            positions[w].append(i)

    all_pos = [p for v in positions.values() for p in v]
    if len(all_pos) < 2:
        return 0

    span = max(all_pos) - min(all_pos) + 1
    return len(all_pos) / (span + 1)


def length_penalty(text):
    return 1 / math.log(len(text.split()) + 10)


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

    # ================= SEARCH =================

    def search(self, query_text, top_k=5, source_filter=None):

        if VERBOSE:
            print("\n" + query_text + "\n")

        clean_query = normalize_query(query_text)
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
            limit=60,
        ).points

        # Normalize scores
        for p in results:
            if p.score is not None:
                p.score /= emb.shape[0]

        # Deduplicate pages
        page_best = {}
        for p in results:
            key = (p.payload.get("source"), p.payload.get("page_number"))
            if key not in page_best or p.score > page_best[key].score:
                page_best[key] = p

        results = sorted(page_best.values(), key=lambda x: x.score, reverse=True)

        hits = results[:15]

        return self._rerank(clean_query, hits, top_k)

    # ================= RERANK =================

    def _rerank(self, query, hits, top_k):

        ocr_texts = [p.payload.get("ocr_text", "") for p in hits]

        bm25 = BM25()
        bm25.fit(ocr_texts)

        q_tokens = tokenize(query)
        query_numbers = extract_numbers(query)

        scores = []

        for i, text in enumerate(ocr_texts):
            t = text.lower()

            emb_score = hits[i].score
            bm25_score = bm25.score(q_tokens, i)

            keyword_score = sum(1 for w in q_tokens if w in t)
            phrase_score = proximity_score(q_tokens, t)
            num_score = len(extract_numbers(t) & query_numbers)

            span_score = answer_span_score(q_tokens, t)
            len_pen = length_penalty(t)

            final = (
                0.45 * emb_score +
                0.20 * bm25_score * len_pen +
                0.10 * keyword_score +
                0.05 * phrase_score +
                0.05 * num_score +
                0.30 * span_score
            )

            scores.append(final)

        # Normalize
        max_s = max(scores) if scores else 1
        scores = [s / (max_s + 1e-8) for s in scores]

        ranked = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)

        # ================= DYNAMIC TOP-K =================
        selected = []

        if len(ranked) == 0:
            return []

        selected.append(ranked[0])

        if len(ranked) > 1:
            gap = scores[ranked[0]] - scores[ranked[1]]

            if gap < 0.15:
                selected.append(ranked[1])

        if len(ranked) > 2 and len(selected) < 2:
            selected.append(ranked[2])

        # fallback max
        selected = selected[:top_k]

        if VERBOSE:
            print("\nFinal Ranking:\n")
            for i in selected:
                page = hits[i].payload.get("page_number")
                print(f"Page {page} | Score={scores[i]:.4f}")

        aggressive_cleanup()

        return [hits[i] for i in selected]