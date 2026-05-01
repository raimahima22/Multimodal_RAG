import torch
import numpy as np
import gc
import re
import math
from collections import defaultdict
from qdrant_client.models import Filter, FieldCondition, MatchText

VERBOSE = True

# ================= STOPWORDS =================
STOPWORDS = {
     "the", "a", "an"
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


def minmax(x):
    if not x:
        return x
    mn, mx = min(x), max(x)
    if abs(mx - mn) < 1e-8:
        return [1.0 for _ in x]
    return [(v - mn) / (mx - mn + 1e-8) for v in x]


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
        freq = self.freqs[i]
        dl = self.doc_lens[i]
        score = 0.0

        for t in query_tokens:
            if t not in freq:
                continue
            f = freq[t]
            idf = self.idf.get(t, 0.0)

            score += idf * (f * (self.k1 + 1)) / (
                f + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl))
            )

        return score


# ================= RETRIEVER =================

class MultimodalRetriever:
    def __init__(self, indexer):
        self.indexer = indexer

    # -------- embedding --------
    def _extract_text_embedding(self, query_text):
        inputs = self.indexer.processor.process_queries([query_text]).to(self.indexer.device)

        with torch.no_grad():
            outputs = self.indexer.model(**inputs)

            if hasattr(outputs, "query_embeds"):
                emb = outputs.query_embeds[0]
            elif hasattr(outputs, "query_embeddings"):
                emb = outputs.query_embeddings[0]
            else:
                emb = outputs[0]

        emb = to_numpy(emb)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        return emb


    # ================= SEARCH =================

    def search(self, query_text, top_k=3, source_filter=None):

        print("\nQUERY:", query_text)

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

        # ================= QDRANT RETRIEVAL =================

        results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_vec,
            using="image",
            query_filter=query_filter,
            limit=60,
        ).points

        # normalize embedding score
        for p in results:
            if p.score is not None:
                p.score /= emb.shape[0]

        hits = sorted(results, key=lambda x: x.score, reverse=True)[:25]

        print(f"\nQdrant retrieval done | Candidates: {len(hits)}")

        # ================= RERANK =================

        ocr_texts = [h.payload.get("ocr_text", "") for h in hits]

        bm25 = BM25()
        bm25.fit(ocr_texts)

        bm25_scores = []
        kw_scores = []
        phrase_scores = []
        num_scores = []

        for text in ocr_texts:
            t = text.lower()

            kw = sum(1 for w in q_tokens if w in t)
            kw_scores.append(kw)

            phrase = 0
            for n in (2, 3):
                for i in range(len(q_tokens) - n + 1):
                    if " ".join(q_tokens[i:i+n]) in t:
                        phrase += 1
            phrase_scores.append(phrase)

            nums = extract_numbers(t)
            num_scores.append(len(nums & query_nums))

        for i in range(len(hits)):
            bm25_scores.append(bm25.score(q_tokens, i))

        emb_scores = [h.score for h in hits]

        # normalize 0–1
        emb_n = minmax(emb_scores)
        bm_n = minmax(bm25_scores)
        kw_n = minmax(kw_scores)
        ph_n = minmax(phrase_scores)
        nm_n = minmax(num_scores)

        # final score
        final_scores = []
        for i in range(len(hits)):
            score = (
                0.50 * emb_n[i] +
                0.20 * bm_n[i] +
                0.15 * kw_n[i] +
                0.10 * ph_n[i] +
                0.05 * nm_n[i]
            )
            final_scores.append(score)

        ranked = sorted(list(enumerate(final_scores)), key=lambda x: x[1], reverse=True)

        # ================= PAGE AGGREGATION (ONLY NOW) =================

        page_best = {}

        for idx, score in ranked:
            h = hits[idx]
            key = (h.payload["source"], h.payload["page_number"])

            if key not in page_best or score > page_best[key]["score"]:
                page_best[key] = {
                    "score": score,
                    "hit": h
                }

        final_pages = sorted(page_best.values(), key=lambda x: x["score"], reverse=True)

        # ================= PRINT FINAL =================

        print("\n================ FINAL TOP PAGES ================\n")

        for i, p in enumerate(final_pages[:top_k], 1):
            page = p["hit"].payload["page_number"]
            print(f"{i}. Page {page} | FINAL={p['score']:.5f}")

        aggressive_cleanup()

        return [p["hit"] for p in final_pages[:top_k]]