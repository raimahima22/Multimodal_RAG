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

def soft_tokenize(text: str):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]

def fuzzy_match_score(query_tokens, text):
    text = text.lower()
    score = 0
    for token in query_tokens:
        if len(token) < 3:
            continue
        if token in text:
            score += 1
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

    def _extract_query_embeddings(self, query_text):
        inputs = self.indexer.processor.process_queries([query_text]).to(self.indexer.device)

        with torch.no_grad():
            outputs = self.indexer.model(**inputs)

            if hasattr(outputs, "query_embeds"):
                emb = outputs.query_embeds[0]
            else:
                emb = outputs[0]

        return emb.tolist()

    # ================= MAIN SEARCH =================

    def search(self, query_text, top_k=10, source_filter=None):

        print(f"\n[{1}/{top_k}] {query_text}...")

        query_vec = self._extract_query_embeddings(query_text)

        query_filter = None
        if source_filter:
            query_filter = Filter(must=[
                FieldCondition(
                    key="source",
                    match=MatchText(text=source_filter.lower())
                )
            ])

        results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_vec,
            using="image",
            query_filter=query_filter,
            limit=60,
        ).points

        print(f"Qdrant retrieval done | Candidates: {len(results)}\n")

        # ================= PAGE GROUPING =================
        page_best = {}

        for p in results:
            key = (p.payload.get("source"), p.payload.get("page_number"))
            if key not in page_best or p.score > page_best[key].score:
                page_best[key] = p

        pages = sorted(page_best.values(), key=lambda x: x.score, reverse=True)

        print("\n==================Initial Ranking===============\n")
        for i, p in enumerate(pages[:10], 1):
            print(f"{i}. Page {p.payload.get('page_number')} | score={p.score:.5f}")

        candidates = pages[:20]

        return self._rerank_debug(query_text, candidates, top_k)

    # ================= RERANK (WITH FULL DEBUG) =================

    def _rerank_debug(self, query, hits, top_k):

        q_tokens = soft_tokenize(query)
        q_numbers = set(re.findall(r"\d+", query))

        corpus = [p.payload.get("ocr_text", "") for p in hits]
        bm25 = BM25()
        if corpus:
            bm25.fit(corpus)

        scored = []

        print("\n=============Rerank Scores==========")

        for i, hit in enumerate(hits):
            text = hit.payload.get("ocr_text", "").lower()

            emb = hit.score

            bm25_score = bm25.score(q_tokens, i) if corpus else 0.0
            kw_score = fuzzy_match_score(q_tokens, text)
            phrase_score = sum(
                1 for t in q_tokens if t in text and len(t) > 3
            )
            num_score = 1.0 if q_numbers and any(n in text for n in q_numbers) else 0.0

            final = (
                emb * 1.2 +
                kw_score * 1.5 +
                bm25_score * 0.5 +
                num_score
            )

            scored.append((final, hit))

            print(
                f"Page {hit.payload.get('page_number')} | "
                f"emb={emb:.3f} | bm25={bm25_score:.3f} | "
                f"kw={kw_score:.3f} | phrase={phrase_score:.3f} | "
                f"num={num_score:.3f} | FINAL={final:.5f}"
            )

        scored.sort(key=lambda x: x[0], reverse=True)

        print("\n================ FINAL TOP PAGES ================\n")
        for i, (score, hit) in enumerate(scored[:top_k], 1):
            print(
                f"{i}. Page {hit.payload.get('page_number')} | FINAL={score:.5f}"
            )

        aggressive_cleanup()

        return [h for _, h in scored[:top_k]]