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
    def search(self, query_text, top_k=10, source_filter=None):

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

        # normalize scores
        for p in results:
            if p.score is not None:
                p.score /= emb.shape[0]

        # -------- Deduplicate by page --------
        page_best = {}
        for p in results:
            key = (p.payload.get("source"), p.payload.get("page_number"))
            if key not in page_best or p.score > page_best[key].score:
                page_best[key] = p

        results = sorted(page_best.values(), key=lambda x: x.score, reverse=True)

        if VERBOSE:
            print(f"Qdrant retrieval done | Candidates: {len(results)}\n")

        hits = results[:15]

        return self._rerank(clean_query, hits, top_k)

    # -------- Reranking (NO OCR here) --------
    def _rerank(self, query, hits, top_k, rrf_k: int = 60):

        if VERBOSE:
            print("\n=============Rerank Scores==========")

        # ✅ USE STORED OCR TEXT
        ocr_texts = [p.payload.get("ocr_text", "") for p in hits]

        # -------- BM25 --------
        bm25 = BM25()
        bm25.fit(ocr_texts)

        q_tokens = tokenize(query)
        bm25_scores = [bm25.score(q_tokens, i) for i in range(len(hits))]

        # -------- Heuristics --------
        content_words = set(q_tokens)
        query_numbers = extract_numbers(query)

        keyword_scores, phrase_scores, number_scores = [], [], []

        for text in ocr_texts:
            t = text.lower()

            exact = sum(1 for w in content_words if w in t)
            partial = sum(1 for w in content_words if any(w in x for x in t.split()))
            keyword_scores.append(exact * 3 + partial * 0.5)

            phrase_bonus = 0
            for n in (2, 3):
                for i in range(len(q_tokens) - n + 1):
                    if " ".join(q_tokens[i:i+n]) in t:
                        phrase_bonus += n * 2
            phrase_scores.append(phrase_bonus)

            nums = extract_numbers(t)
            number_scores.append(len(nums & query_numbers) * 3)

        emb_scores = [p.score for p in hits]

        def norm(x):
            m = max(x) if x else 1
            return [i / (m + 1e-8) for i in x]

        emb_norm    = norm(emb_scores)
        bm25_norm   = norm(bm25_scores)
        kw_norm     = norm(keyword_scores)
        phrase_norm = norm(phrase_scores)
        num_norm    = norm(number_scores)

        indices = list(range(len(hits)))

        rankings = [
            sorted(indices, key=lambda i: emb_norm[i], reverse=True),
            sorted(indices, key=lambda i: bm25_norm[i], reverse=True),
            sorted(indices, key=lambda i: kw_norm[i], reverse=True),
            sorted(indices, key=lambda i: phrase_norm[i], reverse=True),
            sorted(indices, key=lambda i: num_norm[i], reverse=True),
        ]

        # -------- RRF Fusion --------
        rrf_scores = defaultdict(float)
        for ranking in rankings:
            for rank_pos, idx in enumerate(ranking, start=1):
                rrf_scores[idx] += 1.0 / (rrf_k + rank_pos)

        final_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        if VERBOSE:
            print("\nFinal Ranking:\n")
            for i, (idx, score) in enumerate(final_ranking[:top_k], 1):
                page = hits[idx].payload.get("page_number")
                print(f"{i}. Page {page} | RRF={score:.5f}")

        aggressive_cleanup()

        return [hits[idx] for idx, _ in final_ranking[:top_k]]