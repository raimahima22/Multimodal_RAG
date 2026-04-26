import torch
import numpy as np
import gc
import re
import math
from collections import defaultdict
from functools import lru_cache

from src.utils import pdf_to_images
from PIL import Image
from qdrant_client.models import Filter, FieldCondition, MatchText


STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "in", "on", "at", "to",
    "for", "with", "and", "or", "it", "its", "this", "that", "how", "why",
    "when", "where", "who", "which", "do", "does", "did", "was", "were",
    "be", "been", "being", "has", "have", "had", "by", "from", "about",
}


# ---------------- MEMORY ----------------
def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    return np.asarray(x, dtype=np.float32)


# ---------------- TEXT UTILS ----------------
def tokenize(text: str):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def extract_numbers(text: str):
    return set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text))


# ---------------- BM25 ----------------
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


# ---------------- RETRIEVER ----------------
class MultimodalRetriever:
    def __init__(self, indexer):
        self.indexer = indexer
        self._ocr_cache = {}

    # ---------------- EMBEDDING ----------------
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

        # safe normalization (no extra copies)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

        return emb

    # ---------------- OCR (SAFE) ----------------
    def _ocr_page(self, point, generator):
        key = (point.payload.get("source"), point.payload.get("page_number"))

        if key in self._ocr_cache:
            return self._ocr_cache[key]

        source, page_num = key

        try:
            if source.lower().endswith(".pdf"):
                # 🔥 ONLY LOAD ONE PAGE
                pages = pdf_to_images(
                    source,
                    first_page=page_num + 1,
                    last_page=page_num + 1
                )
                img = pages[0]
                del pages
            else:
                img = Image.open(source)

            text = generator._extract_text(img)

            # limit cache size
            if len(self._ocr_cache) > 100:
                self._ocr_cache.clear()
                aggressive_cleanup()

            self._ocr_cache[key] = text

            del img
            return text

        except Exception:
            return ""

    # ---------------- SEARCH ----------------
    def search(self, query_text, top_k=10, source_filter=None, generator=None):

        emb = self._extract_text_embedding(query_text)
        query_vec = emb.tolist()

        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[FieldCondition(
                    key="source",
                    match=MatchText(text=source_filter.lower())
                )]
            )

        # 🔥 REDUCED LIMIT
        results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_vec,
            using="image",
            query_filter=query_filter,
            limit=50,
        ).points

        # normalize score
        for p in results:
            if p.score is not None:
                p.score /= emb.shape[0]

        # deduplicate per page
        page_best = {}
        for p in results:
            key = (p.payload.get("source"), p.payload.get("page_number"))
            if key not in page_best or p.score > page_best[key].score:
                page_best[key] = p

        results = sorted(page_best.values(), key=lambda x: x.score, reverse=True)

        print(f"Qdrant retrieval done | Candidates: {len(results)}")

        # 🔥 LIMIT BEFORE OCR
        hits = results[:8]

        if generator:
            return self._rerank(query_text, hits, generator, top_k)
        else:
            return hits[:top_k]

    # ---------------- RERANK ----------------
    def _rerank(self, query, hits, generator, top_k):

        ocr_texts = [self._ocr_page(p, generator) for p in hits]

        bm25 = BM25()
        bm25.fit(ocr_texts)

        q_tokens = tokenize(query)
        bm25_scores = [bm25.score(q_tokens, i) for i in range(len(hits))]

        query_lower = query.lower()
        content_words = set(query_lower.split()) - STOPWORDS
        query_numbers = extract_numbers(query_lower)

        keyword_scores, phrase_scores, number_scores = [], [], []

        for text in ocr_texts:
            t = text.lower()

            exact = sum(1 for w in content_words if w in t)
            partial = sum(1 for w in content_words if any(w in x for x in t.split()))
            keyword_scores.append(exact * 2 + partial * 0.5)

            phrase_bonus = 0
            q = query_lower.split()
            for n in (2, 3):
                for i in range(len(q) - n + 1):
                    if " ".join(q[i:i+n]) in t:
                        phrase_bonus += n * 2
            phrase_scores.append(phrase_bonus)

            nums = extract_numbers(t)
            number_scores.append(len(nums & query_numbers) * 3)

        emb_scores = [p.score for p in hits]

        def norm(x):
            m = max(x) if x else 1
            return [i / (m + 1e-8) for i in x]

        emb_scores = norm(emb_scores)
        bm25_scores = norm(bm25_scores)
        keyword_scores = norm(keyword_scores)
        phrase_scores = norm(phrase_scores)
        number_scores = norm(number_scores)

        final = []

        for i, p in enumerate(hits):
            score = (
                0.50 * emb_scores[i] +
                0.20 * bm25_scores[i] +
                0.15 * keyword_scores[i] +
                0.10 * phrase_scores[i] +
                0.05 * number_scores[i]
            )
            final.append((score, p))

        final.sort(key=lambda x: x[0], reverse=True)

        # cleanup big objects
        del ocr_texts
        aggressive_cleanup()

        return [p for _, p in final[:top_k]]