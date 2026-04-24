import torch
import numpy as np
import gc
import re
import time
import math
from collections import defaultdict
from src.utils import pdf_to_images
from PIL import Image
from qdrant_client.models import Filter, FieldCondition, MatchText


STOPWORDS = {
    "what","is","are","the","a","an","of","in","on","at","to",
    "for","with","and","or","it","its","this","that","how","why",
    "when","where","who","which","do","does","did","was","were",
    "be","been","being","has","have","had","by","from","about",
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


def extract_numbers(text: str):
    return set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text))


# ---------------- BM25 ----------------
class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = []
        self.doc_lens = []
        self.avgdl = 0
        self.idf = {}

    def fit(self, corpus):
        tokenized = [tokenize(c) for c in corpus]

        self.doc_lens = [len(d) for d in tokenized]
        self.avgdl = sum(self.doc_lens) / max(1, len(tokenized))

        df = defaultdict(int)
        self.doc_freqs = []

        for doc in tokenized:
            freq = defaultdict(int)
            for t in doc:
                freq[t] += 1
            self.doc_freqs.append(dict(freq))
            for t in set(doc):
                df[t] += 1

        for t, n in df.items():
            self.idf[t] = math.log((len(tokenized) - n + 0.5) / (n + 0.5) + 1)

    def score(self, q, i):
        score = 0
        dl = self.doc_lens[i]
        freq = self.doc_freqs[i]

        for t in q:
            if t not in freq:
                continue
            f = freq[t]
            idf = self.idf.get(t, 0)

            score += idf * (f * (self.k1 + 1)) / (
                f + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl))
            )
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
                emb = outputs[0]

        emb = to_numpy(emb)

        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
        return emb

    # ---------------- OCR ----------------
    def _ocr_page(self, point, generator):
        key = (point.payload.get("source"), point.payload.get("page_number"))

        if key in self._ocr_cache:
            return self._ocr_cache[key]

        try:
            source, page = key

            if source.lower().endswith(".pdf"):
                img = pdf_to_images(source)[page]
            else:
                img = Image.open(source)

            text = generator._extract_text(img)
            self._ocr_cache[key] = text
            return text

        except:
            return ""

    # ---------------- SEARCH ----------------
    def search(self, query_text, top_k=3, source_filter=None, generator=None):

        emb = self._extract_text_embedding(query_text)
        qvec = emb.tolist()

        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[FieldCondition(
                    key="source",
                    match=MatchText(text=source_filter.lower())
                )]
            )

        # 🔥 STEP 1: OVERFETCH (CRITICAL FIX)
        results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=qvec,
            using="image",
            query_filter=query_filter,
            limit=200,
        ).points

        # normalize embedding
        for p in results:
            if p.score:
                p.score /= emb.shape[0]

        # ---------------- DEBUG: INITIAL RANKING ----------------
        print("\n================ INITIAL QDRANT RANKING ================\n")

        for i, p in enumerate(results[:15], 1):
            print(f"{i}. Page {p.payload.get('page_number','?')} | score={p.score:.5f}")

        # ---------------- DEDUP (SAFE VERSION) ----------------
        page_map = {}
        for p in results:
            key = (p.payload.get("source"), p.payload.get("page_number"))

            # keep top 2 per page (not just 1 → prevents recall loss)
            if key not in page_map:
                page_map[key] = [p]
            else:
                if len(page_map[key]) < 2:
                    page_map[key].append(p)

        flat = []
        for v in page_map.values():
            flat.extend(v)

        # keep top 80 for reranking
        candidates = sorted(flat, key=lambda x: x.score, reverse=True)[:80]

        print(f"\nCandidates after dedup: {len(candidates)}\n")

        if generator:
            return self._rerank(query_text, candidates, generator, top_k)

        return candidates[:top_k]

    # ---------------- RERANK ----------------
    def _rerank(self, query, hits, generator, top_k):

        ocr = []
        for p in hits:
            ocr.append(self._ocr_page(p, generator))

        bm25 = BM25()
        bm25.fit(ocr)

        q_tokens = tokenize(query)

        bm25_scores = [bm25.score(q_tokens, i) for i in range(len(hits))]

        ql = query.lower()
        q_words = set(ql.split()) - STOPWORDS
        q_nums = extract_numbers(ql)

        kw, ph, num = [], [], []

        for t in ocr:
            tl = t.lower()

            kw.append(sum(1 for w in q_words if w in tl))

            q = ql.split()
            ph_score = 0
            for n in (2, 3):
                for i in range(len(q)-n+1):
                    if " ".join(q[i:i+n]) in tl:
                        ph_score += 1
            ph.append(ph_score)

            num.append(len(extract_numbers(tl) & q_nums))

        emb = [p.score for p in hits]

        def norm(x):
            m = max(x) if x else 1
            return [i/(m+1e-8) for i in x]

        emb, bm25_scores, kw, ph, num = map(norm, [emb, bm25_scores, kw, ph, num])

        scored = []

        print("\n================ RERANK SCORES ================\n")

        for i, p in enumerate(hits):
            final = (
                0.45 * emb[i] +
                0.25 * bm25_scores[i] +
                0.15 * kw[i] +
                0.10 * ph[i] +
                0.05 * num[i]
            )

            scored.append((final, p))

            print(
                f"Page {p.payload.get('page_number','?')} | "
                f"emb={emb[i]:.3f} | bm25={bm25_scores[i]:.3f} | "
                f"kw={kw[i]:.3f} | final={final:.5f}"
            )

        scored.sort(key=lambda x: x[0], reverse=True)

        print("\n================ FINAL TOP PAGES ================\n")

        for i, (s, p) in enumerate(scored[:top_k], 1):
            print(f"{i}. Page {p.payload.get('page_number','?')} | FINAL={s:.5f}")

        return [p for _, p in scored[:top_k]]