import torch
import numpy as np
import gc
import time
import re
import math
from collections import defaultdict

from PIL import Image
from src.utils import get_pdf_page, clear_page_cache

# -----------------------------
# GLOBAL CACHES
# -----------------------------
_ocr_cache = {}

STOPWORDS = {
    "what","is","are","the","a","an","of","in","on","at","to",
    "for","with","and","or","it","its","this","that","how","why",
    "when","where","who","which","do","does","did","was","were",
    "be","been","being","has","have","had","by","from","about",
}

# -----------------------------
# UTILS
# -----------------------------
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


# -----------------------------
# BM25 (lightweight)
# -----------------------------
class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def fit(self, corpus):
        self.tokenized = []
        for doc in corpus:
            self.tokenized.append(tokenize(doc))

        self.corpus_size = len(self.tokenized)
        self.doc_lens = [len(d) for d in self.tokenized]
        self.avgdl = sum(self.doc_lens) / max(1, self.corpus_size)

        self.doc_freqs = []
        df = defaultdict(int)

        for doc in self.tokenized:
            freq = defaultdict(int)
            for tok in doc:
                freq[tok] += 1
            self.doc_freqs.append(dict(freq))

            for tok in set(doc):
                df[tok] += 1

        self.idf = {}
        for term, n in df.items():
            self.idf[term] = math.log(
                (self.corpus_size - n + 0.5) / (n + 0.5) + 1
            )

    def score(self, query_tokens, idx):
        score = 0.0
        dl = self.doc_lens[idx]
        freq_map = self.doc_freqs[idx]

        for tok in query_tokens:
            if tok not in freq_map:
                continue

            f = freq_map[tok]
            idf = self.idf.get(tok, 0.0)

            score += idf * (
                f * (self.k1 + 1)
                / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            )

        return score


# -----------------------------
# RETRIEVER
# -----------------------------
class MultimodalRetriever:
    def __init__(self, indexer):
        self.indexer = indexer

    # -------------------------
    # EMBEDDING
    # -------------------------
    def _extract_text_embedding(self, query_text):
        start = time.time()

        inputs = self.indexer.processor.process_queries([query_text]).to(
            self.indexer.device
        )

        with torch.no_grad():
            outputs = self.indexer.model(**inputs)

            if hasattr(outputs, "query_embeds"):
                embedding = to_numpy(outputs.query_embeds[0])
            elif hasattr(outputs, "query_embeddings"):
                embedding = to_numpy(outputs.query_embeddings[0])
            else:
                embedding = to_numpy(outputs[0])

            # ✅ in-place normalization
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            embedding /= np.maximum(norms, 1e-8)

        del inputs, outputs
        aggressive_cleanup()

        print(f"Embedding done | tokens={embedding.shape[0]} | {time.time()-start:.2f}s")

        return embedding

    # -------------------------
    # OCR WITH CACHE
    # -------------------------
    def _ocr_page(self, point, generator):
        source = point.payload.get("source", "")
        page_num = point.payload.get("page_number", 0)

        key = (source, page_num)
        if key in _ocr_cache:
            return _ocr_cache[key]

        try:
            if source.lower().endswith(".pdf"):
                img = get_pdf_page(source, page_num)
            else:
                img = Image.open(source)

            text = generator._extract_text(img)

            # free memory
            if hasattr(img, "close"):
                img.close()
            del img

            _ocr_cache[key] = text
            return text

        except Exception as e:
            print(f"OCR failed: {e}")
            return ""

    # -------------------------
    # SEARCH
    # -------------------------
    def search(self, query_text, top_k=5, generator=None):
        query_vec = self._extract_text_embedding(query_text)
        num_tokens = query_vec.shape[0]

        # retrieve candidates
        results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_vec.tolist(),
            using="image",
            limit=top_k * 3,
        ).points

        # normalize scores
        for p in results:
            if p.score:
                p.score /= num_tokens

        # group by page
        page_best = {}
        for p in results:
            key = (p.payload.get("source"), p.payload.get("page_number"))
            if key not in page_best or p.score > page_best[key].score:
                page_best[key] = p

        candidates = list(page_best.values())
        candidates.sort(key=lambda x: x.score, reverse=True)

        # reduce candidates
        candidates = candidates[:6]

        print("\nInitial Candidates:")
        for i, p in enumerate(candidates, 1):
            print(i, p.payload.get("page_number"), f"{p.score:.4f}")

        # rerank
        if generator:
            final = self.rerank(query_text, candidates, generator, top_k)
        else:
            final = candidates[:top_k]

        clear_page_cache()
        aggressive_cleanup()

        return final

    # -------------------------
    # RERANK
    # -------------------------
    def rerank(self, query, hits, generator, top_k=3):
        if not hits:
            return []

        ocr_texts = []
        for p in hits:
            ocr_texts.append(self._ocr_page(p, generator))

        # BM25
        bm25 = BM25()
        bm25.fit(ocr_texts)

        query_tokens = tokenize(query)
        query_numbers = extract_numbers(query.lower())
        content_words = set(query_tokens)

        bm25_scores = []
        keyword_scores = []
        number_scores = []

        for i, text in enumerate(ocr_texts):
            text_lower = text.lower()
            text_set = set(text_lower.split())

            # BM25
            bm25_scores.append(bm25.score(query_tokens, i))

            # keyword
            exact = len(content_words & text_set)
            keyword_scores.append(exact * 2.0)

            # numbers
            if query_numbers:
                page_nums = extract_numbers(text_lower)
                number_scores.append(len(query_numbers & page_nums) * 3.0)
            else:
                number_scores.append(0.0)

        def rrf(scores, k=60):
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            ranks = [0.0] * len(scores)
            for rank, idx in enumerate(order):
                ranks[idx] = 1.0 / (k + rank + 1)
            return ranks

        emb_scores = [p.score for p in hits]

        rrf_final = []
        for i, p in enumerate(hits):
            score = (
                0.4 * rrf(emb_scores)[i] +
                0.3 * rrf(bm25_scores)[i] +
                0.2 * rrf(keyword_scores)[i] +
                0.1 * rrf(number_scores)[i]
            )

            p.score = score
            rrf_final.append((score, p))

        del ocr_texts
        gc.collect()

        rrf_final.sort(key=lambda x: x[0], reverse=True)

        print("\nFinal Reranked:")
        for s, p in rrf_final:
            print(p.payload.get("page_number"), f"{s:.5f}")

        return [p for _, p in rrf_final[:top_k]]