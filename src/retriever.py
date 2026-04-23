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
    "what", "is", "are", "the", "a", "an", "of", "in", "on", "at", "to",
    "for", "with", "and", "or", "it", "its", "this", "that", "how", "why",
    "when", "where", "who", "which", "do", "does", "did", "was", "were",
    "be", "been", "being", "has", "have", "had", "by", "from", "about",
}


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text))


# =========================
# BM25
# =========================
class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs = []
        self.idf = {}
        self.doc_lens = []

    def fit(self, corpus: list[str]):
        tokenized = [tokenize(doc) for doc in corpus]

        self.corpus_size = len(tokenized)
        self.doc_lens = [len(d) for d in tokenized]
        self.avgdl = sum(self.doc_lens) / max(1, self.corpus_size)
        self.doc_freqs = []

        df = defaultdict(int)

        for doc in tokenized:
            freq = defaultdict(int)
            for tok in doc:
                freq[tok] += 1

            self.doc_freqs.append(dict(freq))

            for tok in set(doc):
                df[tok] += 1

        for term, n in df.items():
            self.idf[term] = math.log(
                (self.corpus_size - n + 0.5) / (n + 0.5) + 1
            )

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        score = 0.0
        dl = self.doc_lens[doc_idx]
        freq_map = self.doc_freqs[doc_idx]

        for tok in query_tokens:
            if tok not in freq_map:
                continue

            f = freq_map[tok]
            idf = self.idf.get(tok, 0.0)

            num = f * (self.k1 + 1)
            den = f + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl))

            score += idf * num / den

        return score


# =========================
# Retriever
# =========================
class MultimodalRetriever:
    def __init__(self, indexer):
        self.indexer = indexer

    # -------------------------
    # Embedding
    # -------------------------
    def _extract_text_embedding(self, query_text: str):
        start = time.time()

        inputs = self.indexer.processor.process_queries(
            [query_text]
        ).to(self.indexer.device)

        with torch.no_grad():
            outputs = self.indexer.model(**inputs)

            if hasattr(outputs, "query_embeds") and outputs.query_embeds is not None:
                embedding = to_numpy(outputs.query_embeds[0])
            elif hasattr(outputs, "query_embeddings") and outputs.query_embeddings is not None:
                embedding = to_numpy(outputs.query_embeddings[0])
            elif isinstance(outputs, torch.Tensor):
                embedding = to_numpy(outputs[0])
            else:
                embedding = to_numpy(outputs)

        # normalize
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embedding = embedding / norms

        print(f"Query embedded in {time.time() - start:.3f}s")

        del inputs, outputs
        aggressive_cleanup()

        return embedding

    # -------------------------
    # OCR
    # -------------------------
    def _ocr_page(self, point, generator) -> str:
        source = point.payload.get("source", "")
        page_num = point.payload.get("page_number", 0)

        try:
            if source.lower().endswith(".pdf"):
                pages = pdf_to_images(source)
                img = pages[page_num]
            else:
                img = Image.open(source)

            return generator._extract_text(img)

        except Exception:
            return ""

    # -------------------------
    # Search
    # -------------------------
    def search(self, query_text: str, top_k: int = 15, source_filter=None, generator=None):

        query_emb = self._extract_text_embedding(query_text)
        query_vec = query_emb.tolist()
        num_tokens = query_emb.shape[0]

        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchText(text=source_filter.lower())
                    )
                ]
            )

        results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_vec,
            using="image",
            query_filter=query_filter,
            limit=top_k,
        ).points

        for p in results:
            p.score = p.score / num_tokens

        # group by page
        page_best = defaultdict(lambda: {"score": -1, "point": None})

        for p in results:
            key = (p.payload.get("source"), p.payload.get("page_number"))

            if p.score > page_best[key]["score"]:
                page_best[key] = {"score": p.score, "point": p}

        sorted_pages = sorted(page_best.values(), key=lambda x: x["score"], reverse=True)
        initial_hits = [x["point"] for x in sorted_pages[:10]]

        if generator:
            final_hits = self.rerank_hits(query_text, initial_hits, generator, top_k=5)
        else:
            final_hits = initial_hits[:5]

        return final_hits

    # -------------------------
    # Reranking (FIXED)
    # -------------------------
    def rerank_hits(self, query: str, hits, generator, top_k: int = 5):

        if not hits:
            return []

        ocr_texts = []
        for p in hits:
            ocr_texts.append(self._ocr_page(p, generator))

        bm25 = BM25()
        bm25.fit(ocr_texts)
        q_tokens = tokenize(query)

        bm25_scores = [bm25.score(q_tokens, i) for i in range(len(hits))]

        query_lower = query.lower()
        query_numbers = extract_numbers(query_lower)

        keyword_scores = []
        phrase_scores = []
        number_scores = []

        for text in ocr_texts:
            text_lower = text.lower()

            words = set(query_lower.split()) - STOPWORDS

            exact = sum(1 for w in words if w in text_lower)
            keyword_scores.append(exact)

            phrases = 0
            toks = query_lower.split()
            for i in range(len(toks) - 1):
                phrase = " ".join(toks[i:i+2])
                if phrase in text_lower:
                    phrases += 1
            phrase_scores.append(phrases)

            page_nums = extract_numbers(text_lower)
            number_scores.append(len(query_numbers & page_nums))

        emb_scores = [p.score for p in hits]

        def norm(x):
            m = max(x) + 1e-8
            return [v / m for v in x]

        emb_scores = norm(emb_scores)
        bm25_scores = norm(bm25_scores)
        keyword_scores = norm(keyword_scores)
        phrase_scores = norm(phrase_scores)
        number_scores = norm(number_scores)

        final = []

        for i, p in enumerate(hits):
            score = (
                0.35 * emb_scores[i] +
                0.30 * bm25_scores[i] +
                0.20 * keyword_scores[i] +
                0.10 * phrase_scores[i] +
                0.05 * number_scores[i]
            )

            p.score = score
            final.append((score, p))

        final.sort(reverse=True, key=lambda x: x[0])

        return [p for _, p in final[:top_k]]