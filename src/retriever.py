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


# ---------------- BM25 ----------------
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

        df = defaultdict(int)
        self.doc_freqs = []

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

    def score(self, query_tokens, doc_idx):
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


# ---------------- RETRIEVER ----------------
class MultimodalRetriever:
    def __init__(self, indexer, use_reranker: bool = True):
        self.indexer = indexer
        self.use_reranker = use_reranker
        self._ocr_cache = {}   # IMPORTANT: avoids recomputing OCR

    # ---------------- EMBEDDING ----------------
    def _extract_text_embedding(self, query_text: str):
        start = time.time()

        inputs = self.indexer.processor.process_queries([query_text]).to(self.indexer.device)

        with torch.no_grad():
            outputs = self.indexer.model(**inputs)

            if hasattr(outputs, "query_embeds") and outputs.query_embeds is not None:
                embedding = to_numpy(outputs.query_embeds[0])
            elif hasattr(outputs, "query_embeddings") and outputs.query_embeddings is not None:
                embedding = to_numpy(outputs.query_embeddings[0])
            else:
                embedding = to_numpy(outputs[0] if isinstance(outputs, torch.Tensor) else outputs)

        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embedding = embedding / norms

        print(f"Query embedded in {time.time() - start:.3f}s")

        aggressive_cleanup()
        return embedding

    # ---------------- OCR ----------------
    def _ocr_page(self, point, generator):
        key = (point.payload.get("source"), point.payload.get("page_number"))

        if key in self._ocr_cache:
            return self._ocr_cache[key]

        source, page_num = key

        try:
            if source.lower().endswith(".pdf"):
                pages = pdf_to_images(source)
                img = pages[page_num]
            else:
                img = Image.open(source)

            text = generator._extract_text(img)
            self._ocr_cache[key] = text
            return text

        except Exception:
            return ""

    # ---------------- SEARCH ----------------
    def search(self, query_text: str, top_k: int = 10, source_filter: str = None, generator=None):

        query_emb = self._extract_text_embedding(query_text)
        query_vec = query_emb.tolist()

        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[FieldCondition(
                    key="source",
                    match=MatchText(text=source_filter.lower())
                )]
            )

        # OVERFETCH
        results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_vec,
            using="image",
            query_filter=query_filter,
            limit=50,
            score_threshold=None,
        ).points

        # normalize embedding score
        num_tokens = query_emb.shape[0]
        for p in results:
            if p.score is not None:
                p.score /= num_tokens

        # ---------------- DEDUP (CRITICAL FIX) ----------------
        page_best = {}
        for p in results:
            key = (p.payload.get("source"), p.payload.get("page_number"))

            if key not in page_best or p.score > page_best[key].score:
                page_best[key] = p

        # KEEP ONLY TOP 10 UNIQUE PAGES
        results = sorted(page_best.values(), key=lambda x: x.score, reverse=True)[:10]

        print(f"Unique candidates (top 10 pages): {len(results)}")

        # ---------------- RERANK ----------------
        if generator and self.use_reranker:
            return self._hybrid_rerank(query_text, results, generator, top_k)
        else:
            return results[:top_k]

    # ---------------- HYBRID RERANK ----------------
    def _hybrid_rerank(self, query, hits, generator, top_k=5):

        ocr_texts = [self._ocr_page(p, generator) for p in hits]

        bm25 = BM25()
        bm25.fit(ocr_texts)
        q_tokens = tokenize(query)

        bm25_scores = [bm25.score(q_tokens, i) for i in range(len(hits))]

        query_lower = query.lower()
        content_words = set(query_lower.split()) - STOPWORDS
        query_numbers = extract_numbers(query_lower)

        keyword_scores = []
        phrase_scores = []
        number_scores = []

        for text in ocr_texts:
            t = text.lower()

            exact = sum(1 for w in content_words if w in t)
            partial = sum(1 for w in content_words if any(w in tok for tok in t.split()))
            keyword_scores.append(exact * 2 + partial * 0.5)

            phrase_bonus = 0.0
            q_tokens = query_lower.split()
            for n in (2, 3):
                for i in range(len(q_tokens) - n + 1):
                    phrase = " ".join(q_tokens[i:i+n])
                    if phrase in t:
                        phrase_bonus += n * 2
            phrase_scores.append(phrase_bonus)

            page_numbers = extract_numbers(t)
            number_scores.append(len(query_numbers & page_numbers) * 3)

        emb_scores = [p.score for p in hits]

        def norm(x):
            m = max(x) if x else 1.0
            return [i / (m + 1e-8) for i in x]

        emb_scores = norm(emb_scores)
        bm25_scores = norm(bm25_scores)
        keyword_scores = norm(keyword_scores)
        phrase_scores = norm(phrase_scores)
        number_scores = norm(number_scores)

        final = []

        for i, p in enumerate(hits):
            score = (
                0.40 * emb_scores[i] +
                0.25 * bm25_scores[i] +
                0.20 * keyword_scores[i] +
                0.10 * phrase_scores[i] +
                0.05 * number_scores[i]
            )

            final.append((score, p))

            pg = p.payload.get("page_number", "?")
            print(f"Page {pg} | final={score:.4f}")

        final.sort(key=lambda x: x[0], reverse=True)

        return [p for _, p in final[:top_k]]