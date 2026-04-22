import torch
import numpy as np
import gc
import re
from qdrant_client.models import GroupBy
import time
import math
from collections import defaultdict
from src.utils import pdf_to_images, get_pdf_page
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
    """Converts anything to NumPy float32 array"""
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def extract_numbers(text: str) -> set[str]:
    """Pull raw numeric strings from text (integers, decimals, percentages)."""
    return set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text))


class BM25:
    """
    Fit on a small corpus of OCR'd page texts, then score
    each document against a query.
    """

    def __init__(self, k1=1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: list[dict] = []
        self.idf: dict[str, float] = {}
        self.doc_lens: list[int] = []

    def fit(self, corpus: list[str]):
        """corpus: list of raw page texts."""
        tokenized = [tokenize(doc) for doc in corpus]
        self.corpus_size = len(tokenized)
        self.doc_lens = [len(d) for d in tokenized]
        self.avgdl = sum(self.doc_lens) / max(1, self.corpus_size)
        self.doc_freqs = []
        df: dict[str, int] = defaultdict(int)
        for doc in tokenized:
            freq: dict[str, int] = defaultdict(int)
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


class MultimodalRetriever:
    def __init__(self, indexer):
        self.indexer = indexer

    def _extract_text_embedding(self, query_text: str):
        """Extract multi-vector text embedding for ColQwen2.5"""
        start = time.time()

        inputs = self.indexer.processor.process_queries([query_text]).to(self.indexer.device)

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

            # L2 normalize each token vector
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embedding = embedding / norms

        embed_time = time.time() - start
        print(f"Query embedded in {embed_time:.3f}s | Tokens: {embedding.shape[0]}")

        del inputs, outputs
        aggressive_cleanup()
        return embedding, embed_time

    def _ocr_page(self, source: str, page_num: int, generator) -> str:
        """
        OCR a full page (not a chunk) using source + page_num directly.
        Always operates on the full page image regardless of chunk coordinates.
        """
        try:
            if source.lower().endswith(".pdf"):
                img = get_pdf_page(source, page_num)
            else:
                img = Image.open(source)
            text = generator._extract_text(img)
            print(f"  OCR: Page {page_num} → {len(text)} chars extracted")
            return text
        except Exception as e:
            print(f"  OCR failed for page {page_num}: {e}")
            return ""

    def _dedup_by_page(self, results) -> list:
        """
        Since chunking creates multiple Qdrant points per page, multiple chunks
        from the same page can appear in results. This keeps only the highest-scoring
        chunk per (source, page_number) pair, returning one representative point per page.

        The representative point's score = max chunk score for that page.
        """
        best: dict[tuple, object] = {}  # key: (source, page_num) → best point

        for point in results:
            source   = point.payload.get("source", "")
            page_num = point.payload.get("page_number", 0)
            key = (source, page_num)

            if key not in best or point.score > best[key].score:
                best[key] = point

        deduped = list(best.values())
        print(f"Dedup: {len(results)} chunks → {len(deduped)} unique pages")
        return deduped

    def search(self, query_text: str, top_k: int = 10, source_filter: str = None, generator=None):
        start_search = time.time()

        # 1. Get ColQwen2.5 embedding
        query_emb_array, embed_time = self._extract_text_embedding(query_text)
        query_multi_vec = query_emb_array.tolist()

        # 2. Optional filter
        query_filter = (
            Filter(must=[FieldCondition(key="source", match=MatchText(text=source_filter.lower()))])
            if source_filter else None
        )

        # 3. Retrieve candidates from Qdrant
        # Fetch more than top_k because multiple chunks from the same page will be returned
        # and dedup will collapse them. A safe multiplier is 4–6x.
        fetch_limit = top_k * 6
        retr_start = time.time()
        results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_multi_vec,
            using="image",
            query_filter=query_filter,
            limit=fetch_limit,
            score_threshold=None,
        ).points
        retr_time = time.time() - retr_start

        # 4. Normalize embedding score by number of query tokens
        num_query_tokens = query_emb_array.shape[0]
        for point in results:
            if point.score is not None:
                point.score = point.score / num_query_tokens

        print(f"Qdrant retrieval: {len(results)} chunks in {retr_time:.2f}s")

        # 5. Deduplicate: collapse multiple chunks → one point per page (best score wins)
        deduped = self._dedup_by_page(results)

        print(f"Qdrant retrieval done in {time.time() - start_search:.2f}s | Unique pages: {len(deduped)}")

        # 6. Hybrid reranking on deduplicated page-level hits
        if generator:
            final_hits = self._hybrid_rerank(query_text, deduped, generator, top_k=top_k)
        else:
            final_hits = sorted(deduped, key=lambda x: x.score, reverse=True)[:top_k]

        total_time = time.time() - start_search
        print(f"Total search + rerank time: {total_time:.2f}s\n")

        return final_hits

    def _hybrid_rerank(self, query: str, hits, generator, top_k: int = 3):
        """
        Single-pass hybrid reranker (Embedding + BM25 + Keyword + Numeric).
        hits are already deduplicated to one point per page at this stage,
        so OCR always operates on full pages.
        """
        if not hits:
            return []

        # OCR full pages (not chunks) — source + page_num come from payload
        ocr_texts = []
        for point in hits[:10]:
            source   = point.payload.get("source", "")
            page_num = point.payload.get("page_number", 0)
            text = self._ocr_page(source, page_num, generator)
            ocr_texts.append(text)

        # BM25
        bm25 = BM25()
        bm25.fit(ocr_texts)
        query_tokens = tokenize(query)
        bm25_scores = [bm25.score(query_tokens, i) for i in range(len(hits))]

        # Keyword, Phrase & Number signals
        query_lower      = query.lower()
        content_words    = set(query_lower.split()) - STOPWORDS
        query_numbers    = extract_numbers(query_lower)
        query_tokens_raw = query_lower.split()

        keyword_scores = []
        phrase_scores  = []
        number_scores  = []

        for text in ocr_texts:
            text_lower = text.lower()
            page_words = text_lower.split()

            # Keyword match
            exact   = sum(1 for w in content_words if w in text_lower)
            partial = sum(1 for w in content_words if any(w in tok for tok in page_words))
            keyword_scores.append(exact * 2.0 + partial * 0.5)

            # Phrase match
            phrase_bonus = 0.0
            for n in (2, 3):
                for i in range(len(query_tokens_raw) - n + 1):
                    phrase = " ".join(query_tokens_raw[i:i + n])
                    if phrase in text_lower:
                        phrase_bonus += n * 2.0
            phrase_scores.append(phrase_bonus)

            # Numeric match
            page_numbers = extract_numbers(text_lower)
            number_scores.append(len(query_numbers & page_numbers) * 3.0)

        # Weighted combination
        emb_scores  = [p.score for p in hits]
        final_scores = []

        for i in range(len(hits)):
            score = (
                0.35 * (emb_scores[i]      / max(emb_scores)) +
                0.30 * (bm25_scores[i]     / (max(bm25_scores)     + 1e-8)) +
                0.20 * (keyword_scores[i]  / (max(keyword_scores)  + 1e-8)) +
                0.10 * (phrase_scores[i]   / (max(phrase_scores)   + 1e-8)) +
                0.05 * (number_scores[i]   / (max(number_scores)   + 1e-8))
            )
            final_scores.append(score)

            pg = hits[i].payload.get("page_number", "?")
            x  = hits[i].payload.get("x", "-")
            y  = hits[i].payload.get("y", "-")
            print(f"Page {pg} chunk({x},{y}) | emb={emb_scores[i]:.4f} | bm25={bm25_scores[i]:.3f} | "
                  f"kw={keyword_scores[i]:.2f} | final={score:.5f}")

        # Sort and return top_k
        scored_hits = sorted(zip(final_scores, hits), key=lambda x: x[0], reverse=True)
        final_hits  = [point for _, point in scored_hits[:top_k]]

        print(f"Hybrid reranking completed → {len(final_hits)} final pages")
        return final_hits