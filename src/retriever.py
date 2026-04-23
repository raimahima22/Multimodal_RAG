import torch
import numpy as np
import gc
import re
import time
import math
from collections import defaultdict
from functools import lru_cache
from src.utils import pdf_to_images
from src.utils import get_pdf_page, clear_page_cache
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
    """Converts anything to NumPy float32 array."""
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


def safe_normalize(scores: list[float], eps: float = 1e-8) -> list[float]:
    """
    Min-max normalize a list of scores to [0, 1].
    Returns zeros if all scores are equal (avoids division by zero).
    Uses min-max instead of max-only so negative scores are handled correctly.
    """
    mn, mx = min(scores), max(scores)
    span = mx - mn
    if span < eps:
        return [0.0] * len(scores)
    return [(s - mn) / span for s in scores]


class BM25:
    """
    Okapi BM25 scorer.

    Two usage modes:
    - Pre-fit on full corpus: call fit() once at index time for accurate IDF.
    - Fit on a small candidate set: IDF is approximate but still useful for
      relative ranking within the candidate pool.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1        # term-saturation factor
        self.b = b          # length-normalization factor
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        self.doc_lens: list[int] = []

    def fit(self, corpus: list[str]) -> None:
        """
        Fit BM25 on a list of raw page texts.
        Safe to call on corpora of any size, including very small ones.
        """
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

        # Okapi IDF — same formula regardless of corpus size
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
        # OCR cache: (source, page_number) -> str
        # Avoids redundant OCR calls across queries for the same pages.
        self._ocr_cache: dict[tuple[str, int], str] = {}

        # Optional: pre-fitted BM25 on the full indexed corpus.
        # Call fit_bm25_on_corpus() after indexing to enable accurate IDF.
        self._corpus_bm25: BM25 | None = None
        self._corpus_texts: list[str] = []
        self._corpus_ids: list[tuple[str, int]] = []   # (source, page_number)

    # ------------------------------------------------------------------
    # Public: full-corpus BM25 pre-fitting
    # ------------------------------------------------------------------

    def fit_bm25_on_corpus(
        self,
        page_texts: list[str],
        page_ids: list[tuple[str, int]],
    ) -> None:
        """
        Pre-fit BM25 on the entire indexed corpus for accurate IDF scores.
        Call this once after indexing is complete.

        Args:
            page_texts: OCR/text content of every indexed page.
            page_ids:   Matching (source, page_number) tuples for each page.
        """
        assert len(page_texts) == len(page_ids), (
            "page_texts and page_ids must have the same length"
        )
        bm25 = BM25()
        bm25.fit(page_texts)
        self._corpus_bm25 = bm25
        self._corpus_texts = page_texts
        self._corpus_ids = page_ids
        print(f"[BM25] Pre-fitted on {len(page_texts)} pages.")

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _extract_text_embedding(self, query_text: str) -> tuple[np.ndarray, float]:
        """Extract and L2-normalise multi-vector text embedding (ColQwen2.5)."""
        start = time.time()

        inputs = self.indexer.processor.process_queries([query_text]).to(
            self.indexer.device
        )

        with torch.no_grad():
            outputs = self.indexer.model(**inputs)

            if hasattr(outputs, "query_embeds") and outputs.query_embeds is not None:
                embedding = to_numpy(outputs.query_embeds[0])
            elif (
                hasattr(outputs, "query_embeddings")
                and outputs.query_embeddings is not None
            ):
                embedding = to_numpy(outputs.query_embeddings[0])
            elif isinstance(outputs, torch.Tensor):
                embedding = to_numpy(outputs[0])
            else:
                embedding = to_numpy(outputs)

        # L2-normalise each token vector — done outside no_grad (pure NumPy)
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embedding = embedding / norms

        embed_time = time.time() - start
        print(f"Query embedded in {embed_time:.3f}s | Tokens: {embedding.shape[0]}")

        del inputs, outputs
        aggressive_cleanup()
        return embedding, embed_time

    # ------------------------------------------------------------------
    # OCR (with caching)
    # ------------------------------------------------------------------

    def _ocr_page(self, point, generator) -> str:
        source: str = point.payload.get("source", "")
        page_num: int = point.payload.get("page_number", 0)
        cache_key = (source, page_num)

    #  OCR cache hit
        if cache_key in self._ocr_cache:
            return self._ocr_cache[cache_key]

        try:
            if source.lower().endswith(".pdf"):
            #  ONLY LOAD ONE PAGE (CRITICAL FIX)
                img = get_pdf_page(source, page_num)
            else:
                img = Image.open(source)

        # Optional: downscale to reduce RAM
            img = img.resize((1024, 1024))

            text = generator._extract_text(img)
            print(f"  OCR: Page {page_num} → {len(text)} chars")

        except Exception as e:
            print(f"  OCR failed for page {page_num}: {e}")
            text = ""

    #  Limit OCR cache size (prevents RAM leak)
        if len(self._ocr_cache) > 100:
            self._ocr_cache.clear()

        self._ocr_cache[cache_key] = text

    #  Free memory immediately
        del img
        aggressive_cleanup()

        return text

    # ------------------------------------------------------------------
    # Public: search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        source_filter: str | None = None,
        generator=None,
    ) -> list:
        start_search = time.time()

        # 1. ColQwen2.5 query embedding
        query_emb_array, embed_time = self._extract_text_embedding(query_text)
        query_multi_vec = query_emb_array.tolist()

        # 2. Optional Qdrant source filter
        query_filter = (
            Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchText(text=source_filter.lower()),
                    )
                ]
            )
            if source_filter
            else None
        )

        # 3. Retrieve candidates from Qdrant
        results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_multi_vec,
            using="image",
            query_filter=query_filter,
            limit=10,
            score_threshold=None,
        ).points

        # Normalise raw ColQwen2.5 MaxSim score by number of query tokens
        num_query_tokens = query_emb_array.shape[0]
        for point in results:
            if point.score is not None:
                point.score = point.score / num_query_tokens

        print(
            f"Qdrant retrieval done in {time.time() - start_search:.2f}s"
            f" | Candidates: {len(results)}"
        )

        # 4. Hybrid reranking
        if generator:
            final_hits = self._hybrid_rerank(
                query_text, results, generator, top_k=top_k
            )
        else:
            final_hits = sorted(results, key=lambda x: x.score, reverse=True)[:top_k]

        print(f"Total search + rerank time: {time.time() - start_search:.2f}s\n")
        return final_hits

    # ------------------------------------------------------------------
    # Private: hybrid reranker
    # ------------------------------------------------------------------

    def _hybrid_rerank(self, query: str, hits: list, generator, top_k: int = 5) -> list:
        """
        Hybrid reranker combining:
          - ColQwen2.5 embedding score  (35 %)
          - BM25                        (30 %)
          - Exact keyword match         (20 %)
          - Phrase match                (10 % — zeroed when no phrase matches)
          - Numeric overlap             ( 5 % — zeroed when query has no numbers)

        Unused signal weights are redistributed proportionally so the
        total always sums to 1.0.
        """
        if not hits:
            return []

        # --- OCR (cached) ---
        ocr_texts = [self._ocr_page(p, generator) for p in hits]

        # --- BM25 ---
        # Prefer pre-fitted corpus BM25 (accurate IDF).
        # Fall back to fitting on the candidate set only if unavailable.
        if self._corpus_bm25 is not None:
            bm25 = self._corpus_bm25
            # Map each hit to its corpus index so we can call bm25.score()
            id_to_corpus_idx = {pid: i for i, pid in enumerate(self._corpus_ids)}
            bm25_scores = []
            query_tokens = tokenize(query)
            for point in hits:
                key = (
                    point.payload.get("source", ""),
                    point.payload.get("page_number", 0),
                )
                idx = id_to_corpus_idx.get(key)
                if idx is not None:
                    bm25_scores.append(bm25.score(query_tokens, idx))
                else:
                    bm25_scores.append(0.0)
        else:
            # Fallback: fit on the 10-candidate pool.
            # IDF is noisy at this scale; treat it as approximate TF-IDF.
            bm25 = BM25()
            bm25.fit(ocr_texts)
            query_tokens = tokenize(query)
            bm25_scores = [bm25.score(query_tokens, i) for i in range(len(hits))]

        # --- Keyword, phrase, and numeric signals ---
        query_lower = query.lower()
        content_words = set(query_lower.split()) - STOPWORDS
        query_numbers = extract_numbers(query_lower)
        query_tokens_raw = query_lower.split()

        keyword_scores: list[float] = []
        phrase_scores: list[float] = []
        number_scores: list[float] = []

        for text in ocr_texts:
            text_lower = text.lower()

            # --- Keyword: exact substring match on the full string (O(n)) ---
            exact = sum(1.0 for w in content_words if w in text_lower)
            keyword_scores.append(exact)

            # --- Phrase: n-gram (n=2,3) substring matches ---
            phrase_bonus = 0.0
            for n in (2, 3):
                for i in range(len(query_tokens_raw) - n + 1):
                    phrase = " ".join(query_tokens_raw[i : i + n])
                    if phrase in text_lower:
                        phrase_bonus += float(n) * 2.0
            phrase_scores.append(phrase_bonus)

            # --- Numeric overlap ---
            page_numbers = extract_numbers(text_lower)
            number_scores.append(float(len(query_numbers & page_numbers)) * 3.0)

        # --- Decide active signals and base weights ---
        base_weights = {
            "emb":     0.35,
            "bm25":    0.30,
            "keyword": 0.20,
            "phrase":  0.10 if any(s > 0 for s in phrase_scores) else 0.0,
            "number":  0.05 if query_numbers else 0.0,
        }
        total_w = sum(base_weights.values())
        # Redistribute unused weight proportionally
        weights = {k: v / total_w for k, v in base_weights.items()}

        # --- Normalise each signal to [0, 1] with safe min-max ---
        emb_scores = [p.score if p.score is not None else 0.0 for p in hits]
        norm_emb     = safe_normalize(emb_scores)
        norm_bm25    = safe_normalize(bm25_scores)
        norm_keyword = safe_normalize(keyword_scores)
        norm_phrase  = safe_normalize(phrase_scores)
        norm_number  = safe_normalize(number_scores)

        # --- Weighted combination ---
        final_scores: list[float] = []
        for i in range(len(hits)):
            score = (
                weights["emb"]     * norm_emb[i]
                + weights["bm25"]    * norm_bm25[i]
                + weights["keyword"] * norm_keyword[i]
                + weights["phrase"]  * norm_phrase[i]
                + weights["number"]  * norm_number[i]
            )
            final_scores.append(score)

            pg = hits[i].payload.get("page_number", "?")
            print(
                f"Page {pg:>3} | emb={emb_scores[i]:.4f}"
                f" | bm25={bm25_scores[i]:.3f}"
                f" | kw={keyword_scores[i]:.2f}"
                f" | phrase={phrase_scores[i]:.2f}"
                f" | num={number_scores[i]:.2f}"
                f" | final={score:.5f}"
            )

        # --- Sort and return top_k ---
        scored_hits = sorted(
            zip(final_scores, hits), key=lambda x: x[0], reverse=True
        )
        final_hits = [point for _, point in scored_hits[:top_k]]
        print(f"Hybrid reranking completed → {len(final_hits)} final pages")
        return final_hits