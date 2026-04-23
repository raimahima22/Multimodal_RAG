import torch
import numpy as np
import gc
import time
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


    def __init__(self, k1 = 1.5, b:float = 0.75):
        self.k1 = k1 #term saturation
        self.b = b #length normalization
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: list[dict] = [] #count term frequency per document
        self.idf: dict[str, float] = {} #idf per term
        self.doc_lens: list[int] = [] #number of tokens per document
   
    def fit(self, corpus:list[str]):
        """corpus: list of raw page texts."""
        tokenized = [tokenize (doc) for doc in corpus] #tokenize corpus
        self.corpus_size = len(tokenized) #corpus size calculate
        self.doc_lens = [len(d) for d in tokenized] #document lengths
        self.avgdl = sum(self.doc_lens) / max(1, self.corpus_size)
        self.doc_freqs = []
        #compute term frequencies + document frequencies
        df:dict[str, int] = defaultdict(int)
        for doc in tokenized:
            freq: dict[str, int] = defaultdict(int)
            for tok in doc:
                freq[tok] += 1
            self.doc_freqs.append(dict(freq))
            for tok in set(doc):
                df[tok] += 1
        #compute IDF
        for term, n in df.items():
            self.idf[term] = math.log(
                (self.corpus_size - n + 0.5) / (n + 0.5) + 1
            )
    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        score = 0.0
        dl = self.doc_lens[doc_idx]
        freq_map = self.doc_freqs[doc_idx]
        #iterate query tokens
        for tok in query_tokens:
            if tok not in freq_map:
                continue
            f = freq_map[tok]
            idf = self.idf.get(tok, 0.0)
            num = f * (self.k1 + 1) #boosts repeated terms
            den = f + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl)) #penalizes long documents
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
       
            # ColQwen2.5 specific
            if hasattr(outputs, "query_embeds") and outputs.query_embeds is not None:
                embedding = to_numpy(outputs.query_embeds[0])
            elif hasattr(outputs, "query_embeddings") and outputs.query_embeddings is not None:
                embedding = to_numpy(outputs.query_embeddings[0])
            elif isinstance(outputs, torch.Tensor):
                embedding = to_numpy(outputs[0])
            else:
                embedding = to_numpy(outputs)  # last resort
       
        # L2 normalize each token vector (ColQwen2.5 expects this)
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embedding = embedding / norms
   
        embed_time = time.time() - start
        print(f"Query embedded in {embed_time:.3f}s | Tokens: {embedding.shape[0]}")
   
        del inputs, outputs
        aggressive_cleanup()
        return embedding, embed_time
    
    def _ocr_page(self, point, generator) -> str:
        """OCR a page from a point's payload using generator's EasyOCR reader."""
        source = point.payload.get("source", "")
        page_num = point.payload.get("page_number", 0)
        try:
            if source.lower().endswith(".pdf"):
                pages = pdf_to_images(source)
                img = pages[page_num]
            else:
                img = Image.open(source)
            text = generator._extract_text(img)
            print(f"  OCR: Page {page_num} → {len(text)} chars extracted")
            return text
        except Exception as e:
            print(f"  OCR failed for page {page_num}: {e}")
            return ""


    def search(self, query_text: str, top_k: int = 15, source_filter: str = None, generator=None):
        start_search = time.time()
       
        # 1. Get multi-vector query embedding
        query_emb_array, embed_time = self._extract_text_embedding(query_text)
        num_query_tokens = query_emb_array.shape[0]
       
        query_multi_vec = query_emb_array.tolist()


        # 2. Optional source filter
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


        # 3. Query Qdrant (Multi-vector late interaction)
        retr_start = time.time()
        results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_multi_vec,
            using="image",                   # vector name used during indexing
            query_filter=query_filter,
            limit=top_k,
            score_threshold=None,             # lowered a bit for ColQwen2.5
        ).points
        retr_time = time.time() - retr_start


        # Average score by number of query tokens (common practice)
        for point in results:
            if hasattr(point, 'score') and point.score is not None:
                point.score = point.score / num_query_tokens


        # Group by page and keep best score per page
        page_best = defaultdict(lambda: {"score": -1.0, "point": None})
        for point in results:
            key = (point.payload.get("source"), point.payload.get("page_number"))
            if point.score > page_best[key]["score"]:
                page_best[key] = {"score": point.score, "point": point}


        # sorted_pages = sorted(page_best.values(), key=lambda x: x["score"], reverse=True)
        # top_results = [item["point"] for item in sorted_pages[:5]]
        sorted_pages = sorted(page_best.values(), key=lambda x: x["score"], reverse=True)


        # get more candidates before reranking
        initial_hits = [item["point"] for item in sorted_pages[:10]]

        print("\n Initial Candidates (before reranking):")
        for i, p in enumerate(initial_hits, 1):
            src = p.payload.get("source", "unknown")
            pg = p.payload.get("page_number", "?")
            score = p.score
            print(f"  {i}. {src} (Page {pg}) — score={score:.4f}")      
        # 5. Rerank (MANDATORY)
        if generator:
            final_hits = self.rerank_hits(query_text, initial_hits,generator, top_k=5)
        else:
            final_hits = initial_hits[:5]

        print("\n Top Retrieved Pages:")
        for i, p in enumerate(final_hits, 1):
            src = p.payload.get("source", "unknown")
            pg = p.payload.get("page_number", "?")
            score = p.score
            print(f"{i}. {src} (Page {pg}) — score={score:.4f}")


        aggressive_cleanup()
        return final_hits

    def rerank_hits(self, query: str, hits, generator, top_k: int = 5):
        """
        Multi-signal hybrid reranker.
 
        Signals
        -------
        1.  ColQwen2.5 late-interaction embedding score  (semantic)
        2.  BM25 on OCR text                             (lexical, tf-idf weighted)
        3.  Exact / partial keyword match                (surface form)
        4.  N-gram phrase match (bi/trigram)             (phrase precision)
        5.  Numeric / entity exact match                 (critical for tables, stats)
 
        Fusion: Reciprocal Rank Fusion (RRF) across all signals.
        """
        if not hits:
            return []
 
        # OCR every candidate page 
        ocr_texts: list[str] = []
        valid_hits = []
        for point in hits:
            text = self._ocr_page(point, generator)
            ocr_texts.append(text)
            valid_hits.append(point)
 
        # BM25 on OCR corpus lexical relevance
        bm25 = BM25()
        bm25.fit(ocr_texts)
        query_tokens = tokenize(query)
 
        bm25_scores = [
            bm25.score(query_tokens, i) for i in range(len(valid_hits))
        ]
 
        # Keyword signals
        query_lower = query.lower()
        raw_query_words = set(query_lower.split())
        content_words = raw_query_words - STOPWORDS  # only meaningful terms
 
        # numbers / numeric entities mentioned in the query
        query_numbers = extract_numbers(query_lower)
        query_tokens_raw = query_lower.split()
 
        keyword_scores: list[float] = []
        phrase_scores: list[float] = []
        number_scores: list[float] = []
 
        for text in ocr_texts:
            text_lower = text.lower()
            page_words = text_lower.split()
 
            #  Exact content-word matches (stopwords stripped)
            exact = sum(1 for w in content_words if w in text_lower)
 
            #  Partial / substring matches for content words
            partial = sum(
                1 for w in content_words
                if any(w in tok for tok in page_words)
                and w not in text_lower  # not already counted as exact
            )
 
            keyword_scores.append(exact * 2.0 + partial * 0.5)
 
            # N-gram phrase match (2- and 3-word phrases)
            phrase_bonus = 0.0
            for n in (2, 3):
                for i in range(len(query_tokens_raw) - n + 1):
                    phrase = " ".join(query_tokens_raw[i : i + n])
                    if phrase in text_lower:
                        phrase_bonus += n * 2.0   # longer phrase = stronger signal
            phrase_scores.append(phrase_bonus)
 
            #  Numeric / entity exact match
            if query_numbers:
                page_numbers = extract_numbers(text_lower)
                matched = len(query_numbers & page_numbers)
                number_scores.append(matched * 3.0)
            else:
                number_scores.append(0.0)
 
        # Reciprocal Rank Fusion (RRF)
        # k=60 is the standard RRF constant
        def rrf_ranks(scores: list[float], k: int = 60) -> list[float]:
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            ranks = [0.0] * len(scores)
            for rank, idx in enumerate(order):
                ranks[idx] = 1.0 / (k + rank + 1)
            return ranks
 
        emb_scores = [p.score for p in valid_hits]
        rrf_emb    = rrf_ranks(emb_scores)
        rrf_bm25   = rrf_ranks(bm25_scores)
        rrf_kw     = rrf_ranks(keyword_scores)
        rrf_phrase = rrf_ranks(phrase_scores)
        rrf_num    = rrf_ranks(number_scores)
 
        # Signal weights — tune based on your document types.
        # For dense-text docs raise w_bm25; for tables/figures raise w_num.
        w_emb    = 0.30
        w_bm25   = 0.25
        w_kw     = 0.20
        w_phrase = 0.15
        w_num    = 0.10
 
        scored = []
        for i, point in enumerate(valid_hits):
            final = (
                w_emb    * rrf_emb[i]
                + w_bm25   * rrf_bm25[i]
                + w_kw     * rrf_kw[i]
                + w_phrase * rrf_phrase[i]
                + w_num    * rrf_num[i]
            )
 
            pg = point.payload.get("page_number", "?")
            print(
                f"  Page {pg} | emb={emb_scores[i]:.4f} | "
                f"bm25={bm25_scores[i]:.3f} | kw={keyword_scores[i]:.2f} | "
                f"phrase={phrase_scores[i]:.2f} | num={number_scores[i]:.2f} | "
                f"RRF={final:.5f}"
            )
 
            # Store the fused score back on the point so callers can read it
            point.score = final
            scored.append((final, point))
 
        scored.sort(key=lambda x: x[0], reverse=True)
        final_hits = [p for _, p in scored[:top_k]]
 
        print(f"\n Reranking done: {len(hits)} candidates → {len(final_hits)} selected")
        return final_hits