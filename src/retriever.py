import torch
import numpy as np
import gc
import re
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
                img = get_pdf_page(source, page_num)
            else:
                img = Image.open(source)
            text = generator._extract_text(img)
            print(f"  OCR: Page {page_num} → {len(text)} chars extracted")
            return text
        except Exception as e:
            print(f"  OCR failed for page {page_num}: {e}")
            return ""


    def search(self, query_text: str, top_k: int = 10, source_filter: str = None, generator=None):
        start_search = time.time()
       
        # 1. Get ColQwen2.5 embedding
        query_emb_array, embed_time = self._extract_text_embedding(query_text)
        query_multi_vec = query_emb_array.tolist()

        # 2. Optional filter
        query_filter = Filter(must=[FieldCondition(key="source", match=MatchText(text=source_filter.lower()))]) if source_filter else None

        # 3. Retrieve more candidates from Qdrant (we'll rerank them)
        results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_multi_vec,
            using="image",
            query_filter=query_filter,
            limit=10,                   
            score_threshold=None,
        ).points

        # Normalize embedding score
        num_query_tokens = query_emb_array.shape[0]
        for point in results:
            if point.score is not None:
                # point.score = point.score / num_query_tokens
                page_tokens = point.payload.get("num_patches", 1)
                point.score = point.score / (num_query_tokens * np.log1p(page_tokens))

        print(f"Qdrant retrieval done in {time.time() - start_search:.2f}s | Candidates: {len(results)}")

        # 4. Hybrid Reranking (BM25 + Keywords + Numbers + Embedding)
        if generator:
            final_hits = self._hybrid_rerank(query_text, results, generator, top_k=top_k)
        else:
            final_hits = sorted(results, key=lambda x: x.score, reverse=True)[:top_k]

        total_time = time.time() - start_search
        print(f"Total search + rerank time: {total_time:.2f}s\n")

        return final_hits


    def _hybrid_rerank(self, query: str, hits, generator, top_k: int = 3):
        """Single-pass hybrid reranker (Embedding + BM25 + Keyword + Numeric)"""
        if not hits:
            return []

        # --- OCR once and cache ---
        ocr_texts = []
        for point in hits[:10]:
            text = self._ocr_page(point, generator)   # Still necessary unless you pre-compute
            ocr_texts.append(text)

        # --- BM25 ---
        bm25 = BM25()
        bm25.fit(ocr_texts)
        query_tokens = tokenize(query)

        bm25_scores = [bm25.score(query_tokens, i) for i in range(len(hits))]

        # --- Keyword, Phrase & Number signals ---
        query_lower = query.lower()
        content_words = set(query_lower.split()) - STOPWORDS
        query_numbers = extract_numbers(query_lower)
        query_tokens_raw = query_lower.split()

        keyword_scores = []
        phrase_scores = []
        number_scores = []

        for text in ocr_texts:
            text_lower = text.lower()
            page_words = text_lower.split()

            # Keyword match
            exact = sum(1 for w in content_words if w in text_lower)
            partial = sum(1 for w in content_words if any(w in tok for tok in page_words))
            keyword_scores.append(exact * 2.0 + partial * 0.5)

            # Phrase match
            phrase_bonus = 0.0
            for n in (2, 3):
                for i in range(len(query_tokens_raw) - n + 1):
                    phrase = " ".join(query_tokens_raw[i:i+n])
                    if phrase in text_lower:
                        phrase_bonus += n * 2.0
            phrase_scores.append(phrase_bonus)

            # Numeric match
            page_numbers = extract_numbers(text_lower)
            number_scores.append(len(query_numbers & page_numbers) * 3.0)

        # --- Combine scores using weighted sum (faster than RRF) ---
        emb_scores = [p.score for p in hits]

        final_scores = []
        for i in range(len(hits)):
            score = (
                0.35 * (emb_scores[i] / max(emb_scores)) +           # normalize embedding
                0.30 * (bm25_scores[i] / (max(bm25_scores) + 1e-8)) +
                0.20 * (keyword_scores[i] / (max(keyword_scores) + 1e-8)) +
                0.10 * (phrase_scores[i] / (max(phrase_scores) + 1e-8)) +
                0.05 * (number_scores[i] / (max(number_scores) + 1e-8))
            )
            final_scores.append(score)

            # Debug print
            pg = hits[i].payload.get("page_number", "?")
            print(f"Page {pg} | emb={emb_scores[i]:.4f} | bm25={bm25_scores[i]:.3f} | "
                  f"kw={keyword_scores[i]:.2f} | final={score:.5f}")

        # Sort and return top_k
        scored_hits = sorted(zip(final_scores, hits), key=lambda x: x[0], reverse=True)
        final_hits = [point for _, point in scored_hits[:top_k]]

        print(f"Hybrid reranking completed → {len(final_hits)} final pages")
        return final_hits