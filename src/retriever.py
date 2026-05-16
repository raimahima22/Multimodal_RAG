import torch
import numpy as np
import gc
import re
import math
from collections import defaultdict
from qdrant_client.models import Filter, FieldCondition, MatchText

VERBOSE = True

#stopwords defined
STOPWORDS = {
    "what", "is", "are", "the", "a", "an",
    "in", "on", "at", "to", "of",
    "and", "or"
}

# utility functions

def aggressive_cleanup():
    """
    Aggressively clear Python and CUDA memory.

    Useful after embedding generation or retrieval operations to
    reduce GPU memory usage and avoid out-of-memory errors.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def to_numpy(x):
    """
    Convert tensor or array-like object to NumPy array.

    Args:
        x:
            Torch tensor or NumPy-compatible object.

    Returns:
        np.ndarray
    """
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    return np.asarray(x, dtype=np.float32)

def tokenize(text: str):
    """
    Tokenize and normalize text.

    Processing steps:
    -----------------
    1. Lowercase conversion
    2. Extract alphanumeric tokens
    3. Remove stopwords

    Args:
        text (str)

    Returns:
        List[str]
    """
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]

def normalize_query(text: str):
    """
    Normalize query into a clean tokenized string.

    Args:
        text (str)

    Returns:
        str
    """
    return " ".join(tokenize(text))

def extract_numbers(text: str):
    """
    Extract numeric patterns from text.
    """
    return set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text))


def minmax(x):
    """
    Perform min-max normalization
    """
    if not x:
        return x
    mn, mx = min(x), max(x)
    if abs(mx - mn) < 1e-8:
        return [1.0 for _ in x]
    return [(v - mn) / (mx - mn + 1e-8) for v in x]



class BM25:
    """
    Lightweight BM25 retriever for OCR text reranking.

    BM25 is a probabilistic ranking algorithm widely used in
    information retrieval systems.

    Used here to rerank OCR text after vector retrieval.
    """
    def __init__(self, k1=1.5, b=0.75):
        """
        Initialize BM25 parameters.

        Args:
            k1 (float):
                Controls term frequency scaling.

            b (float):
                Controls document length normalization.
        """
        self.k1 = k1
        self.b = b

    def fit(self, corpus):
        """
        Build BM25 statistics from corpus.

        Args:
            corpus (List[str]):
                OCR texts from candidate patches/pages.
        """
        self.tokenized = [tokenize(doc) for doc in corpus] # tokenize corpus
        self.doc_lens = [len(d) for d in self.tokenized] #document lengths
        self.avgdl = sum(self.doc_lens) / max(1, len(self.tokenized))

        df = defaultdict(int) #document frequencies
        self.freqs = []

        for doc in self.tokenized:
            freq = defaultdict(int)
            for t in doc:
                freq[t] += 1
            self.freqs.append(freq)
            for t in set(doc):
                df[t] += 1
        
        #IDF computation
        self.idf = {
            t: math.log((len(self.tokenized) - n + 0.5) / (n + 0.5) + 1)
            for t, n in df.items()
        }

    def score(self, query_tokens, i):
        """
        Compute BM25 score for a document.

        Args:
            query_tokens (List[str])
            i (int):
                Document index

        Returns:
            float
        """
        freq = self.freqs[i]
        dl = self.doc_lens[i]
        score = 0.0

        for t in query_tokens:
            if t not in freq:
                continue
            f = freq[t]
            idf = self.idf.get(t, 0.0)

            score += idf * (f * (self.k1 + 1)) / (
                f + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl))
            )

        return score


class MultimodalRetriever:
    """
    Hybrid multimodal retriever using:

    1. ColQwen2.5 embeddings
    2. Qdrant vector search
    3. OCR-based BM25 reranking
    4. Keyword matching
    5. Phrase matching
    6. Numeric matching

    Retrieval Pipeline:
    -------------------
    Query
      ↓
    Text Embedding
      ↓
    Qdrant Multi-vector Search
      ↓
    OCR-based Hybrid Reranking
      ↓
    Page Aggregation
      ↓
    Final Top Pages
    """
    def __init__(self, indexer):
        self.indexer = indexer

    # embedding 
    def _extract_text_embedding(self, query_text):
        """
        Generate query embeddings using ColQwen2.5.

        Args:
            query_text (str)

        Returns:
            np.ndarray
        """
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
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        return emb


    # search function

    def search(self, query_text, top_k=3, source_filter=None):
        """
        Perform hybrid multimodal retrieval.

        Pipeline:
        ---------
        1. Normalize query
        2. Generate query embeddings
        3. Retrieve candidates from Qdrant
        4. OCR-based reranking
        5. Aggregate page results
        6. Return top pages

        Args:
            query_text (str):
                User query.

            top_k (int):
                Number of final pages to return.

            source_filter (str | None):
                Restrict retrieval to a specific source file.

        Returns:
            List[ScoredPoint]
        """


        print("\nQUERY:", query_text)

        clean_query = normalize_query(query_text)
        q_tokens = tokenize(clean_query)
        query_nums = extract_numbers(query_text)

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

        #qdrant retrieval

        results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_vec,
            using="image",
            query_filter=query_filter,
            limit=60,
        ).points

        # normalize embedding score
        for p in results:
            if p.score is not None:
                p.score /= emb.shape[0]

        hits = sorted(results, key=lambda x: x.score, reverse=True)[:25]

        print(f"\nQdrant retrieval done | Candidates: {len(hits)}")

        # rerank
        ocr_texts=[]
        for h in hits:
            patch_ocr = h.payload.get("patch_ocr", "")
            page_ocr = h.payload.get("page_ocr", "")
            
            # Smart selection: prefer patch if meaningful, else use page
            if len(patch_ocr) > 40:
                ocr_texts.append(patch_ocr)
            else:
                ocr_texts.append(page_ocr)

        

        bm25 = BM25()
        bm25.fit(ocr_texts)

        # additional scoring signals
        bm25_scores = [bm25.score(q_tokens, i) for i in range(len(hits))]
        emb_scores = [h.score for h in hits]
        kw_scores = []
        phrase_scores = []
        num_scores = []

        for text in ocr_texts:
            t = text.lower()

            kw = sum(1 for w in q_tokens if w in t)
            kw_scores.append(kw)

            phrase = 0
            for n in (2, 3):
                for i in range(len(q_tokens) - n + 1):
                    if " ".join(q_tokens[i:i+n]) in t:
                        phrase += 1
            phrase_scores.append(phrase)

            nums = extract_numbers(t)
            num_scores.append(len(nums & query_nums))

        # for i in range(len(hits)):
        #     bm25_scores.append(bm25.score(q_tokens, i))

        # emb_scores = [h.score for h in hits]

        # normalize the scores
        emb_n = minmax(emb_scores)
        bm_n = minmax(bm25_scores)
        kw_n = minmax(kw_scores)
        ph_n = minmax(phrase_scores)
        nm_n = minmax(num_scores)

        # final weighted score
        final_scores = []
        for i in range(len(hits)):
            score = (
                0.50 * emb_n[i] +
                0.20 * bm_n[i] +
                0.15 * kw_n[i] +
                0.10 * ph_n[i] +
                0.05 * nm_n[i]
            )
            final_scores.append(score)

        ranked = sorted(list(enumerate(final_scores)), key=lambda x: x[1], reverse=True)

        print("\n================ INITIAL RERANKED RESULTS (TOP 10) ================\n")
        for i, (idx, score) in enumerate(ranked[:10], 1):
            h = hits[idx]
            print(f"{i}. Page {h.payload['page_number']} | Score={score:.5f}")

        #page aggregation
        # Multiple patches may belong to the same page.
        # Keep only the highest-scoring patch per page.

        page_best = {}

        for idx, score in ranked:
            h = hits[idx]
            key = (h.payload["source"], h.payload["page_number"])
            #keep only the best patch for unique pages
            if key not in page_best or score > page_best[key]["score"]:
                page_best[key] = {
                    "score": score,
                    "hit": h
                }

        final_pages = sorted(page_best.values(), key=lambda x: x["score"], reverse=True) #sort pages by score

        

        print("\n================ FINAL TOP PAGES ================\n")

        for i, p in enumerate(final_pages[:top_k], 1):
            page = p["hit"].payload["page_number"]
            print(f"{i}. Page {page} | FINAL={p['score']:.5f}")

        aggressive_cleanup()

        #return only hits
        return [p["hit"] for p in final_pages[:top_k]]