import torch
import numpy as np
import time
from collections import defaultdict


class MultimodalRetriever:
    def __init__(self, indexer):
        self.indexer = indexer

    def _extract_text_embedding(self, query_text: str):
        """Extract normalized text embedding for ColSmol / ColPali models"""
        start = time.time()
        
        inputs = self.indexer.processor(
            text=[query_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.indexer.device)

        with torch.no_grad():
            outputs = self.indexer.model(**inputs)

            # === ColSmol / ColPali family handling ===
            if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
                # Most common case for these models when only text is passed
                embedding = outputs.text_embeds[0]          # shape: (embed_dim,)
            
            elif hasattr(outputs, "last_hidden_state"):
                # Fallback: some models still return last_hidden_state
                # Take the [CLS] token or mean pool (last token is often not best)
                last_hidden = outputs.last_hidden_state[0]   # (seq_len, hidden_size)
                embedding = last_hidden[-1]                  # last token (common heuristic)
                # Alternative (often better): mean pooling
                # embedding = last_hidden.mean(dim=0)
            
            elif isinstance(outputs, torch.Tensor):
                # Some lightweight / custom ColSmol-500M variants directly return the tensor
                # Common patterns:
                if outputs.dim() == 3:          # (batch, seq_len, dim) → take last token
                    embedding = outputs[0, -1]
                elif outputs.dim() == 2:        # (batch, dim) → already pooled
                    embedding = outputs[0]
                else:
                    raise ValueError(f"Unexpected output tensor shape: {outputs.shape}")
            
            else:
                raise AttributeError(
                    f"Model output has neither 'text_embeds' nor 'last_hidden_state'. "
                    f"Got type: {type(outputs)}. "
                    f"Available attributes: {dir(outputs)}"
                )

            # Move to CPU and convert to numpy
            embedding = embedding.cpu().numpy().astype(np.float32)

            # L2 normalization (very important for cosine similarity / Qdrant)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        embed_time = time.time() - start
        return embedding, embed_time

    def search(self, query_text: str, top_k: int = 15):
        start_search = time.time()
        
        # Get the query embedding (we'll fix this next)
        query_vec, embed_time = self._extract_text_embedding(query_text)

        retr_start = time.time()
        results = self.indexer.local_client.query_points(
            collection_name=self.indexer.collection_name,
            query=[query_vec.tolist()],     
            using="image",               
            limit=top_k,
        ).points
        retr_time = time.time() - retr_start

        # Group best result per page 
        page_best = defaultdict(lambda: {"score": -1.0, "point": None})
        for point in results:
            key = (point.payload.get("source"), point.payload.get("page_number"))
            if point.score > page_best[key]["score"]:
                page_best[key] = {"score": point.score, "point": point}

        sorted_pages = sorted(page_best.values(), key=lambda x: x["score"], reverse=True)
        top_results = [item["point"] for item in sorted_pages[:5]]

        total_time = time.time() - start_search
        print(f"Query embed: {embed_time:.3f}s | Retrieval: {retr_time:.3f}s | Total: {total_time:.3f}s")

        return top_results