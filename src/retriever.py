import torch
import numpy as np
import time
from collections import defaultdict


class MultimodalRetriever:
    def __init__(self, indexer):
        self.indexer = indexer

    def _extract_text_embedding(self, query_text: str):
        start = time.time()
        inputs = self.indexer.processor(
            text=[query_text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        ).to(self.indexer.device)

        with torch.no_grad():
            outputs = self.indexer.model.get_text_features(**inputs)
            embedding = outputs.pooler_output.squeeze(0).cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        embed_time = time.time() - start
        return embedding, embed_time

    def search(self, query_text: str, top_k: int = 15):
        start_search = time.time()
        
        query_vec, embed_time = self._extract_text_embedding(query_text)

        # Qdrant retrieval
        retr_start = time.time()
        results = self.indexer.client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_vec.tolist(),
            limit=top_k,
        ).points
        retr_time = time.time() - retr_start

        # Group best per page
        page_best = defaultdict(lambda: {"score": -1.0, "point": None})
        for point in results:
            key = (point.payload["source"], point.payload.get("page_number"))
            if point.score > page_best[key]["score"]:
                page_best[key] = {"score": point.score, "point": point}

        sorted_pages = sorted(page_best.values(), key=lambda x: x["score"], reverse=True)
        top_results = [item["point"] for item in sorted_pages[:5]]

        total_search_time = time.time() - start_search

        print(f"Query embed: {embed_time:.3f}s | Retrieval: {retr_time:.3f}s | Total search: {total_search_time:.3f}s")
        if top_results:
            # print(f"   Top score: {top_results[0].score:.4f}")

        return top_results