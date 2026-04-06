# src/retrieval.py

import torch    
from colpali_engine.models import ColPali, ColPaliProcessor
from qdrant_client import QdrantClient

qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION = "colpali_docs"

model = ColPali.from_pretrained("vidore/colpali-v1.3").eval()
processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3")

def retrieve(query: str, top_k: int = 3):
    query_inputs = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        q_embeddings = model(**query_inputs)

    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=q_embeddings.cpu().numpy().tolist(),
        limit=top_k
    )

    hits = [(r.payload["filename"], r.score) for r in results]
    return hits