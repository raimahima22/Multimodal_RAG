# src/indexing.py

import os
import torch
from PIL import Image

from colpali_engine.models import ColPali, ColPaliProcessor
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

# Load ColPali & Processor
MODEL_NAME = "vidore/colpali-v1.3"

model = ColPali.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).eval()
processor = ColPaliProcessor.from_pretrained(MODEL_NAME)

# Setup Vector DB (Qdrant)
qdrant = QdrantClient(url="http://localhost:6333")
qdrant.recreate_collection(
    collection_name="colpali_docs",
    vectors=VectorParams(size=model.config.hidden_size, distance="Cosine")
)

DOC_DIR = "data/docs"

def index_documents():
    for fname in os.listdir(DOC_DIR):
        if not fname.lower().endswith(".png"):
            continue

        path = os.path.join(DOC_DIR, fname)
        image = Image.open(path).convert("RGB")

        inputs = processor.process_images([image]).to(model.device)
        with torch.no_grad():
            embeddings = model(**inputs)

        # embeddings is multi‑vector per image page
        vector_list = embeddings.cpu().numpy().tolist()

        qdrant.upsert(
            collection_name="colpali_docs",
            points=[{
                "id": fname,
                "vector": vector_list,
                "payload": {"filename": fname}
            }]
        )

    print("Indexing complete!")