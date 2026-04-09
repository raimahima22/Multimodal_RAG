import torch
import gc
import numpy as np
import time
from pathlib import Path
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoModel, AutoProcessor
from src.utils import pdf_to_images


class MultimodalIndexer:
    def __init__(self, collection_name="mrag_collection", force_recreate=False):
        self.device = "cpu"
        self.collection_name = collection_name
        self.model_id = "google/siglip2-so400m-patch16-naflex"
        
        self.chunk_size = 512
        self.overlap = 128
        self.stride = self.chunk_size - self.overlap

        print(f" Loading SigLIP2 on CPU → {self.model_id}")

        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        print(" Model and processor loaded.")

        self.client = QdrantClient(path="qdrant_db")
        
        if force_recreate:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
                print(f" Deleted existing collection: {self.collection_name}")
        
        self._setup_collection()

    def _setup_collection(self):
        vector_size = 1152
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Created new collection: {self.collection_name}")
        else:
            print(f" Using existing collection: {self.collection_name}")

    def _extract_image_embedding(self, pil_img):
        """Extract normalized image embedding and print timing per patch"""
        start = time.time()
        
        pil_img = pil_img.convert("RGB")
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            embedding = outputs.pooler_output[0].cpu().numpy().astype(np.float32)
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
        embed_time = time.time() - start
        # print(f"    → Patch embed: {embed_time:.3f}s", end=" ")   # Print per patch
        
        return embedding

    def _process_and_upsert(self, pil_img, source, page_num):
        start_page = time.time()
        w, h = pil_img.size
        points = []

        # Generate sliding window coordinates
        y_coords = list(range(0, max(1, h - self.chunk_size + 1), self.stride))
        if y_coords and y_coords[-1] + self.chunk_size < h:
            y_coords.append(h - self.chunk_size)

        x_coords = list(range(0, max(1, w - self.chunk_size + 1), self.stride))
        if x_coords and x_coords[-1] + self.chunk_size < w:
            x_coords.append(w - self.chunk_size)

        # Remove duplicates if any
        y_coords = sorted(set(y_coords))
        x_coords = sorted(set(x_coords))

        for y in y_coords:
            for x in x_coords:
                patch = pil_img.crop((x, y, x + self.chunk_size, y + self.chunk_size))
                
                vector = self._extract_image_embedding(patch)

                point_id = abs(hash(f"{source}_{page_num}_{x}_{y}")) % (10**15)

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector.tolist(),          # Fixed: convert to list
                        payload={
                            "page_number": page_num,
                            "source": str(source),
                            "x": x,
                            "y": y,
                            "chunk_size": self.chunk_size
                        }
                    )
                )

        # Upsert
        upsert_start = time.time()
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
        upsert_time = time.time() - upsert_start

        page_time = time.time() - start_page

        print(f" Page total: {page_time:.2f}s")

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return page_time                      # Important: return the time

    def index_document(self, pdf_path):
        start_doc = time.time()
        images = pdf_to_images(pdf_path)
        print(f"\n Processing PDF: {pdf_path} ({len(images)} pages)")

        total_time = 0.0
        for i, img in enumerate(images):
            page_time = self._process_and_upsert(img, pdf_path, i)
            total_time += page_time
            
        doc_time = time.time() - start_doc
        print(f" Finished PDF: {pdf_path} | Total: {doc_time:.2f}s | Avg/page: {doc_time/len(images):.2f}s\n")
        return doc_time

    def index_image(self, image_path):
        start = time.time()
        img = Image.open(image_path)
        self._process_and_upsert(img, image_path, 0)
        total_time = time.time() - start
        print(f" Indexed image: {image_path} | Time: {total_time:.2f}s\n")

    def index_all_data(self, data_dir="data"):
        start_total = time.time()
        print("Starting full indexing...\n")
        
        data_path = Path(data_dir)
        for file_path in data_path.rglob("*"):
            if file_path.suffix.lower() in [".pdf", ".jpg", ".jpeg", ".png"]:
                if file_path.suffix.lower() == ".pdf":
                    self.index_document(str(file_path))
                else:
                    self.index_image(str(file_path))
        
        total_index_time = time.time() - start_total
        print(f" ALL INDEXING COMPLETED in {total_index_time:.2f} seconds!\n")