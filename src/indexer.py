import torch
import gc
import numpy as np
import time
from pathlib import Path
from PIL import Image
import shutil
# import pytesseract
import re
import sys
from transformers.utils.import_utils import is_flash_attn_2_available
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    MultiVectorConfig,
    MultiVectorComparator,
)
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers import AutoProcessor


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class MultimodalIndexer:
    def __init__(self, collection_name="mrag_collection", force_recreate=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.collection_name = collection_name
        self.model_name = "vidore/colqwen2.5-v0.2"

        self.chunk_size = 384
        self.overlap = 96
        self.stride = self.chunk_size - self.overlap

        print(f"Loading model  → {self.model_name}")
        print(f"Chunk size: {self.chunk_size}px | Overlap: {self.overlap}px")

        

        self.model = ColQwen2_5.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            # low_cpu_mem_usage=True,
            device_map="auto",
            attn_implementation ="flash_attention_2" if is_flash_attn_2_available() else None,
    
        ).eval()

        self.processor = ColQwen2_5_Processor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        print("Model and processor loaded successfully.")

        self.local_client = QdrantClient(":memory:")

        if force_recreate:
            self._recreate_collection()
        else:
            self._setup_collection()

   

    def _setup_collection(self):
        if self.local_client.collection_exists(self.collection_name):
            print(f" Using existing collection: {self.collection_name}")
            print(self.local_client.get_collection(self.collection_name))
            return

        # Detect embedding dimension
        print(" Detecting embedding dimension...")
        with torch.no_grad():
            dummy_img = Image.new("RGB", (224, 224))
            inputs = self.processor.process_images([dummy_img]).to(self.device)
            outputs = self.model(**inputs)
            embed_dim = outputs.image_embeds.shape[-1] if hasattr(outputs, "image_embeds") else 128

        print(f"Embedding dim: {embed_dim}")

        vectors_config = {
            "image": VectorParams(
                size=embed_dim,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM
                )
            )
        }

        self.local_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config
        )
        print(f" Successfully created collection '{self.collection_name}' with 'image' vector")

    def is_collection_empty(self) -> bool:
        if not self.local_client.collection_exists(self.collection_name):
            return True
        try:
            info = self.local_client.get_collection(self.collection_name)
            if info.points_count == 0:
                return True
            
            points, _ = self.local_client.scroll(
                collection_name=self.collection_name, limit=3, with_vectors=False
            )
            return len(points) == 0
        except Exception as e:
            print(f"Error checking collection: {e}")
            return True

    def _extract_image_embeddings(self, pil_img: Image.Image) -> np.ndarray:
        start = time.time()
        pil_img = pil_img.convert("RGB")

        inputs = self.processor.process_images([pil_img], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                embeddings = outputs.image_embeds[0]
            elif hasattr(outputs, "last_hidden_state"):
                embeddings = outputs.last_hidden_state[0]
            else:
                embeddings = outputs.hidden_states[-1][0] if hasattr(outputs, "hidden_states") else outputs[0]

            embeddings = embeddings.cpu().numpy().astype(np.float32)

        print(f"  → {embeddings.shape[0]} tokens | Time: {time.time() - start:.3f}s")

        del inputs, outputs
        aggressive_cleanup()
        return embeddings

    def _process_and_upsert(self, pil_img: Image.Image, source: str, page_num: int):
        start_page = time.time()
        w, h = pil_img.size
        points = []

        y_coords = list(range(0, max(1, h - self.chunk_size + 1), self.stride))
        if y_coords and y_coords[-1] + self.chunk_size < h:
            y_coords.append(h - self.chunk_size)

        x_coords = list(range(0, max(1, w - self.chunk_size + 1), self.stride))
        if x_coords and x_coords[-1] + self.chunk_size < w:
            x_coords.append(w - self.chunk_size)

        y_coords = sorted(set(y_coords))
        x_coords = sorted(set(x_coords))

        for y in y_coords:
            for x in x_coords:
                patch = pil_img.crop((x, y, x + self.chunk_size, y + self.chunk_size))
                multi_vector = self._extract_image_embeddings(patch)

                point_id = abs(hash(f"{source}_{page_num}_{x}_{y}")) % (10**15)

                points.append(
                    PointStruct(
                        id=point_id,
                        vector={"image": multi_vector.tolist()},
                        payload={
                            "page_number": page_num,
                            "source": str(source),
                            "x": x,
                            "y": y,
                            "chunk_size": self.chunk_size,
                            "num_tokens": int(multi_vector.shape[0])
                        }
                    )
                )

        if points:
            self.local_client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )

        page_time = time.time() - start_page
        print(f"Page {page_num} completed: {page_time:.2f}s ")
        aggressive_cleanup()
        return page_time

    def index_document(self, pdf_path: str):
        start_doc = time.time()
        images = pdf_to_images(pdf_path)
        print(f"\nProcessing PDF: {pdf_path} ({len(images)} pages)")

        total_time = 0.0
        for i, img in enumerate(images):
            page_time = self._process_and_upsert(img, pdf_path, i)
            total_time += page_time

        doc_time = time.time() - start_doc
        print(f"Finished PDF: {pdf_path} | Total: {doc_time:.2f}s | Avg/page: {doc_time/len(images):.2f}s\n")
        aggressive_cleanup()
        return doc_time

    def index_image(self, image_path: str):
        start = time.time()
        img = Image.open(image_path).convert("RGB")
        self._process_and_upsert(img, image_path, 0)
        total_time = time.time() - start
        aggressive_cleanup()
        print(f"Indexed image: {image_path} | Time: {total_time:.2f}s\n")

    def index_all_data(self, data_dir: str = "data"):
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
        aggressive_cleanup()
        print(f"ALL INDEXING COMPLETED in {total_index_time:.2f} seconds!\n")