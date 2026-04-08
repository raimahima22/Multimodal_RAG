import torch
import gc
import numpy as np
from pathlib import Path
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import SiglipModel, SiglipProcessor
from src.utils import pdf_to_images

class MultimodalIndexer:
    def __init__(self, collection_name="mrag_collection"):
        self.device = "cpu" 
        self.collection_name = collection_name
        
        # SigLIP configuration
        model_id = "google/siglip2-base-patch16-224"
        self.chunk_size = 224
        self.overlap = 44 # Standard overlap to prevent cutting text in half
        self.stride = self.chunk_size - self.overlap # 180 pixels

        print(f"Loading SigLIP on {self.device}...")
        self.model = SiglipModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = SiglipProcessor.from_pretrained(model_id)

        self.client = QdrantClient(path="qdrant_db") 
        self._setup_collection()

    def _setup_collection(self):
        vector_size = 768 
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size, 
                    distance=Distance.COSINE
                )
            )

    def _process_and_upsert(self, pil_img, source, page_num):
        pil_img = pil_img.convert("RGB")
        w, h = pil_img.size
        points = []

        # Calculate tiling coordinates to cover the whole image
        # This handles the bottom/right edges by "snapping" the last tile to the edge
        y_coords = list(range(0, h - self.chunk_size + 1, self.stride))
        if not y_coords or y_coords[-1] + self.chunk_size < h:
            y_coords.append(max(0, h - self.chunk_size))
            
        x_coords = list(range(0, w - self.chunk_size + 1, self.stride))
        if not x_coords or x_coords[-1] + self.chunk_size < w:
            x_coords.append(max(0, w - self.chunk_size))

        for y in y_coords:
            for x in x_coords:
                # 1. Extract the high-res chunk
                patch = pil_img.crop((x, y, x + self.chunk_size, y + self.chunk_size))
                
                # 2. Process through SigLIP
                inputs = self.processor(images=patch, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                    
                    # SigLIP returns a pooler_output or raw tensor depending on version
                    if hasattr(outputs, "pooler_output"):
                        vector = outputs.pooler_output[0].cpu().numpy().tolist()
                    else:
                        vector = outputs[0].cpu().numpy().tolist()

                # 3. Create unique ID including coordinates
                point_id = abs(hash(f"{source}_{page_num}_{x}_{y}")) % (10**15)
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "page_number": page_num, 
                            "source": source,
                            "x": x,
                            "y": y
                        }
                    )
                )

        # Batch upsert all tiles for this page
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        gc.collect()

    def index_all_data(self, data_dir="data"):
        data_path = Path(data_dir)
        for file_path in data_path.rglob("*"):
            if file_path.suffix.lower() in [".pdf", ".jpg", ".png"]:
                self.index_document(str(file_path)) if file_path.suffix.lower() == ".pdf" else self.index_image(str(file_path))

    def index_document(self, pdf_path):
        images = pdf_to_images(pdf_path)
        for i, img in enumerate(images):
            self._process_and_upsert(img, pdf_path, i)
        print(f"Indexed {pdf_path} (spatial tiles)")

    def index_image(self, image_path):
        img = Image.open(image_path)
        self._process_and_upsert(img, image_path, 0)
        print(f"Indexed image {image_path} (spatial tiles)")