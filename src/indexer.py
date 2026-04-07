import torch
import gc
import numpy as np
from pathlib import Path
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
# from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import AutoProcessor, AutoModel
from transformers import SiglipModel, SiglipProcessor
from src.utils import pdf_to_images

class MultimodalIndexer:
    def __init__(self, collection_name="mrag_collection"):
        self.device = "cpu" 
        self.collection_name = collection_name
        
        # Consistent with https://huggingface.co/google/siglip-base-patch16-224
        model_id = "google/siglip-base-patch16-224"
        
        print(f" Loading SigLIP on {self.device}...")

        # self.model = AutoModel.from_pretrained(
        #     model_id, 
        #     dtype=torch.float32,
        #     low_cpu_mem_usage=True 
        # ).to(self.device).eval()
        self.model = SiglipModel.from_pretrained(model_id).to(self.device).eval()

        self.processor = SiglipProcessor.from_pretrained(model_id)

        # Using local path to avoid connection issues if Docker isn't running
        self.client = QdrantClient(path="qdrant_db") 
        self._setup_collection()

    def _setup_collection(self):
        # google/siglip-base-patch16-224 hidden_size is 768
        # Note: If using ColPali, it projects to 128. Pure SigLIP-base is 768.
        vector_size = 768 

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size, 
                    distance=Distance.COSINE,
                    # multivector_config={"comparator": "max_sim"} 
                )
            )

    def _process_and_upsert(self, pil_img, source, page_num):
        # SigLIP documentation recommends 224x224 for this specific checkpoint
        # We ensure it's RGB
        pil_img = pil_img.convert("RGB").resize((224, 224))

        # FIX: SigLIP processor uses the standard __call__ interface
        # Reference: https://huggingface.co/docs/transformers/model_doc/siglip
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # outputs = self.model.get_image_features(**inputs).last_hidden_state
            # For multivector max_sim, we want the sequence of patches
            # We use last_hidden_state if we want the 'Late Interaction' behavior
            # full_output = self.model.vision_model(**inputs).last_hidden_state
            outputs = self.model.get_image_features(**inputs)
            # vector = outputs.detach().cpu().numpy().flatten().tolist()
            if isinstance(outputs, torch.Tensor):
                vector = outputs[0].detach().cpu().numpy().flatten().tolist()
            else:
        # fallback (if output is a model object)
                vector = outputs.pooler_output[0].detach().cpu().numpy().flatten().tolist()
        
        # full_output shape: [batch, num_patches, hidden_size]
        # We take the first item in batch and convert to list
        # vector = full_output[0].cpu().float().numpy().tolist()
        
        point_id = abs(hash(f"{source}_{page_num}")) % (10**15)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={"page_number": page_num, "source": source}
            )
        ]
    )
        
        # Cleanup
        del outputs
        # del full_output
        del inputs
        gc.collect()

    def index_all_data(self, data_dir="data"):
        data_path = Path(data_dir)
        for file_path in data_path.rglob("*"):
            if file_path.suffix.lower() in [".pdf", ".jpg", ".png"]:
                if file_path.suffix.lower() == ".pdf":
                    self.index_document(str(file_path))
                else:
                    self.index_image(str(file_path))

    def index_document(self, pdf_path):
        images = pdf_to_images(pdf_path)
        for i, img in enumerate(images):
            self._process_and_upsert(img, pdf_path, i)
        print(f"Indexed {pdf_path}")

    def index_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        self._process_and_upsert(img, image_path, 0)
        print(f"Indexed image {image_path}")