import torch
import gc
import numpy as np
import time
from pathlib import Path
from PIL import Image
import fitz

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    MultiVectorConfig, MultiVectorComparator,
)

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class MultimodalIndexer:

    def __init__(self, collection_name="mrag_collection", force_recreate=False):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.collection_name = collection_name
        self.model_name = "vidore/colqwen2.5-v0.2"

        self.chunk_size = 384
        self.overlap = 96
        self.stride = self.chunk_size - self.overlap

        self.BATCH_SIZE = 64   # 🔥 increase if VRAM allows
        self.UPSERT_BATCH = 256

        print(f"Loading model → {self.model_name}")

        self.model = ColQwen2_5.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="flash_attention_2"
            if is_flash_attn_2_available() else None,
        ).eval()

        self.processor = ColQwen2_5_Processor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.client = QdrantClient(path="/content/drive/MyDrive/qdrant_db")

        if force_recreate:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=self._get_vector_config()
            )
        else:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=self._get_vector_config()
                )

    # -------------------------

    def _get_vector_config(self):
        dummy = Image.new("RGB", (224, 224))
        inputs = self.processor.process_images([dummy]).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs)

        dim = out.image_embeds.shape[-1]

        return {
            "image": VectorParams(
                size=dim,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM
                )
            )
        }

    # -------------------------

    def _get_coords(self, w, h):
        xs = list(range(0, max(1, w - self.chunk_size + 1), self.stride))
        ys = list(range(0, max(1, h - self.chunk_size + 1), self.stride))

        if xs[-1] + self.chunk_size < w:
            xs.append(w - self.chunk_size)
        if ys[-1] + self.chunk_size < h:
            ys.append(h - self.chunk_size)

        return [(x, y) for y in ys for x in xs]

    # -------------------------

    def _process_page(self, img, source, page_num):

        t0 = time.time()

        w, h = img.size
        coords = self._get_coords(w, h)

        # ✅ STEP 1: crop ALL patches once
        patches = [
            img.crop((x, y, x+self.chunk_size, y+self.chunk_size))
            for x, y in coords
        ]

        # ✅ STEP 2: preprocess ONCE
        t_pre = time.time()
        inputs_all = self.processor.process_images(patches)
        print(f"Preprocess: {time.time()-t_pre:.2f}s")

        all_points = []

        # ✅ STEP 3: batched GPU inference
        for i in range(0, len(coords), self.BATCH_SIZE):

            batch_inputs = {
                k: v[i:i+self.BATCH_SIZE].to(self.device)
                for k, v in inputs_all.items()
            }

            with torch.no_grad():
                outputs = self.model(**batch_inputs)
                embeds = outputs.image_embeds.detach().cpu()

            # ✅ STEP 4: convert to points
            for j in range(embeds.shape[0]):
                x, y = coords[i + j]

                all_points.append(
                    PointStruct(
                        id=abs(hash(f"{source}_{page_num}_{x}_{y}")) % (10**15),
                        vector={"image": embeds[j].tolist()},
                        payload={
                            "page_number": page_num,
                            "source": source,
                            "x": x,
                            "y": y,
                            "chunk_size": self.chunk_size,
                            "num_tokens": int(embeds[j].shape[0])
                        }
                    )
                )

            del batch_inputs, outputs, embeds

        # ✅ STEP 5: batched upsert
        for i in range(0, len(all_points), self.UPSERT_BATCH):
            self.client.upsert(
                collection_name=self.collection_name,
                points=all_points[i:i+self.UPSERT_BATCH],
                wait=False
            )

        if hasattr(img, "close"):
            img.close()

        page_time = time.time() - t0
        print(f"Page {page_num} | {len(coords)} patches | {page_time:.2f}s")

        return page_time

    # -------------------------

    def index_document(self, pdf_path):

        print(f"\nProcessing: {pdf_path}")

        doc = fitz.open(pdf_path)
        total = 0

        for i in range(len(doc)):
            page = doc[i]

            mat = fitz.Matrix(250/72, 250/72)
            pix = page.get_pixmap(matrix=mat)

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            total += self._process_page(img, pdf_path, i)

        doc.close()

        print(f"Total time: {total:.2f}s | Avg/page: {total/len(doc):.2f}s")

    # -------------------------

    def index_all_data(self, data_dir="data"):

        start = time.time()

        for file in Path(data_dir).rglob("*"):
            if file.suffix.lower() == ".pdf":
                self.index_document(str(file))
            elif file.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                img = Image.open(file).convert("RGB")
                self._process_page(img, str(file), 0)

        print(f"TOTAL TIME: {time.time()-start:.2f}s")

    # -------------------------

    def close(self):
        self.client.close()