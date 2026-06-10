import transformers.integrations.peft as _ti

_original_convert = _ti._convert_peft_config_moe

def _patched_convert_peft_config_moe(peft_config, model_type):
    mapping = getattr(_ti, '_MOE_TARGET_MODULE_MAPPING', {})
    if model_type not in mapping:
        return peft_config
    return _original_convert(peft_config, model_type)

_ti._convert_peft_config_moe = _patched_convert_peft_config_moe


import torch
import gc
import numpy as np
import time
from pathlib import Path
from PIL import Image
import pytesseract

from src.utils import pdf_to_images

from transformers.utils.import_utils import is_flash_attn_2_available

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    MultiVectorConfig,
    MultiVectorComparator,
    Filter,
    FieldCondition,
    MatchValue,
)

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


# ---------------------------
# Utilities
# ---------------------------

def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    return np.asarray(x, dtype=np.float32)


# ---------------------------
# Indexer
# ---------------------------

class MultimodalIndexer:

    def __init__(self, collection_name="mrag_collection", force_recreate=False):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.collection_name = collection_name
        self.model_name = "vidore/colqwen2.5-v0.2"

        self.chunk_size = 512
        self.overlap = 160
        self.stride = self.chunk_size - self.overlap

        print(f"Loading model: {self.model_name}")

        self.model = ColQwen2_5.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="flash_attention_2"
            if is_flash_attn_2_available()
            else None,
        ).eval()

        self.processor = ColQwen2_5_Processor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.local_client = QdrantClient(
            path="/content/drive/MyDrive/final_qdrant_db"
        )

        if force_recreate:
            self.local_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "image": VectorParams(
                        size=128,
                        distance=Distance.COSINE,
                        multivector_config=MultiVectorConfig(
                            comparator=MultiVectorComparator.MAX_SIM
                        ),
                    )
                },
            )
        else:
            self._setup_collection()

    # ---------------------------
    # Collection setup
    # ---------------------------

    def _setup_collection(self):
        if self.local_client.collection_exists(self.collection_name):
            print("Using existing collection")
            return

        dummy = Image.new("RGB", (224, 224))
        inputs = self.processor.process_images([dummy]).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs)
            dim = out.image_embeds.shape[-1]

        self.local_client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "image": VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                )
            },
        )

    # ---------------------------
    # PATCH CHECK (CRITICAL)
    # ---------------------------

    def patch_exists(self, source, page, x, y):
        try:
            res, _ = self.local_client.scroll(
                collection_name=self.collection_name,
                limit=1,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=str(source)),
                        ),
                        FieldCondition(
                            key="page_number",
                            match=MatchValue(value=page),
                        ),
                        FieldCondition(
                            key="x",
                            match=MatchValue(value=x),
                        ),
                        FieldCondition(
                            key="y",
                            match=MatchValue(value=y),
                        ),
                    ]
                ),
            )
            return len(res) > 0
        except:
            return False

    # ---------------------------
    # Embeddings
    # ---------------------------

    def _extract_embeddings(self, img: Image.Image):
        img = img.convert("RGB")
        inputs = self.processor.process_images([img]).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs)

            if hasattr(out, "image_embeds"):
                emb = out.image_embeds[0]
            else:
                emb = out[0]

        return to_numpy(emb)

    # ---------------------------
    # OCR
    # ---------------------------

    def _ocr(self, img):
        try:
            return pytesseract.image_to_string(img).strip()
        except:
            return ""

    # ---------------------------
    # CORE PATCH PROCESSING
    # ---------------------------

    def _process_page(self, img, source, page_num):

        page_ocr = self._ocr(img)

        w, h = img.size

        y_coords = list(range(0, max(1, h - self.chunk_size + 1), self.stride))
        x_coords = list(range(0, max(1, w - self.chunk_size + 1), self.stride))

        points = []

        for y in y_coords:
            for x in x_coords:

                # 🔥 RESUME LOGIC (KEY FIX)
                if self.patch_exists(source, page_num, x, y):
                    continue

                patch = img.crop((x, y, x + self.chunk_size, y + self.chunk_size))

                patch_ocr = self._ocr(patch)
                if len(patch_ocr) < 20:
                    patch_ocr = ""

                emb = self._extract_embeddings(patch)

                pid = abs(hash(f"{source}_{page_num}_{x}_{y}")) % (10**15)

                points.append(
                    PointStruct(
                        id=pid,
                        vector={"image": emb.tolist()},
                        payload={
                            "source": str(source),
                            "page_number": page_num,
                            "x": x,
                            "y": y,
                            "num_tokens": int(emb.shape[0]),
                            "page_ocr": page_ocr,
                            "patch_ocr": patch_ocr,
                        },
                    )
                )

        if points:
            self.local_client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    # ---------------------------
    # DOCUMENT INDEXING (FIXED)
    # ---------------------------

    def index_document(self, pdf_path: str):

        images = pdf_to_images(pdf_path)

        print(f"Indexing {pdf_path} | pages={len(images)}")

        for i, img in enumerate(images):
            print(f"Page {i} processing...")
            self._process_page(img, pdf_path, i)

        print("Done indexing document.")

    # ---------------------------
    # IMAGE INDEXING
    # ---------------------------

    def index_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        self._process_page(img, image_path, 0)

    # ---------------------------
    # DIRECTORY INDEXING
    # ---------------------------

    def index_all(self, data_dir="data"):

        path = Path(data_dir)

        for f in path.rglob("*"):
            if f.suffix.lower() == ".pdf":
                self.index_document(str(f))
            elif f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                self.index_image(str(f))

    # ---------------------------
    # CLOSE
    # ---------------------------

    def close(self):
        if self.local_client:
            self.local_client.close()