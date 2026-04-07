import torch
import numpy as np

class MultimodalRetriever:
    def __init__(self, indexer):
        self.indexer = indexer

    def search(self, query_text, top_k=1):
        # SigLIP uses the processor to encode text
        inputs = self.indexer.processor(
            text=[query_text], 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.indexer.device)
        
        with torch.no_grad():
            outputs = self.indexer.model.get_text_features(**inputs)
            print(type(outputs))  # should be <class 'transformers.modeling_outputs.BaseModelOutputWithPooling'>
            
            # === THE REAL FIX ===
            # BaseModelOutputWithPooling does NOT return a raw tensor.
            # outputs[0] was giving you a (1, dim) tensor → .tolist() became [[...]] → Qdrant validation error.
            # We now:
            #   1. Prefer the correct named attribute (pooler_output or text_embeds – exactly what your comment wanted)
            #   2. Squeeze the batch dimension
            #   3. Fallback to mean-pooling only if we somehow still have extra dimensions
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb_tensor = outputs.pooler_output          # most common for SigLIP/CLIP text towers
            elif hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
                emb_tensor = outputs.text_embeds            # what your comment was trying to use
            else:
                emb_tensor = outputs[0]                     # fallback (what you were using)

            query_vec = emb_tensor.cpu().numpy()
            
            # Robust flattening – this is what was missing / not working in previous fixes
            query_vec = np.squeeze(query_vec)               # turns (1, dim) → (dim,)
            
            # Safety net: if we ever get last_hidden_state (1, seq_len, dim) instead of pooled embedding
            if query_vec.ndim > 1:
                query_vec = np.mean(query_vec, axis=0)      # simple mean pooling (you can change to last token if you prefer)

            # Normalise (good practice for cosine similarity / Qdrant)
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm

        # Now query_vec is guaranteed to be a flat 1D list of floats
        results = self.indexer.client.query_points(
            collection_name=self.indexer.collection_name,
            query=query_vec.tolist(),   # ← this is now a proper list[float]
            limit=top_k
        ).points
        
        return results