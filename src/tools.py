from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator

# Lazy loading for efficiency
_shared_indexer = None
_sbc_retriever = None
_spd_retriever = None
_generator = None

def normalize(x):
    if isinstance(x, list):
        return " ".join(map(str, x))
    if isinstance(x, dict):
        return x.get("content", str(x))
    return str(x)

def _get_shared_indexer(collection_name: str):
    global _shared_indexer
    if _shared_indexer is None:
        _shared_indexer = MultimodalIndexer(collection_name=collection_name)  # reuse model
    return _shared_indexer

def _get_sbc_retriever():
    global _sbc_retriever
    if _sbc_retriever is None:
        indexer = _get_shared_indexer("sbc_collection")
        _sbc_retriever = MultimodalRetriever(indexer)
    return _sbc_retriever

def _get_spd_retriever():
    global _spd_retriever
    if _spd_retriever is None:
        indexer = _get_shared_indexer("spd_collection")
        _spd_retriever = MultimodalRetriever(indexer)
    return _spd_retriever

def _get_generator():
    global _generator
    if _generator is None:
        _generator = MultimodalGenerator()
    return _generator


def search_sbc(query: str) -> str:
    """Search only SBC (Summary of Benefits and Coverage) documents."""
    retriever = _get_sbc_retriever()
    hits = retriever.search(query, top_k=3)
    
    if not hits:
        return "No relevant information found in SBC documents."
    
    generator = _get_generator()
    answer = generator.generate_answer(query, hits)
    #normalize output
    answer = normalize(answer)

    return answer


def search_spd(query: str) -> str:
    """Search only SPD (Summary Plan Description) documents."""
    retriever = _get_spd_retriever()
    hits = retriever.search(query, top_k=3)
    
    if not hits:
        return "No relevant information found in SPD documents."
    
    generator = _get_generator()
    answer = generator.generate_answer(query, hits)
    #normalize output
    answer = normalize(answer)

    return answer

# Add this at the bottom of src/tools.py (after all existing functions)

def preload_all_models():
    """Preload all models, indexers, retrievers and generator at startup."""
    print(" Preloading all models and indexers... This may take 10-30 seconds.")
    
    try:
        # Preload Generator + LLM
        print("   • Loading MultimodalGenerator (LLM)...")
        generator = _get_generator()
        
        # Preload both retrievers (this loads indexers + embeddings)
        print("   • Loading SBC Retriever...")
        _get_sbc_retriever()
        
        print("   • Loading SPD Retriever...")
        _get_spd_retriever()
        
        # Optional: Do a small dummy search to warm up embeddings and Qdrant
        print("   • Warming up retrieval...")
        dummy_query = "What is the deductible?"
        
        sbc_ret = _get_sbc_retriever()
        spd_ret = _get_spd_retriever()
        
        _ = sbc_ret.search(dummy_query, top_k=1)
        _ = spd_ret.search(dummy_query, top_k=1)
        
        print(" All models preloaded successfully!")
        
    except Exception as e:
        print(f" Preloading failed (some models may load on first use): {e}")