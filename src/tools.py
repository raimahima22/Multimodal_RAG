from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator

# Lazy loading for efficiency
_shared_indexers = {}
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
    """Get or create indexer for a specific collection."""
    global _shared_indexers
    
    if collection_name not in _shared_indexers:
        print(f"   → Creating indexer for collection: {collection_name}")
        _shared_indexers[collection_name] = MultimodalIndexer(
            collection_name=collection_name
        )
    else:
        print(f"   → Reusing existing indexer for: {collection_name}")
    
    return _shared_indexers[collection_name]

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
    print(" Preloading all models and indexers...")
    
    try:
        print("   • Loading Generator...")
        _get_generator()
        
        print("   • Loading SBC Retriever...")
        _get_sbc_retriever()
        
        print("   • Loading SPD Retriever...")
        _get_spd_retriever()
        
        print(" All models and both collections preloaded successfully!")
        
    except Exception as e:
        print(f" Preload error: {e}")