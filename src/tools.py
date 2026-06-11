from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator

# Lazy loading for efficiency
_sbc_indexer = None
_spd_indexer = None
_sbc_retriever = None
_spd_retriever = None
_generator = None

def _get_sbc_retriever():
    global _sbc_indexer, _sbc_retriever
    if _sbc_retriever is None:
        _sbc_indexer = MultimodalIndexer(collection_name="sbc_collection")
        _sbc_retriever = MultimodalRetriever(_sbc_indexer)
    return _sbc_retriever

def _get_spd_retriever():
    global _spd_indexer, _spd_retriever
    if _spd_retriever is None:
        _spd_indexer = MultimodalIndexer(collection_name="spd_collection")
        _spd_retriever = MultimodalRetriever(_spd_indexer)
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
    return answer


def search_spd(query: str) -> str:
    """Search only SPD (Summary Plan Description) documents."""
    retriever = _get_spd_retriever()
    hits = retriever.search(query, top_k=3)
    
    if not hits:
        return "No relevant information found in SPD documents."
    
    generator = _get_generator()
    answer = generator.generate_answer(query, hits)
    return answer