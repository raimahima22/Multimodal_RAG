# from src.indexer import MultimodalIndexer
# from src.retriever import MultimodalRetriever
# from src.generator import MultimodalGenerator

# # Lazy loading for efficiency
# _shared_indexer = None
# _sbc_retriever = None
# _spd_retriever = None
# _generator = None

# def normalize(x):
#     if isinstance(x, list):
#         return " ".join(map(str, x))
#     if isinstance(x, dict):
#         return x.get("content", str(x))
#     return str(x)

# def _get_shared_indexer(collection_name: str):
#     global _shared_indexer
#     if _shared_indexer is None:
#         _shared_indexer = MultimodalIndexer(collection_name=collection_name)  # reuse model
#     return _shared_indexer

# def _get_sbc_retriever():
#     global _sbc_retriever
#     if _sbc_retriever is None:
#         indexer = _get_shared_indexer("sbc_collection")
#         _sbc_retriever = MultimodalRetriever(indexer)
#     return _sbc_retriever

# def _get_spd_retriever():
#     global _spd_retriever
#     if _spd_retriever is None:
#         indexer = _get_shared_indexer("spd_collection")
#         _spd_retriever = MultimodalRetriever(indexer)
#     return _spd_retriever

# def _get_generator():
#     global _generator
#     if _generator is None:
#         _generator = MultimodalGenerator()
#     return _generator


# def search_sbc(query: str) -> str:
#     """Search only SBC (Summary of Benefits and Coverage) documents."""
#     retriever = _get_sbc_retriever()
#     hits = retriever.search(query, top_k=3)
    
#     if not hits:
#         return "No relevant information found in SBC documents."
    
#     generator = _get_generator()
#     answer = generator.generate_answer(query, hits)
#     #normalize output
#     answer = normalize(answer)

#     return answer


# def search_spd(query: str) -> str:
#     """Search only SPD (Summary Plan Description) documents."""
#     retriever = _get_spd_retriever()
#     hits = retriever.search(query, top_k=3)
    
#     if not hits:
#         return "No relevant information found in SPD documents."
    
#     generator = _get_generator()
#     answer = generator.generate_answer(query, hits)
#     #normalize output
#     answer = normalize(answer)

#     return answer

# src/tools.py
"""
Lazy-loaded retrieval tools for the LangGraph agent.

Two independent pipelines are maintained so SBC and SPD data
are always kept in separate Qdrant collections.
"""

from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator

# ── Module-level singletons ───────────────────────────────────────────────────

_sbc_indexer: MultimodalIndexer | None = None
_spd_indexer: MultimodalIndexer | None = None
_sbc_retriever: MultimodalRetriever | None = None
_spd_retriever: MultimodalRetriever | None = None
_generator: MultimodalGenerator | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize(x) -> str:
    """Ensure tool output is always a plain string."""
    if isinstance(x, list):
        return " ".join(map(str, x))
    if isinstance(x, dict):
        return x.get("content", str(x))
    return str(x)


def _get_sbc_retriever() -> MultimodalRetriever:
    """Return (or create) the SBC retriever backed by its own indexer."""
    global _sbc_indexer, _sbc_retriever
    if _sbc_retriever is None:
        # Each collection gets its own indexer instance so the underlying
        # Qdrant client points to the correct collection.
        _sbc_indexer = MultimodalIndexer(collection_name="sbc_collection")
        _sbc_retriever = MultimodalRetriever(_sbc_indexer)
    return _sbc_retriever


def _get_spd_retriever() -> MultimodalRetriever:
    """Return (or create) the SPD retriever backed by its own indexer."""
    global _spd_indexer, _spd_retriever
    if _spd_retriever is None:
        _spd_indexer = MultimodalIndexer(collection_name="spd_collection")
        _spd_retriever = MultimodalRetriever(_spd_indexer)
    return _spd_retriever


def _get_generator() -> MultimodalGenerator:
    global _generator
    if _generator is None:
        _generator = MultimodalGenerator()
    return _generator


# ── Public tool functions ─────────────────────────────────────────────────────

def search_sbc(query: str) -> str:
    """Search only SBC (Summary of Benefits and Coverage) documents."""
    retriever = _get_sbc_retriever()
    hits = retriever.search(query, top_k=3)

    if not hits:
        return "No relevant information found in SBC documents."

    generator = _get_generator()
    answer = generator.generate_answer(query, hits)
    return normalize(answer)


def search_spd(query: str) -> str:
    """Search only SPD (Summary Plan Description) documents."""
    retriever = _get_spd_retriever()
    hits = retriever.search(query, top_k=3)

    if not hits:
        return "No relevant information found in SPD documents."

    generator = _get_generator()
    answer = generator.generate_answer(query, hits)
    return normalize(answer)