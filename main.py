import sys
import gc
import torch
from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator

def aggressive_cleanup():
    
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main(force_reindex: bool = False):
    print(" Initializing Multimodal RAG System...\n")
    
    # Initialize components
    indexer = MultimodalIndexer(force_recreate=force_reindex)
    retriever = MultimodalRetriever(indexer)
    generator = MultimodalGenerator()

    # Smart Indexing 
    print("--- Phase 1: Checking Index ---")
    
    if force_reindex:
        print(" Force reindexing requested...")
        indexer.index_all_data("data")
    elif indexer.is_collection_empty():
        print(" Collection is empty → Starting indexing...")
        indexer.index_all_data("data")
    else:
        print(" Collection already has data. Skipping indexing.\n")

    print(" System is ready! You can now ask questions.")
    print("Type 'exit', 'quit', or 'q' to stop.\n")

    while True:
        query = input(" Your Question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q', '']:
            print(" Goodbye!")
            break
        
        source_input = input("Source (sbc / spd ): ").strip().lower()

        #map input to the filter
        if source_input in ["sbc", "sbc.pdf"]:
            source_filter = "data/sbc.pdf"
        elif source_input in ["spd", "spd.pdf"]:
            source_filter = "data/spd.pdf"
        else:
            source_filter = None
        
        # print(f"\n Searching for: '{query}'")
        print(f"\n Searching in: {'Both documents' if source_filter is None else source_filter}")
        
        hits = retriever.search(query, top_k=15, source_filter=source_filter)
        
        if hits:
            best_hit = hits[0]
            # source = best_hit.payload['source', 'Unknown']
            source = best_hit.payload.get('source', 'Unknown')
            page = best_hit.payload.get('page_number', 'N/A')
            
            print(f" Found relevant match in: {source} (Page {page})")
            
            print(" Generating Answer...")
            answer = generator.generate_answer(query, best_hit)
            
            print(f"\n--- FINAL ANSWER ---\n{answer}\n")
            print("-" * 80 + "\n")
            aggressive_cleanup()
        else:
            print(" No relevant documents found.\n")


if __name__ == "__main__":
    force_reindex = "--reindex" in sys.argv or "-r" in sys.argv
    
    if force_reindex:
        print(" Reindex mode activated\n")
    
    main(force_reindex=force_reindex)