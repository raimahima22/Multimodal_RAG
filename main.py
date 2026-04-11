import sys
from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator


def main(force_reindex: bool = False):
    print(" Initializing Multimodal RAG System...\n")
    
    # Initialize components
    indexer = MultimodalIndexer(force_recreate=force_reindex)
    retriever = MultimodalRetriever(indexer)
    generator = MultimodalGenerator()

    # === Smart Indexing ===
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

        print(f"\n Searching for: '{query}'")
        
        hits = retriever.search(query, top_k=15)
        
        if hits:
            best_hit = hits[0]
            source = best_hit.payload['source']
            page = best_hit.payload.get('page_number', 'N/A')
            
            print(f" Found relevant match in: {source} (Page {page})")
            
            print(" Generating Answer...")
            answer = generator.generate_answer(query, best_hit)
            
            print(f"\n--- FINAL ANSWER ---\n{answer}\n")
            print("-" * 80 + "\n")
        else:
            print(" No relevant documents found.\n")


if __name__ == "__main__":
    force_reindex = "--reindex" in sys.argv or "-r" in sys.argv
    
    if force_reindex:
        print(" Reindex mode activated\n")
    
    main(force_reindex=force_reindex)