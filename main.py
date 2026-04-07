import os
from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator

def main():
    print(" Initializing Multimodal RAG System...\n")
    
    # Initialize components (done only once)
    indexer = MultimodalIndexer()
    retriever = MultimodalRetriever(indexer)
    generator = MultimodalGenerator()

    # Index the data (done only once)
    print("--- Phase 1: Indexing Data ---")
    indexer.index_all_data("data")
    print(" Indexing completed!\n")
    

    print(" System is ready! You can now ask questions.")
    print("Type 'exit', 'quit', or 'q' to stop.\n")

    while True:
        # Get query from user
        query = input(" Your Question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q', '']:
            print(" Goodbye!")
            break

        print(f"\n Searching for: '{query}'")
        
        # Retrieve relevant content
        hits = retriever.search(query, top_k=3)
        
        if hits:
            best_hit = hits[0]
            source = best_hit.payload['source']
            page = best_hit.payload.get('page_number')
            
            print(f" Found relevant match in: {source} (Page {page})")
            print(f" Relevance Score: {best_hit.score:.4f}\n")
            
            # Generate answer
            print(" Generating Answer...")
            answer = generator.generate_answer(query, best_hit)
            
            print(f"\n--- FINAL ANSWER ---\n")
            print(answer)
            print("-" * 80 + "\n")
            
        else:
            print(" No relevant documents found.\n")


if __name__ == "__main__":
    main()