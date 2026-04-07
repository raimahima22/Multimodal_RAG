import os
from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator

def main():
    # Initialize the components
    # This will load the ColQwen2.5 model (this takes a moment)
    indexer = MultimodalIndexer()
    retriever = MultimodalRetriever(indexer)
    generator = MultimodalGenerator()

    # 1. Index everything inside the 'data' folder
    # This method now handles both PDFs and Images automatically.
    print("--- Phase 1: Indexing Data ---")
    indexer.index_all_data("data")

    # 2. Define your query
    # Change this to whatever you want to ask about your documents!
    query = "Summarize the key visual data found in these documents."
    
    print(f"\n--- Phase 2: Searching for: '{query}' ---")
    
    # 3. Retrieve the most relevant visual context
    # ColQwen will find the specific page or image that matches your text
    hits = retriever.search(query, top_k=1)
    
    if hits:
        # print(f"Found relevant match in: {hits[0].payload['source']} (Page {hits[0].payload['page_number']})")
        best_hit = hits[0]
        source = best_hit.payload['source']
        page = best_hit.payload.get('page_number')
        print(f"Found relevant match in: {source} (Page {page})")
        
        # 4. Generate the final answer using the Vision-LLM (Gemini or Groq)
        print("--- Phase 3: Generating Answer ---")
        answer = generator.generate_answer(query, hits[0])
        
        print(f"\n--- FINAL AI ANSWER ---\n{answer}")
    else:
        print("No relevant documents found in the database. Check if your 'data' folder has files!")

if __name__ == "__main__":
    main()