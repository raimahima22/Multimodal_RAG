import sys
import gc
import torch
import gradio as gr

from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(force_reindex: bool = False):
    print("Initializing Multimodal RAG System...\n")

    #initialize components
    indexer = MultimodalIndexer(force_recreate=force_reindex)
    retriever = MultimodalRetriever(indexer)
    generator = MultimodalGenerator()

    print("Warming up model...")
    _ = retriever._extract_text_embedding("warmup query")
    print("Ready!")

    # INDEXING
    print("--- Phase 1: Checking Index ---")

    if force_reindex:
        print("Force reindexing...")
        indexer.index_all_data("data")
    elif indexer.is_collection_empty():
        print("Collection is empty → indexing...")
        indexer.index_all_data("data")
    else:
        print("Collection already has data. Skipping indexing.\n")

    print("System is ready!\n")

    
    # SOURCE MAP
   
    source_map = {
        "sbc": "data/sbc.pdf",
        "spd": "data/spd.pdf"
    }

  
    # GRADIO FUNCTION
  
    def answer_query(query, source_input):

        query = query.strip()
        source_input = source_input.strip().lower()

        if not query:
            return "Please enter a question."

        # === SOURCE FILTER LOGIC (kept exactly as you wanted) ===
        if source_input in ["sbc", "sbc.pdf"]:
            source_filter = "data/sbc.pdf"
        elif source_input in ["spd", "spd.pdf"]:
            source_filter = "data/spd.pdf"
        else:
            source_filter = None

        print(f"\nSearching for: {query}")
        print(f"Searching in: {'Both documents' if source_filter is None else source_filter}")

        hits = retriever.search(query, top_k=15, source_filter=source_filter)

        if not hits:
            return "No relevant documents found."

        best_hit = hits[0]
        # context_hits = hits[:5]

        source = best_hit.payload.get('source', 'Unknown')
        page = best_hit.payload.get('page_number', 'N/A')

        print(f"Found match: {source} (Page {page})")
        print("Generating answer...")

        answer = generator.generate_answer(query, best_hit)

        aggressive_cleanup()

        return f"""
 Source: {source} (Page {page})

 Filter: {"Both documents" if source_filter is None else source_filter}

 Answer:
{answer}
"""


    # GRADIO UI
  
    iface = gr.Interface(
        fn=answer_query,
        inputs=[
            gr.Textbox(label="Query", placeholder="Ask your question..."),
            gr.Textbox(label="Source (optional: sbc / spd)", placeholder="e.g. sbc")
        ],
        outputs=gr.Textbox(label="Response"),
        title="Multimodal RAG System",
        description="Ask questions with optional source filtering (sbc / spd)"
    )

    iface.launch(share=True)


# ENTRY POINT

if __name__ == "__main__":
    force_reindex = "--reindex" in sys.argv or "-r" in sys.argv

    if force_reindex:
        print("Reindex mode activated\n")

    main(force_reindex=force_reindex)