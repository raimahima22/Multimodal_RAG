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

    # =========================
    # INIT COMPONENTS
    # =========================
    indexer = MultimodalIndexer(force_recreate=force_reindex)
    retriever = MultimodalRetriever(indexer)
    generator = MultimodalGenerator()

    print("Warming up model...")
    _ = retriever._extract_text_embedding("warmup query")
    print("Ready!\n")


    # INDEXING PHASE
  
    print("--- Phase 1: Checking Index ---")

    if force_reindex:
        print("Force reindexing...")
        indexer.index_all_data("data")

    elif indexer.is_collection_empty():
        print("Collection empty → indexing...")
        indexer.index_all_data("data")

    else:
        print("Collection already exists. Skipping indexing.\n")

    print("System ready for queries!\n")

    # GRADIO FUNCTION

    def answer_query(query):

        if not query.strip():
            return "Please enter a question."

        print(f"Searching for: {query}")

        hits = retriever.search(query, top_k=15)

        if not hits:
            return "No relevant documents found."

        best_hit = hits[0]

        source = best_hit.payload.get("source", "Unknown")
        page = best_hit.payload.get("page_number", "N/A")

        print(f"Best match: {source} (Page {page})")

        answer = generator.generate_answer(query, best_hit)

        aggressive_cleanup()

        return f"""
 Source: {source} (Page {page})

 Answer:
{answer}
"""

    # GRADIO UI
    iface = gr.Interface(
        fn=answer_query,
        inputs=gr.Textbox(
            placeholder="Ask your question here...",
            label="Query"
        ),
        outputs=gr.Textbox(label="Response"),
        title="Multimodal RAG System",
        description="Ask questions about your indexed PDFs/images"
    )

    iface.launch(share=True)


# ENTRY POINT

if __name__ == "__main__":

    force_reindex = "--reindex" in sys.argv or "-r" in sys.argv

    if force_reindex:
        print("Reindex mode activated\n")

    main(force_reindex=force_reindex)