import sys
import gc
import torch
import ipywidgets as widgets
from IPython.display import display

from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(force_reindex: bool = False):

    print(" Initializing Multimodal RAG System...\n")

    # =========================
    # INIT COMPONENTS
    # =========================
    indexer = MultimodalIndexer(force_recreate=force_reindex)
    retriever = MultimodalRetriever(indexer)
    generator = MultimodalGenerator()

    print(" Warming up model...")
    _ = retriever._extract_text_embedding("warmup query")
    print(" Ready!\n")

    # =========================
    # INDEXING PHASE
    # =========================
    print("--- Phase 1: Checking Index ---")

    if force_reindex:
        print("Force reindexing...")
        indexer.index_all_data("data")

    elif indexer.is_collection_empty():
        print(" Collection empty → indexing...")
        indexer.index_all_data("data")

    else:
        print(" Collection already exists. Skipping indexing.\n")

    print(" System ready for queries!\n")

    # =========================
    # COLAB UI (IPYWIDGETS)
    # =========================

    text_box = widgets.Text(
        placeholder="Ask your question here...",
        description="Query:",
        layout=widgets.Layout(width="70%")
    )

    button = widgets.Button(
        description="Ask",
        button_style="success"
    )

    output = widgets.Output()

    def on_click(_):

        with output:
            output.clear_output()

            query = text_box.value.strip()

            if not query:
                print(" Please enter a question.")
                return

            print(f" Searching for: {query}\n")

            hits = retriever.search(query, top_k=15)

            if not hits:
                print(" No relevant documents found.")
                return

            best_hit = hits[0]

            source = best_hit.payload.get("source", "Unknown")
            page = best_hit.payload.get("page_number", "N/A")

            print(f" Best match: {source} (Page {page})\n")
            print(" Generating answer...\n")

            answer = generator.generate_answer(query, best_hit)

            print("\n--- FINAL ANSWER ---\n")
            print(answer)
            print("\n" + "-" * 60 + "\n")

            aggressive_cleanup()

    button.on_click(on_click)

    display(text_box, button, output)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":

    force_reindex = "--reindex" in sys.argv or "-r" in sys.argv

    if force_reindex:
        print(" Reindex mode activated\n")

    main(force_reindex=force_reindex)