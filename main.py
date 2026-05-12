import sys
import gc
import torch
import gradio as gr
from datetime import datetime
from src.utils import clear_page_cache
import os
import json
from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator

HISTORY_FILE = "chat_history.json"

def save_to_history(query, source_input, answer):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    
    history.append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "source": source_input,
        "answer": answer
    })
    
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

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
    # Source options (kept simple like your original)
    source_options = {
        "Both Documents": None,
        "SBC": "data/sbc.pdf",
        "SPD": "data/spd.pdf"
    }

    # ====================== MAIN CHAT FUNCTION ======================
    def answer_query(message: str, history: list, source_choice: str):
        """This function is called after user message is added to history"""
        if not message or not message.strip():
            return history, ""

        query = message.strip()
        # Convert dropdown choice to your original filter logic
        source_input = source_choice.lower() if source_choice != "Both Documents" else ""

        if source_input in ["sbc", "sbc.pdf"]:
            source_filter = "data/sbc.pdf"
        elif source_input in ["spd", "spd.pdf"]:
            source_filter = "data/spd.pdf"
        else:
            source_filter = None

        print(f"\nSearching for: {query}")
        print(f"Searching in: {'Both documents' if source_filter is None else source_filter}")

        try:
            hits = retriever.search(query, top_k=3, source_filter=source_filter)

            if not hits:
                response = "No relevant documents found."
            else:
                context_hits = sorted(hits, key=lambda x: x.score, reverse=True)
                best_hit = context_hits[0]

                source = best_hit.payload.get('source', 'Unknown')
                page = best_hit.payload.get('page_number', 'N/A')

                print(f"Found match: {source} (Page {page})")
                print("Generating answer...")

                answer = generator.generate_answer(query, context_hits)

                response = f"""**Source**: {source} (Page {page})
**Filter**: {"Both documents" if source_filter is None else source_filter}

{answer}"""

                save_to_history(query, source_choice, answer)

            # Append assistant response
            history.append((query, response))
            return history, ""

        except Exception as e:
            print(f"Error during query: {e}")
            error_response = f"An error occurred: {str(e)}"
            history.append((query, error_response))
            return history, ""
        finally:
            clear_page_cache()
            aggressive_cleanup()

    # ====================== GRADIO UI ======================
    with gr.Blocks(title="Multimodal RAG Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 Multimodal RAG Chatbot")
        gr.Markdown("Ask questions about SBC and SPD documents")

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    height=650,
                    type="tuples",
                    show_copy_button=True,
                    allow_tags=False
                )
                msg = gr.Textbox(
                    placeholder="Type your question here and press Enter...",
                    label="Your Question",
                    scale=7
                )

            with gr.Column(scale=1):
                gr.Markdown("**Document Filter**")
                source_dropdown = gr.Dropdown(
                    choices=list(source_options.keys()),
                    value="Both Documents",
                    label="Search in",
                    info="sbc / spd / both"
                )
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

        # ====================== EVENT HANDLING ======================
        def add_user_message(message, history):
            if not message:
                return "", history
            history = history + [(message, None)]   # None = waiting for bot response
            return "", history

        # Submit flow
        msg.submit(
            fn=add_user_message,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        ).then(
            fn=answer_query,
            inputs=[msg, chatbot, source_dropdown],   # Note: msg is already cleared, but we use the one from previous step
            outputs=[chatbot, msg]
        )

        clear_btn.click(
            fn=lambda: [],
            inputs=None,
            outputs=chatbot,
            queue=False
        )

        gr.Markdown("---\nBuilt with Qdrant + Multimodal RAG")

    try:
        demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
    finally:
        print("Shutting down Qdrant client...")
        try:
            indexer.local_client.close()
        except Exception:
            pass


if __name__ == "__main__":
    force_reindex = "--reindex" in sys.argv or "-r" in sys.argv
    if force_reindex:
        print("🔄 Reindex mode activated\n")
    main(force_reindex=force_reindex)

    
#     # SOURCE MAP
   
#     source_map = {
#         "sbc": "data/sbc.pdf",
#         "spd": "data/spd.pdf"
#     }

  
#     # GRADIO FUNCTION
  
#     def answer_query(query, source_input):

#         query = query.strip()
#         source_input = source_input.strip().lower()

#         if not query:
#             return "Please enter a question."

#         # === SOURCE FILTER LOGIC (kept exactly as you wanted) ===
#         if source_input in ["sbc", "sbc.pdf"]:
#             source_filter = "data/sbc.pdf"
#         elif source_input in ["spd", "spd.pdf"]:
#             source_filter = "data/spd.pdf"
#         else:
#             source_filter = None

#         print(f"\nSearching for: {query}")
#         print(f"Searching in: {'Both documents' if source_filter is None else source_filter}")


#         try:
#             hits = retriever.search(query, top_k=3, source_filter=source_filter)

#             if not hits:
#                 return "No relevant documents found."

#             # context_hits = hits
#             context_hits = sorted(hits, key=lambda x: x.score, reverse=True)
#             best_hit = context_hits[0]

#             source = best_hit.payload.get('source', 'Unknown')
#             page = best_hit.payload.get('page_number', 'N/A')

#             print(f"Found match: {source} (Page {page})")
#             print("Generating answer...")

#             answer = generator.generate_answer(query, context_hits)
#             save_to_history(query, source_input, answer)

#             return f"""
# Source: {source} (Page {page})
# Filter: {"Both documents" if source_filter is None else source_filter}

# Answer:
# {answer}
# """
#         except Exception as e:
#             print(f"Error during query: {e}")
#             return f"An error occurred: {str(e)}"
#         finally:
#             clear_page_cache()
#             aggressive_cleanup()

#     # GRADIO UI
  
#     iface = gr.Interface(
#         fn=answer_query,
#         inputs=[
#             gr.Textbox(label="Query", placeholder="Ask your question..."),
#             gr.Textbox(label="Source (optional: sbc / spd)", placeholder="e.g. sbc")
#         ],
#         outputs=gr.Textbox(label="Response", lines=15, max_lines=25),
#         title="Multimodal RAG System",
#         description="Ask questions with optional source filtering (sbc / spd)"
#     )
#     try:

#         iface.launch(share=True)
#     finally:
#         print("Shutting down Qdrant client...")
#         try:
#             indexer.local_client.close()
#         except Exception:
#             pass



# ENTRY POINT

if __name__ == "__main__":
    force_reindex = "--reindex" in sys.argv or "-r" in sys.argv

    if force_reindex:
        print("Reindex mode activated\n")

    main(force_reindex=force_reindex)