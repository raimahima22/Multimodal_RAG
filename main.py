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
    #  SOURCE OPTIONS
    source_options = {
        "Both Documents": None,
        "SBC": "data/sbc.pdf",
        "SPD": "data/spd.pdf"
    }

    # CHAT FUNCTION 
    def answer_query(message: str, history: list, source_choice: str):
        if not message or not message.strip():
            return history, "Please enter a question."

        query = message.strip()
        source_filter = source_options.get(source_choice)

        print(f"\n Query: {query}")
        print(f" Source: {'Both Documents' if source_filter is None else source_filter}")

        try:
            hits = retriever.search(query, top_k=3, source_filter=source_filter)

            if not hits:
                response = "No relevant documents found."
            else:
                context_hits = sorted(hits, key=lambda x: x.score, reverse=True)
                best_hit = context_hits[0]

                source = best_hit.payload.get('source', 'Unknown')
                page = best_hit.payload.get('page_number', 'N/A')

                answer = generator.generate_answer(query, context_hits)

                response = f"""**Source**: {source} (Page {page})  
**Filter**: {"Both documents" if source_filter is None else source_filter}

{answer}"""

                save_to_history(query, source_choice, answer)

            history.append((query, response))
            return history, ""

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            print(f" Error: {e}")
            history.append((query, error_msg))
            return history, ""
        finally:
            clear_page_cache()
            aggressive_cleanup()

    # ====================== GRADIO INTERFACE (Fixed) ======================
    with gr.Blocks(
        title="Multimodal RAG Chat",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("# Multimodal RAG Chatbot")
        gr.Markdown("Ask intelligent questions about **SBC** and **SPD** documents.")

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    height=650,
                    show_copy_button=True,
                    type="tuples",           # Explicitly set to avoid warning
                    allow_tags=False
                )
                
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    label="Your Question",
                    scale=8
                )

            with gr.Column(scale=1):
                gr.Markdown("###  Document Filter")
                source_dropdown = gr.Dropdown(
                    choices=list(source_options.keys()),
                    value="Both Documents",
                    label="Search Scope",
                    info="Choose which document to search"
                )

                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

                gr.Markdown("### Examples")
                gr.Examples(
                    examples=[
                        ["What is the main objective of the SBC?", "SBC"],
                        ["Summarize the key points of SPD", "SPD"],
                        ["Compare requirements between SBC and SPD", "Both Documents"],
                    ],
                    inputs=[msg, source_dropdown],
                    label="Quick Start"
                )

        # Submit logic
        def user_submit(message, history):
            if not message:
                return "", history
            history = history + [(message, None)]
            return "", history

        msg.submit(
            user_submit, 
            inputs=[msg, chatbot], 
            outputs=[msg, chatbot]
        ).then(
            answer_query,
            inputs=[msg, chatbot, source_dropdown],
            outputs=[chatbot, msg]
        )

        clear_btn.click(
            fn=lambda: [], 
            inputs=None, 
            outputs=chatbot, 
            queue=False
        )

        gr.Markdown("---\nBuilt with Qdrant • Multimodal RAG")

    # Launch
    try:
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
    finally:
        print("Shutting down Qdrant client...")
        try:
            indexer.local_client.close()
        except:
            pass

    
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